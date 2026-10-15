"""Phase 3D / task B: soft-voting and stacking ensembles from cached D3 scores
(no retraining). N-B TOST (+/-0.005) and one-sided superiority vs best single."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sst
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR
from creditrisk.calibration import Isotonic, prob_metrics
from creditrisk.repeats import get_dataset, load_score
from creditrisk.thresholds import youden_threshold
from stats_study import nb, holm, MARGIN, RATIO

MODELS = ['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def one_sided_superior(d, margin=MARGIN, ratio=RATIO):
    """p-value for H0: delta <= +margin (superiority by more than margin)."""
    d = np.asarray(d, float); J = len(d); se = np.sqrt((1 / J + ratio) * d.var(ddof=1))
    return 1 - sst.t.cdf((d.mean() - margin) / se, J - 1)


def metrics(y_te, p_te, auc_score, y_tv, p_oof):
    pm = prob_metrics(y_te, p_te)
    t = youden_threshold(y_tv, p_oof)
    return dict(ROC_AUC=roc_auc_score(y_te, auc_score), PR_AUC=average_precision_score(y_te, auc_score),
                Brier=pm['Brier'], ECE=pm['ECE'], t_star=t, pi=float(y_tv.mean()), abs_t_pi=abs(t - y_tv.mean()))


def main():
    X, y, _ = get_dataset('D3'); yv = y.values
    rows = []
    for rep in range(10):
        Z = {m: load_score('D3', rep, m, 'none') for m in MODELS}
        tv, te = Z[MODELS[0]]['idx_tv'], Z[MODELS[0]]['idx_te']
        for m in MODELS:                                  # identical splits across models
            assert (Z[m]['idx_tv'] == tv).all() and (Z[m]['idx_te'] == te).all(), (rep, m)
        y_tv, y_te = yv[tv], yv[te]
        cal_oof, cal_te = {}, {}
        for m in MODELS:
            o, t_ = Z[m]['oof'].astype(float), Z[m]['te'].astype(float)
            iso = Isotonic().fit(o, y_tv); cal_oof[m], cal_te[m] = iso.predict(o), iso.predict(t_)
            r = metrics(y_te, cal_te[m], t_, y_tv, cal_oof[m]); r.update(rep=rep, method=m, kind='single')
            rows.append(r)
        v_oof = np.mean([cal_oof[m] for m in MODELS], axis=0); v_te = np.mean([cal_te[m] for m in MODELS], axis=0)
        r = metrics(y_te, v_te, v_te, y_tv, v_oof); r.update(rep=rep, method='Soft voting', kind='ensemble'); rows.append(r)
        Zo = np.column_stack([logit(Z[m]['oof'].astype(float)) for m in MODELS])
        Zt = np.column_stack([logit(Z[m]['te'].astype(float)) for m in MODELS])
        meta = LogisticRegression(C=1.0, max_iter=1000)
        m_oof = cross_val_predict(meta, Zo, y_tv, cv=StratifiedKFold(5, shuffle=True, random_state=42 + rep),
                                  method='predict_proba')[:, 1]
        meta.fit(Zo, y_tv); m_te = meta.predict_proba(Zt)[:, 1]
        iso = Isotonic().fit(m_oof, y_tv)
        r = metrics(y_te, iso.predict(m_te), m_te, y_tv, iso.predict(m_oof)); r.update(rep=rep, method='Stacking (LR meta)', kind='ensemble')
        rows.append(r)
    D = pd.DataFrame(rows); D.round(5).to_csv(RESULTS_DIR / "phase3d_ensembles.csv", index=False)
    pd.set_option('display.width', 250)
    S = D.groupby('method')[['ROC_AUC', 'PR_AUC', 'Brier', 'ECE', 't_star', 'abs_t_pi']].mean().round(4)
    print(S.sort_values('ROC_AUC', ascending=False).to_string())
    A = D.pivot(index='rep', columns='method', values='ROC_AUC')
    best = A[MODELS].mean().idxmax()
    print(f"\nbest single: {best} ({A[best].mean():.4f})")
    tests = []
    for ens in ('Soft voting', 'Stacking (LR meta)'):
        d = (A[ens] - A[best]).values; r = nb(d); r.update(ensemble=ens, vs=best, family='H17', p_sup=one_sided_superior(d))
        tests.append(r)
    T = pd.DataFrame(tests)
    T['tost_p_holm'] = holm(T.tost_p.values); T['sup_p_holm'] = holm(T.p_sup.values)
    T['equivalent'] = T.tost_p_holm < .05; T['superior_by_gt_margin'] = T.sup_p_holm < .05
    T['label'] = np.where(T.equivalent, 'equivalent', np.where((T.ci_lo <= 0) & (T.ci_hi >= 0), 'INCONCLUSIVE', 'not equivalent'))
    print("\nH17 family (vs best single):\n", T[['ensemble', 'vs', 'mean_delta', 'ci_lo', 'ci_hi', 'tost_p_holm', 'equivalent',
                                                'sup_p_holm', 'superior_by_gt_margin', 'label']].round(4).to_string(index=False))
    mem = []
    for ens in ('Soft voting', 'Stacking (LR meta)'):
        for m in MODELS:
            d = (A[ens] - A[m]).values; r = nb(d); r.update(ensemble=ens, vs=m, p_sup=one_sided_superior(d)); mem.append(r)
    M = pd.DataFrame(mem); M['tost_p_holm'] = holm(M.tost_p.values); M['sup_p_holm'] = holm(M.p_sup.values)
    M['equivalent'] = M.tost_p_holm < .05
    M['label'] = np.where(M.equivalent, 'equivalent', np.where((M.ci_lo <= 0) & (M.ci_hi >= 0), 'INCONCLUSIVE', 'not equivalent'))
    print("\nvs each member (Holm over 10):\n", M[['ensemble', 'vs', 'mean_delta', 'ci_lo', 'ci_hi', 'tost_p_holm', 'equivalent', 'label']].round(4).to_string(index=False))
    pd.concat([T.assign(family='H17'), M.assign(family='members')]).round(5).to_csv(RESULTS_DIR / "phase3d_ensembles_tests.csv", index=False)
    ok = bool(T.equivalent.all()) and not bool(T.superior_by_gt_margin.any())
    print(f"\nH17 verdict: {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    # ensemble value: Brier / ECE comparison
    B = D.pivot(index='rep', columns='method', values='Brier')
    print("Brier delta vs best single (ensemble - best):", {e: round((B[e] - B[best]).mean(), 5) for e in ('Soft voting', 'Stacking (LR meta)')})


if __name__ == "__main__":
    main()
