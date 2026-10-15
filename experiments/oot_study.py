"""Phase 3A / task D: rolling-origin out-of-time study on issue year.

For test year T in {2013, 2014, 2015}: fit on issues <= T-2, calibrate + pick
thresholds on year T-1, test on year T. All terms for T=2013/2014; 36-month
loans only for T=2015 (60-month 2015+ loans are mostly censored by 2018Q4).
Strict label. Training windows > 400k rows are seeded-subsampled.
Resumable: one pickle per (T, model) in .cache/oot/.
"""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR, CACHE_DIR
from creditrisk.calibration import make_calibrators, apply_cal, prob_metrics
from creditrisk.data import full_frame, label_strict, issue_year, term_months, profit_p1, profit_p2
from creditrisk.features import build_frame
from creditrisk.models import fit_model
from creditrisk.stats import paired_mean_diff_ci
from creditrisk import thresholds as T_

MODELS = ['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']
FOLDS = [2013, 2014, 2015]
SCEN = ['P1', 'P2_3%']
OOT_DIR = CACHE_DIR / "oot3b"
SEED = 42
MISSING_MAX = 0.50          # drop features missing in >50% of the training window


def fold_indices(Tyr, yr, tm):
    keep = np.ones(len(yr), bool) if Tyr < 2015 else (tm == 36)
    tr = np.where(keep & (yr <= Tyr - 2))[0]
    ca = np.where(keep & (yr == Tyr - 1))[0]
    te = np.where(keep & (yr == Tyr))[0]
    if len(tr) > 400_000:
        tr = np.sort(np.random.RandomState(SEED).choice(tr, 400_000, replace=False))
    return tr, ca, te


def usable_columns(X, tr):
    miss = X.iloc[tr].isna().mean()
    return [c for c in X.columns if miss[c] <= MISSING_MAX], [c for c in X.columns if miss[c] > MISSING_MAX]


def make_data():
    d = label_strict(full_frame())
    d['year'] = issue_year(d).values
    d['term_m'] = term_months(d).values
    d = d[d.year <= 2015]                      # vintages >= 2016 dropped from headline
    X, y, _ = build_frame(d, True, 'fico')
    P = {'P1': profit_p1(d).values, 'P2_3%': profit_p2(d, 0.03).values}
    return X, y.values, P, d.year.values, d.term_m.values


def auc_ci(y, s, n=200, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v.append(roc_auc_score(y[i], s[i]))
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def run_unit(Tyr, model, X, y, P, yr, tm, n_jobs, seed=SEED):
    path = OOT_DIR / f"{Tyr}_{model.replace(' ', '_')}_s{seed}.pkl"
    if path.exists():
        return pickle.load(open(path, 'rb'))
    t0 = time.time()
    tr, ca, te = fold_indices(Tyr, yr, tm)
    cols, dropped = usable_columns(X, tr)
    X = X[cols]
    m = fit_model(model, X.iloc[tr], pd.Series(y[tr]), 'none', seed, n_jobs)
    s_ca, s_te = m.predict_proba(X.iloc[ca])[:, 1], m.predict_proba(X.iloc[te])[:, 1]
    # in-time reference: random 80/20 inside the test vintage
    a, b = train_test_split(te, test_size=0.2, random_state=SEED, stratify=y[te])
    m2 = fit_model(model, X.iloc[a], pd.Series(y[a]), 'none', seed, n_jobs)
    auc_in = roc_auc_score(y[b], m2.predict_proba(X.iloc[b])[:, 1])
    cal = make_calibrators(s_ca, y[ca])['isotonic']
    pi_tr, pi_ca, pi_te = y[tr].mean(), y[ca].mean(), y[te].mean()
    fold, rules = [], []
    for cname, (c_ca, c_te) in {'raw': (s_ca, s_te),
                                'isotonic': (apply_cal(cal, s_ca), apply_cal(cal, s_te))}.items():
        pm_te, pm_ca = prob_metrics(y[te], c_te), prob_metrics(y[ca], c_ca)
        lo, hi = auc_ci(y[te], c_te)
        t_ca, t_te = T_.youden_threshold(y[ca], c_ca), T_.youden_threshold(y[te], c_te)
        jm_te, flat_te = T_.youden_flatness(y[te], c_te)
        flag = c_te >= t_ca
        j_stale = float(flag[y[te] == 1].mean() - flag[y[te] == 0].mean())
        fold.append(dict(T=Tyr, model=model, seed=seed, dropped='|'.join(dropped), calib=cname, n_train=len(tr), n_cal=len(ca), n_test=len(te),
                         pi_train=pi_tr, pi_cal=pi_ca, pi_test=pi_te,
                         AUC_oot=roc_auc_score(y[te], c_te), AUC_oot_lo=lo, AUC_oot_hi=hi,
                         AUC_intime=auc_in, AUC_cal_year=roc_auc_score(y[ca], c_ca),
                         Brier_test=pm_te['Brier'], ECE_test=pm_te['ECE'], LogLoss_test=pm_te['LogLoss'],
                         ECE_cal_insample=pm_ca['ECE'], t_star_cal=t_ca, t_star_test=t_te,
                         flat_width_test=flat_te, J_test_max=jm_te, J_test_at_tcal=j_stale,
                         J_loss=jm_te - j_stale, secs=time.time() - t0))
        for sc in SCEN:
            p_ca, p_te = P[sc][ca], P[sc][te]
            c_fn, c_fp = -p_ca[y[ca] == 1].mean(), p_ca[y[ca] == 0].mean()
            thr = {'approve_all': np.inf, 'fixed_0.5': 0.5, 'youden': t_ca,
                   'f1_max': T_.f1_max_threshold(y[ca], c_ca),
                   'prevalence_train': pi_tr, 'profit_max': T_.profit_max_threshold(c_ca, p_ca),
                   'oracle_youden': t_te, 'oracle_pi_test': pi_te,
                   'oracle_profit_max': T_.profit_max_threshold(c_te, p_te)}
            if cname != 'raw':
                thr['elkan'] = T_.elkan_threshold(c_fp, c_fn)
            for r, t in thr.items():
                o = T_.policy_outcome(c_te, t, y[te], p_te)
                d = p_te * (c_te < t) * 1000
                _, plo, phi, _ = paired_mean_diff_ci(d, 300, seed=1)
                rules.append(dict(T=Tyr, model=model, seed=seed, calib=cname, scenario=sc, rule=r, threshold=t,
                                  profit_per_1000=o['profit_per_1000'], profit_lo=plo, profit_hi=phi,
                                  accept_rate=o['accept_rate'], bad_rate_approved=o['bad_rate_approved']))
    out = (pd.DataFrame(fold), pd.DataFrame(rules))
    OOT_DIR.mkdir(exist_ok=True)
    pickle.dump(out, open(path, 'wb'))
    print(f"[oot] T={Tyr} {model} n_tr={len(tr)} n_cal={len(ca)} n_te={len(te)} {time.time()-t0:.0f}s", flush=True)
    return out


def main(n_jobs=8, seeds=(42,), models=None, tag="phase3b_oot"):
    X, y, P, yr, tm = make_data()
    F, R = [], []
    for Tyr in FOLDS:
        tr, _, _ = fold_indices(Tyr, yr, tm)
        _, dropped = usable_columns(X, tr)
        print(f"[hygiene] T={Tyr}: features missing in >50% of training window: {dropped or 'none'}", flush=True)
        for model in (models or MODELS):
            for sd in seeds:
                f, r = run_unit(Tyr, model, X, y, P, yr, tm, n_jobs, sd)
                F.append(f); R.append(r)
    F, R = pd.concat(F, ignore_index=True), pd.concat(R, ignore_index=True)
    F.round(5).to_csv(RESULTS_DIR / f"{tag}_fold_metrics.csv", index=False)
    R.round(3).to_csv(RESULTS_DIR / f"{tag}.csv", index=False)
    print("done", F.shape, R.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tag", default="phase3b_oot")
    a = ap.parse_args()
    main(a.n_jobs, tuple(a.seeds), a.models, a.tag)
