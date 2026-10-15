"""Phase 3C / task A: equivalence and significance tests on cached scores.
Nadeau-Bengio corrected resampled t-tests (variance factor 1/J + n_te/n_tr),
Holm over the 9 comparisons per dataset/metric, TOST at +/-0.005 AUC, plus a
paired Poisson-bootstrap check on repeat 0 and per-fold OOT bootstraps."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sst
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR
from creditrisk.models import MODEL_NAMES
from creditrisk.repeats import get_dataset, load_score
from creditrisk.study import analyse
import oot_scores as OS
from oot_study import FOLDS

MARGIN = 0.005
RATIO = 0.25           # n_test / n_train (20% test vs final model refit on 80%)
BOOSTING = ['Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost']


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0
    for i, idx in enumerate(o):
        run = max(run, (n - i) * p[idx]); adj[idx] = min(run, 1.0)
    return adj


def nb(d, ratio=RATIO, margin=MARGIN):
    d = np.asarray(d, float); J = len(d); m = d.mean(); s2 = d.var(ddof=1)
    se = np.sqrt((1 / J + ratio) * s2) if s2 > 0 else 1e-12
    df = J - 1; t = m / se; tq = sst.t.ppf(.975, df)
    p = 2 * (1 - sst.t.cdf(abs(t), df))
    p_lo = 1 - sst.t.cdf((m + margin) / se, df)      # H0: diff <= -margin
    p_hi = sst.t.cdf((m - margin) / se, df)          # H0: diff >= +margin
    return dict(mean_delta=m, ci_lo=m - tq * se, ci_hi=m + tq * se, p=p, tost_p=max(p_lo, p_hi))


def boot_auc_diff(y, S, ref, n_boot=1000, seed=0, chunk=50):
    """Paired Poisson bootstrap of AUC(model) - AUC(ref); S: dict name -> scores."""
    y = np.asarray(y).astype(float); n = len(y); rng = np.random.default_rng(seed)
    orders = {m: np.argsort(s, kind='stable') for m, s in S.items()}
    out = {m: [] for m in S}
    for _ in range(0, n_boot, chunk):
        W = rng.poisson(1.0, (chunk, n)).astype(float)
        auc = {}
        for m, o in orders.items():
            w, ys = W[:, o], y[o]
            neg = w * (1 - ys); pos = w * ys
            cneg = np.cumsum(neg, axis=1) - neg
            auc[m] = (pos * cneg).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))
        for m in S:
            out[m].extend(auc[m] - auc[ref])
    return {m: np.array(v) for m, v in out.items()}


def main():
    rows = []
    for ds, n_rep in (('D3', 10), ('D2', 10), ('D1', 5)):
        R, C = analyse(ds, list(range(n_rep)), with_profit=(ds != 'D1'))
        c = C[(C.calib == 'raw') & (C.weighting == 'none')]
        r = R[(R.rule == 'youden') & (R.calib == 'raw') & (R.weighting == 'none')]
        met = {'ROC_AUC': c.pivot(index='rep', columns='model', values='ROC_AUC'),
               'Brier': c.pivot(index='rep', columns='model', values='Brier'),
               'F1_tuned': r.pivot(index='rep', columns='model', values='F1')}
        best = met['ROC_AUC'].mean().idxmax()
        # paired bootstrap on repeat 0 (AUC)
        X, y, _ = get_dataset(ds); yv = y.values
        S = {m: load_score(ds, 0, m, 'none') for m in MODEL_NAMES}
        te = S[best]['idx_te']
        bo = boot_auc_diff(yv[te], {m: S[m]['te'].astype(float) for m in MODEL_NAMES}, best, 1000)
        for metric, M in met.items():
            res = {m: nb((M[m] - M[best]).values) for m in MODEL_NAMES if m != best}
            ms = list(res)
            ph = holm([res[m]['p'] for m in ms]); pt = holm([res[m]['tost_p'] for m in ms])
            for i, m in enumerate(ms):
                row = dict(dataset=ds, metric=metric, model=m, best=best, **res[m], p_holm=ph[i], tost_p_holm=pt[i],
                           equivalent=bool(pt[i] < .05) if metric == 'ROC_AUC' else np.nan,
                           mean_value=M[m].mean(), sd_value=M[m].std(ddof=1), best_value=M[best].mean())
                if metric == 'ROC_AUC':
                    lo, hi = np.quantile(bo[m], [.025, .975]); row.update(boot_lo=lo, boot_hi=hi, boot_mean=bo[m].mean())
                rows.append(row)
    T = pd.DataFrame(rows)
    # ------------------------------------------------------------ OOT: paired bootstrap per fold
    t = OS.oot_table(); yo = t['y']; orows = []
    for Tyr in FOLDS:
        sc = {m: OS.load(Tyr, m, 42) for m in OS.MAIN5}
        te = sc[OS.MAIN5[0]]['idx_te']
        aucs = {m: np.mean([_auc(yo[OS.load(Tyr, m, s)['idx_te']], OS.load(Tyr, m, s)['s_te']) for s in (42, 43, 44)]) for m in OS.MAIN5}
        best = max(aucs, key=aucs.get)
        bo = boot_auc_diff(yo[te], {m: sc[m]['s_te'].astype(float) for m in OS.MAIN5}, best, 1000)
        ms = [m for m in OS.MAIN5 if m != best]
        pb = []
        for m in ms:
            v = bo[m]; pb.append(2 * min((v <= 0).mean(), (v >= 0).mean()))
        ph = holm(pb)
        for i, m in enumerate(ms):
            lo, hi = np.quantile(bo[m], [.025, .975])
            sd = np.std([_auc(yo[OS.load(Tyr, m, s)['idx_te']], OS.load(Tyr, m, s)['s_te']) -
                         _auc(yo[OS.load(Tyr, best, s)['idx_te']], OS.load(Tyr, best, s)['s_te']) for s in (42, 43, 44)], ddof=1)
            orows.append(dict(dataset=f'OOT-T{Tyr}', metric='ROC_AUC', model=m, best=best, mean_delta=bo[m].mean(),
                              ci_lo=lo, ci_hi=hi, p=pb[i], p_holm=ph[i], tost_p=np.nan, tost_p_holm=np.nan,
                              equivalent=bool(lo > -MARGIN and hi < MARGIN), seed_sd_delta=sd,
                              mean_value=aucs[m], best_value=aucs[best]))
    T = pd.concat([T, pd.DataFrame(orows)], ignore_index=True)
    T.round(5).to_csv(RESULTS_DIR / "phase3c_stats.csv", index=False)
    pd.set_option('display.width', 250)
    for ds in ('D3', 'D2', 'D1'):
        a = T[(T.dataset == ds) & (T.metric == 'ROC_AUC')]
        print(f"\n=== {ds}: best by mean AUC = {a.best.iloc[0]} ({a.best_value.iloc[0]:.4f}); N-B corrected, Holm over 9 ===")
        print(a[['model', 'mean_value', 'mean_delta', 'ci_lo', 'ci_hi', 'p_holm', 'tost_p_holm', 'equivalent', 'boot_lo', 'boot_hi']]
              .sort_values('mean_value', ascending=False).round(4).to_string(index=False))
    a = T[(T.dataset == 'D3') & (T.metric == 'ROC_AUC')]
    eq = set(a[a.equivalent == True].model) | ({a.best.iloc[0]} if a.best.iloc[0] in BOOSTING else set())
    nb_eq = [m for m in BOOSTING if m in eq]
    print(f"\nH9: boosting models equivalent to best on D3 (incl. best if boosting): {nb_eq} ({len(nb_eq)}/5) -> "
          f"{'SUPPORTED' if len(nb_eq) >= 3 else 'REJECTED'}")
    print("not equivalent:", sorted(set(MODEL_NAMES) - eq))
    for ds in ('D3', 'D2', 'D1'):
        for metric in ('Brier', 'F1_tuned'):
            b = T[(T.dataset == ds) & (T.metric == metric)]
            print(f"{ds} {metric}: models with Holm p<.05 vs best: {b[b.p_holm < .05].model.tolist()}")
    o = T[T.dataset.str.startswith('OOT')]
    print("\nOOT folds:\n", o[['dataset', 'model', 'best', 'mean_delta', 'ci_lo', 'ci_hi', 'p_holm', 'equivalent', 'seed_sd_delta']].round(4).to_string(index=False))


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


if __name__ == "__main__":
    main()
