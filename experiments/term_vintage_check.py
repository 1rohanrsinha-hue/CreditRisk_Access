"""Phase 4 / task A: does term mix explain the vintage sign change in the
default-risk-ranking dollar-capacity gain? Cells = test vintage x term.
Rankings: default_risk (calibrated isotonic score), expected_profit, rate_only
(interest rate). P2-3%, dollar capacity, k in {30,50,70}%.  Gain = profit per
$1000-of-loans-scored above k x approve-all, averaged over 5 models x 3 seeds
(one averaged per-loan column, Poisson bootstrap).  H20 (frozen at b3240f4)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import Isotonic
from creditrisk.capacity import approve_dollar
from creditrisk.expected_profit import ExpectedProfit
from oot_study import fold_indices, usable_columns, FOLDS
import oot_scores as OS

KS = (0.3, 0.5, 0.7)
RANKS = ('default_risk', 'expected_profit', 'rate_only')
NB = 1000


def main():
    t = OS.oot_table(); X, y, yr, tm, d = t['X'], t['y'], t['yr'], t['tm'], t['d']
    P, funded = t['P']['P2_3%'], t['funded']
    rate = d.int_rate.values.astype(float)
    rng = np.random.default_rng(0)
    rows, draws = [], {}
    for T in FOLDS:
        tr, ca, te = fold_indices(T, yr, tm)
        cols, _ = usable_columns(X, tr); Xc = X[cols]
        tte = tm[te]
        acc = {}      # (rank,k,term) -> list of per-loan columns (full-te length) ; cor lists
        cors = {}
        for seed in (42, 43, 44):
            ep = ExpectedProfit().fit(Xc.iloc[tr], y[tr], P[tr], seed=seed, n_jobs=4)
            mg, mb = ep.mu(Xc.iloc[te])
            for model in OS.MAIN5:
                z = OS.load(T, model, seed)
                s_ca, s_te = z['s_ca'].astype(float), z['s_te'].astype(float)
                p = Isotonic().fit(s_ca, y[ca]).predict(s_te)
                E = (1 - p) * mg + p * mb
                keys = {'default_risk': s_te, 'expected_profit': -E, 'rate_only': rate[te]}
                for term in ('all', 36, 60):
                    m = np.ones(len(te), bool) if term == 'all' else (tte == term)
                    if m.sum() == 0:
                        continue
                    idx = np.where(m)[0]
                    cors.setdefault(term, []).append(np.corrcoef(p[idx], rate[te][idx])[0, 1])
                    for rk in RANKS:
                        for k in KS:
                            a = approve_dollar(keys[rk][idx], funded[te][idx], k)
                            col = np.zeros(len(te)); col[idx] = P[te][idx] * (a.astype(float) - k) * 1000
                            acc.setdefault((rk, k, term), []).append(col[idx])
        for (rk, k, term), cl in acc.items():
            idx = np.where(np.ones(len(te), bool) if term == 'all' else (tte == term))[0]
            avg = np.mean(cl, axis=0); n = len(idx)
            bs = np.concatenate([(w := rng.poisson(1.0, (100, n)).astype(np.float32)) @ avg.astype(np.float32) / w.sum(1) for _ in range(NB // 100)])
            draws[(T, term, rk, k)] = bs
            lo, hi = np.quantile(bs, [.025, .975])
            rows.append(dict(T=T, term=str(term), ranking=rk, k=k, n_test=n, gain_per_1000=float(avg.mean()),
                             ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0 or hi < 0),
                             mean_rate=float(rate[te][idx].mean()), default_rate=float(y[te][idx].mean()),
                             corr_p_rate=float(np.mean(cors[term])), mean_funded=float(funded[te][idx].mean())))
        print(f"[term-vintage] T={T} done", flush=True)
    C = pd.DataFrame(rows); C.round(4).to_csv(RESULTS_DIR / "phase4_term_vintage.csv", index=False)
    pd.set_option('display.width', 250)
    print(C.round(3).to_string(index=False))
    # ---- H20 (36m only, default_risk, k=.5)
    g = {T: draws[(T, 36, 'default_risk', 0.5)] for T in FOLDS}
    pt = {T: C[(C['T'] == T) & (C.term == '36') & (C.ranking == 'default_risk') & (C.k == .5)].gain_per_1000.iloc[0] for T in FOLDS}
    h = []
    for T in (2014, 2015):
        diff = g[T] - g[2013]; lo, hi = np.quantile(diff, [.025, .975])
        h.append(dict(contrast=f"gain({T}) - gain(2013), 36m, default-risk, k=.5", point=pt[T] - pt[2013], ci_lo=lo, ci_hi=hi))
    H = pd.DataFrame(h); H.round(2).to_csv(RESULTS_DIR / "phase4_h20.csv", index=False)
    print("\nH20 point gains (36m, k=.5):", {k: round(v, 1) for k, v in pt.items()}); print(H.round(2).to_string(index=False))
    if all(H.point > 0):
        v = "SUPPORTED" if all(H.ci_lo > 0) else "INCONCLUSIVE"
    else:
        v = "REJECTED"
    print("H20 verdict:", v)
    # ---- figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    cells = [(2013, 'all'), (2013, '36'), (2013, '60'), (2014, 'all'), (2014, '36'), (2014, '60'), (2015, '36')]
    for ax, rk in zip(axes, RANKS):
        S = C[(C.ranking == rk) & (C.k == .5)].set_index(['T', 'term'])
        xs = np.arange(len(cells))
        v_ = [S.loc[c, 'gain_per_1000'] for c in [(a, str(b)) for a, b in cells]]
        lo_ = [S.loc[c, 'ci_lo'] for c in [(a, str(b)) for a, b in cells]]
        hi_ = [S.loc[c, 'ci_hi'] for c in [(a, str(b)) for a, b in cells]]
        ax.bar(xs, v_, color='#4c78a8'); ax.errorbar(xs, v_, yerr=[np.array(v_) - lo_, np.array(hi_) - np.array(v_)], fmt='none', ecolor='k', lw=1)
        ax.axhline(0, color='k', lw=.6); ax.set_xticks(xs); ax.set_xticklabels([f"{a}\n{b}" for a, b in cells], fontsize=7)
        ax.set_title(rk.replace('_', ' '), fontsize=9); ax.set_ylabel("gain over random, USD per 1000 loans", fontsize=8)
    fig.suptitle("OOT dollar-capacity gain over random, P2-3%, k=50% (5 models × 3 seeds), by test vintage and term", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "phase4_term_vintage.png", dpi=300)


if __name__ == "__main__":
    main()
