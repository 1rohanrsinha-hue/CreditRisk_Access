"""Phase 5 / task A: H23 (frozen at f60b319). Refit the OOT folds with training, calibration and
test sets all restricted to 36-month loans, for T = 2013, 2014, 2015. Gains over random, P2-3%,
dollar capacity, k in {30,50,70}%, mean over 5 models x 3 seeds, Poisson bootstrap."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR
from creditrisk.calibration import Isotonic
from creditrisk.capacity import approve_dollar
from creditrisk.expected_profit import ExpectedProfit
from creditrisk.models import fit_model
from oot_study import usable_columns, FOLDS, SEED
import oot_scores as OS

KS = (0.3, 0.5, 0.7); RANKS = ('default_risk', 'expected_profit', 'rate_only'); NB = 1000
SEEDS = (42, 43, 44)


def fold36(T, yr, tm):
    keep = tm == 36
    tr = np.where(keep & (yr <= T - 2))[0]; ca = np.where(keep & (yr == T - 1))[0]; te = np.where(keep & (yr == T))[0]
    if len(tr) > 400_000:
        tr = np.sort(np.random.RandomState(SEED).choice(tr, 400_000, replace=False))
    return tr, ca, te


def main():
    t = OS.oot_table(); X, y, yr, tm, d = t['X'], t['y'], t['yr'], t['tm'], t['d']
    P, funded = t['P']['P2_3%'], t['funded']; rate = d.int_rate.values.astype(float)
    rng = np.random.default_rng(0); rows, draws = [], {}
    for T in FOLDS:
        tr, ca, te = fold36(T, yr, tm)
        cols, _ = usable_columns(X, tr); Xc = X[cols]
        print(f"[36m] T={T} train {len(tr)} cal {len(ca)} test {len(te)} dropped {sorted(set(X.columns)-set(cols))}", flush=True)
        acc = {}; cors = []
        for seed in SEEDS:
            ep = ExpectedProfit().fit(Xc.iloc[tr], y[tr], P[tr], seed=seed, n_jobs=4)
            mg, mb = ep.mu(Xc.iloc[te])
            for model in OS.MAIN5:
                t0 = time.time()
                m = fit_model(model, Xc.iloc[tr], pd.Series(y[tr]), 'none', seed, 4)
                s_ca = m.predict_proba(Xc.iloc[ca])[:, 1]; s_te = m.predict_proba(Xc.iloc[te])[:, 1]
                p = Isotonic().fit(s_ca, y[ca]).predict(s_te)
                E = (1 - p) * mg + p * mb
                keys = {'default_risk': s_te, 'expected_profit': -E, 'rate_only': rate[te]}
                cors.append(np.corrcoef(p, rate[te])[0, 1])
                for rk in RANKS:
                    for k in KS:
                        a = approve_dollar(keys[rk], funded[te], k)
                        acc.setdefault((rk, k), []).append(P[te] * (a.astype(float) - k) * 1000)
                print(f"[36m] T={T} s{seed} {model} {time.time()-t0:.0f}s", flush=True)
        n = len(te)
        for (rk, k), cl in acc.items():
            avg = np.mean(cl, axis=0)
            bs = np.concatenate([(w := rng.poisson(1.0, (100, n)).astype(np.float32)) @ avg.astype(np.float32) / w.sum(1) for _ in range(NB // 100)])
            draws[(T, rk, k)] = bs; lo, hi = np.quantile(bs, [.025, .975])
            rows.append(dict(T=T, term='36', ranking=rk, k=k, n_train=len(tr), n_test=n, gain_per_1000=float(avg.mean()), ci_lo=lo, ci_hi=hi,
                             excludes_zero=bool(lo > 0 or hi < 0), mean_rate=float(rate[te].mean()), default_rate=float(y[te].mean()),
                             corr_p_rate=float(np.mean(cors))))
    C = pd.DataFrame(rows); C.round(4).to_csv(RESULTS_DIR / "phase5_term36.csv", index=False)
    pd.set_option('display.width', 250); print(C.round(3).to_string(index=False))
    g = {T: draws[(T, 'default_risk', 0.5)] for T in FOLDS}
    pt = {T: C[(C['T'] == T) & (C.ranking == 'default_risk') & (C.k == .5)].gain_per_1000.iloc[0] for T in FOLDS}
    h = []
    for T in (2014, 2015):
        lo, hi = np.quantile(g[T] - g[2013], [.025, .975]); h.append(dict(contrast=f"gain({T}) - gain(2013)", point=pt[T] - pt[2013], ci_lo=lo, ci_hi=hi))
    H = pd.DataFrame(h); H.round(1).to_csv(RESULTS_DIR / "phase5_h23.csv", index=False)
    print("\nH23 point gains:", {k: round(v, 1) for k, v in pt.items()}); print(H.round(1).to_string(index=False))
    v = ("SUPPORTED" if all(H.ci_lo > 0) else "INCONCLUSIVE") if all(H.point > 0) else "REJECTED"
    print("H23 verdict:", v)


if __name__ == "__main__":
    main()
