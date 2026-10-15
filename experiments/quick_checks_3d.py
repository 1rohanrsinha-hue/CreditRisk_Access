"""Phase 3D / task A: (1) OOT capacity gains with paired-bootstrap CIs (3 seeds)
and (2) a Saerens-EM diagnostic that separates classifier/calibration failure
from drift."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR
from creditrisk.calibration import Isotonic
from creditrisk.capacity import approve_dollar
from creditrisk.expected_profit import ExpectedProfit
from oot_study import fold_indices, usable_columns, FOLDS
import oot_scores as OS
from drift_fixes import prior_shift

KS = (0.3, 0.5, 0.7)
RANKS = ('default_risk', 'expected_profit', 'expected_roi')


def em(p, pi_old, cap=1000, tol=1e-7):
    pi, it = pi_old, 0
    for it in range(1, cap + 1):
        new = prior_shift(p, pi_old, pi).mean()
        if abs(new - pi) < tol:
            pi = new; break
        pi = new
    return pi, it, it >= cap


def main(n_boot=400, chunk=50):
    t = OS.oot_table(); X, y, yr, tm, month = t['X'], t['y'], t['yr'], t['tm'], t['month']
    P, funded = t['P']['P2_3%'], t['funded']
    rows = []
    rng = np.random.default_rng(0)
    for T in FOLDS:
        tr, ca, te = fold_indices(T, yr, tm)
        cols, _ = usable_columns(X, tr); Xc = X[cols]
        for seed in (42, 43, 44):
            ep = ExpectedProfit().fit(Xc.iloc[tr], y[tr], P[tr], seed=seed, n_jobs=4)
            mg, mb = ep.mu(Xc.iloc[te])
            cols_A, meta = [], []
            for model in OS.MAIN5:
                z = OS.load(T, model, seed)
                s_ca, s_te = z['s_ca'].astype(float), z['s_te'].astype(float)
                p_te = Isotonic().fit(s_ca, y[ca]).predict(s_te)
                E = (1 - p_te) * mg + p_te * mb
                keys = {'default_risk': s_te, 'expected_profit': -E, 'expected_roi': -E / funded[te]}
                for rk in RANKS:
                    for k in KS:
                        a = approve_dollar(keys[rk], funded[te], k)
                        cols_A.append(P[te] * (a.astype(float) - k) * 1000)
                        meta.append((T, seed, model, rk, k, float(cols_A[-1].mean())))
            A = np.stack(cols_A, axis=1)                    # n x 45
            n = len(te); means = []
            for _ in range(0, n_boot, chunk):
                W = rng.poisson(1.0, (chunk, n)).astype(np.float32)
                means.append((W @ A.astype(np.float32)) / n)
            M = np.vstack(means)
            lo, hi = np.quantile(M, .025, axis=0), np.quantile(M, .975, axis=0)
            for (Tm, sd, model, rk, k, g), l, h in zip(meta, lo, hi):
                rows.append(dict(T=Tm, seed=sd, model=model, ranking=rk, k=k, gain_per_1000=g, ci_lo=l, ci_hi=h,
                                 excludes_zero=bool(l > 0 or h < 0)))
            print(f"[quick] capacity T={T} seed={seed}", flush=True)
    C = pd.DataFrame(rows); C.round(1).to_csv(RESULTS_DIR / "phase3d_quick_capacity.csv", index=False)
    pd.set_option('display.width', 250)
    g = C.groupby(['T', 'ranking', 'k']).agg(gain=('gain_per_1000', 'mean'), ci_lo=('ci_lo', 'mean'), ci_hi=('ci_hi', 'mean'),
                                              n_pos=('ci_lo', lambda v: int((v > 0).sum())), n_neg=('ci_hi', lambda v: int((v < 0).sum())),
                                              n=('ci_lo', 'size')).round(0)
    print("\n=== OOT capacity gain over random, P2-3%, dollar capacity (mean over 5 models × 3 seeds; CI bounds averaged) ===")
    print(g.to_string())
    # ------------------------------------------------------------ EM diagnostic
    er = []
    for T in FOLDS:
        tr, ca, te = fold_indices(T, yr, tm)
        for model in OS.MAIN5:
            for seed in (42, 43, 44):
                z = OS.load(T, model, seed)
                s_ca, s_te = z['s_ca'].astype(float), z['s_te'].astype(float); y_ca, y_te = y[ca], y[te]
                r = np.random.RandomState(seed)
                perm = r.permutation(len(ca)); A_, B_ = perm[:len(ca) // 2], perm[len(ca) // 2:]
                perm2 = r.permutation(len(te)); C_, D_ = perm2[:len(te) // 2], perm2[len(te) // 2:]
                cases = {
                    '(a) T-1 random halves: calibrate on A, EM on B': (s_ca[A_], y_ca[A_], s_ca[B_], y_ca[B_]),
                    '(b) year T random halves: calibrate on half, EM on other': (s_te[C_], y_te[C_], s_te[D_], y_te[D_]),
                    '(c) calibrate on T-1, EM on year T': (s_ca, y_ca, s_te, y_te)}
                for name, (sc, yc, st, yt) in cases.items():
                    iso = Isotonic().fit(sc, yc); p = iso.predict(st)
                    pi_old, pi_true = yc.mean(), yt.mean()
                    pi_hat, it, capped = em(p, pi_old)
                    er.append(dict(case=name, T=T, model=model, seed=seed, pi_old=pi_old, pi_true=pi_true,
                                   mean_p=p.mean(), pi_hat=pi_hat, abs_err=abs(pi_hat - pi_true), iters=it, hit_cap=capped,
                                   mean_p_minus_pi_true=p.mean() - pi_true))
    E = pd.DataFrame(er); E.round(5).to_csv(RESULTS_DIR / "phase3d_quick_em.csv", index=False)
    s = E.groupby(['case', 'T']).agg(pi_old=('pi_old', 'mean'), pi_true=('pi_true', 'mean'), mean_p=('mean_p', 'mean'),
                                     pi_hat=('pi_hat', 'mean'), abs_err=('abs_err', 'mean'), iters=('iters', 'mean'),
                                     capped=('hit_cap', 'mean')).round(4)
    print("\n=== EM diagnostic (mean over 5 models × 3 seeds) ===\n", s.to_string())
    print("\nper-model |pi_hat - pi_true| by case:\n", E.groupby(['case', 'model']).abs_err.mean().unstack(0).round(4).to_string())


if __name__ == "__main__":
    main()
