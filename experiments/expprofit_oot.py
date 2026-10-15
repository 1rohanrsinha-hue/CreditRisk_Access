"""Phase 3B / task C (H6): expected-profit rule evaluated properly out of time.
Per fold T: classifier + two LightGBM profit regressors trained on <= T-2,
isotonic calibrator and shrinkage margin m chosen on T-1, test on T.
Approve iff E[profit | x] > m."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.expected_profit import ExpectedProfit
from creditrisk.models import fit_model
from creditrisk.stats import paired_mean_diff_ci
from creditrisk import thresholds as T
from oot_study import make_data, fold_indices, usable_columns, FOLDS

GRID = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
SEEDS = (42, 43, 44)


def main(n_jobs=4, n_boot=1000):
    X, y, P, yr, tm = make_data()
    rows = []
    for Tyr in FOLDS:
        tr, ca, te = fold_indices(Tyr, yr, tm)
        cols, dropped = usable_columns(X, tr); Xc = X[cols]
        for seed in SEEDS:
            t0 = time.time()
            clf = fit_model('LightGBM', Xc.iloc[tr], pd.Series(y[tr]), 'none', seed, n_jobs)
            s_ca, s_te = clf.predict_proba(Xc.iloc[ca])[:, 1], clf.predict_proba(Xc.iloc[te])[:, 1]
            cal = make_calibrators(s_ca, y[ca])['isotonic']
            p_ca, p_te = apply_cal(cal, s_ca), apply_cal(cal, s_te)
            for sc in ('P1', 'P2_3%'):
                ep = ExpectedProfit().fit(Xc.iloc[tr], y[tr], P[sc][tr], seed=seed, n_jobs=n_jobs)
                E_ca, E_te = ep.expected(Xc.iloc[ca], p_ca), ep.expected(Xc.iloc[te], p_te)
                prof_ca = np.array([P[sc][ca][E_ca > m].sum() for m in GRID])
                m_star = GRID[int(np.argmax(prof_ca))]
                c_fn, c_fp = -P[sc][ca][y[ca] == 1].mean(), P[sc][ca][y[ca] == 0].mean()
                appr = {'approve_all': np.ones(len(te), bool),
                        'expected_profit_m0': E_te > 0,
                        'expected_profit_mstar': E_te > m_star,
                        'youden_T-1': p_te < T.youden_threshold(y[ca], p_ca),
                        'elkan_T-1': p_te < T.elkan_threshold(c_fp, c_fn) if c_fp > 0 and c_fn > 0 else np.zeros(len(te), bool),
                        'profit_max_T-1': p_te < T.profit_max_threshold(p_ca, P[sc][ca])}
                for r, a in appr.items():
                    d = P[sc][te] * (a.astype(float) - 1.0) * 1000      # vs approve-all
                    m, lo, hi, p = paired_mean_diff_ci(d, n_boot, seed=13) if r != 'approve_all' else (0, 0, 0, 1)
                    rows.append(dict(T=Tyr, seed=seed, scenario=sc, rule=r, m_star=m_star,
                                     profit_per_1000=P[sc][te][a].sum() / len(te) * 1000,
                                     accept_rate=a.mean(),
                                     bad_rate_approved=y[te][a].mean() if a.any() else np.nan,
                                     d_vs_approve_all=m, ci_lo=lo, ci_hi=hi, dropped='|'.join(dropped)))
            print(f"[expprofit-oot] T={Tyr} seed={seed} {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); R.round(2).to_csv(RESULTS_DIR / "phase3b_expprofit_oot.csv", index=False)
    pd.set_option('display.width', 250)
    s42 = R[(R.seed == 42) & (R.rule.isin(['approve_all', 'expected_profit_m0', 'expected_profit_mstar',
                                           'youden_T-1', 'elkan_T-1', 'profit_max_T-1']))]
    print(s42[['T', 'scenario', 'rule', 'm_star', 'profit_per_1000', 'accept_rate', 'bad_rate_approved',
               'd_vs_approve_all', 'ci_lo', 'ci_hi']].round(3).to_string(index=False))
    # H6 verdict (seed 42 CI primary; all seeds shown)
    ep = R[R.rule == 'expected_profit_mstar']
    print("\nH6: expected_profit(m*) - approve_all, paired 95% CI lower bound by fold/scenario/seed:")
    print(ep.pivot_table(index=['scenario', 'T'], columns='seed', values='ci_lo').round(0).to_string())
    print(ep.pivot_table(index=['scenario', 'T'], columns='seed', values='d_vs_approve_all').round(0).to_string())
    pos = ep[(ep.seed == 42) & (ep.ci_lo > 0)].groupby('scenario').T.nunique()
    print("folds with CI lower bound > 0 (seed 42):", pos.to_dict())


if __name__ == "__main__":
    main()
