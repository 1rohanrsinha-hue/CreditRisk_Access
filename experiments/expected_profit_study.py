"""Phase 3A / task C: score-dependent cost rule vs Elkan vs profit-max (D2)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.data import full_frame, profit_p1, profit_p2
from creditrisk.expected_profit import ExpectedProfit
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import summarize, paired_mean_diff_ci
from creditrisk import thresholds as T

SCEN = {'P1': lambda F: profit_p1(F), 'P2_0%': lambda F: profit_p2(F, 0.0),
        'P2_5%': lambda F: profit_p2(F, 0.05)}
CLF = 'LightGBM'


def main(n_rep=10, n_boot=1000):
    X, y, _ = get_dataset('D2'); yv = y.values
    F = full_frame().loc[X.index]
    PR = {k: f(F).values for k, f in SCEN.items()}
    rows, boots, calib_rows = [], [], []
    for rep in range(n_rep):
        z = load_score('D2', rep, CLF, 'none')
        tv, te = z['idx_tv'], z['idx_te']
        cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
        so, st = apply_cal(cal, z['oof'].astype(float)), apply_cal(cal, z['te'].astype(float))
        y_tv, y_te = yv[tv], yv[te]
        for sc in SCEN:
            t0 = time.time()
            p_tv, p_te = PR[sc][tv], PR[sc][te]
            ep = ExpectedProfit().fit(X.iloc[tv], y_tv, p_tv, seed=42 + rep, n_jobs=4)
            mg, mb = ep.mu(X.iloc[te])
            E = (1 - st) * mg + st * mb
            c_fn, c_fp = -p_tv[y_tv == 1].mean(), p_tv[y_tv == 0].mean()
            appr = {'expected_profit': E > 0,
                    'elkan': st < T.elkan_threshold(c_fp, c_fn),
                    'profit_max': st < T.profit_max_threshold(so, p_tv),
                    'youden': st < T.youden_threshold(y_tv, so),
                    'approve_all': np.ones(len(st), bool)}
            for r, a in appr.items():
                rows.append(dict(scenario=sc, rep=rep, rule=r,
                                 profit_per_1000=p_te[a].sum() / len(a) * 1000,
                                 accept_rate=a.mean(),
                                 bad_rate_approved=y_te[a].mean() if a.any() else np.nan))
            if rep == 0:
                for ref in ('elkan', 'profit_max'):
                    d = p_te * (appr['expected_profit'].astype(float) - appr[ref].astype(float)) * 1000
                    m, lo, hi, p = paired_mean_diff_ci(d, n_boot, seed=7)
                    boots.append(dict(scenario=sc, rule='expected_profit', vs=ref,
                                      d_profit_per_1000=m, ci_lo=lo, ci_hi=hi, p=p))
                # regressor calibration by decile of prediction (test)
                for name, mu, mask in (('mu_good|repaid', mg, y_te == 0), ('mu_bad|default', mb, y_te == 1),
                                       ('E[profit] all', E, np.ones(len(E), bool))):
                    pred, real = mu[mask], p_te[mask]
                    dec = pd.qcut(pd.Series(pred).rank(method='first'), 10, labels=False)
                    g = pd.DataFrame({'pred': pred, 'real': real, 'dec': dec}).groupby('dec').mean()
                    for d_, r_ in g.iterrows():
                        calib_rows.append(dict(scenario=sc, target=name, decile=int(d_) + 1,
                                               mean_predicted=r_.pred, mean_realized=r_.real))
                    from sklearn.metrics import r2_score
                    calib_rows.append(dict(scenario=sc, target=name, decile=0,
                                           mean_predicted=float(pred.mean()), mean_realized=float(real.mean()),
                                           r2=float(r2_score(real, pred))))
            print(f"[ep] rep={rep} {sc} {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    S = summarize(R, ['scenario', 'rule'], ['profit_per_1000', 'accept_rate', 'bad_rate_approved'])
    S.round(4).to_csv(RESULTS_DIR / "phase3a_expprofit.csv", index=False)
    pd.DataFrame(boots).round(1).to_csv(RESULTS_DIR / "phase3a_expprofit_bootstrap.csv", index=False)
    C = pd.DataFrame(calib_rows); C.round(2).to_csv(RESULTS_DIR / "phase3a_expprofit_calibration.csv", index=False)
    pd.set_option('display.width', 250)
    print(S.round(3).to_string(index=False)); print(pd.DataFrame(boots).round(0).to_string(index=False))
    print(C[C.decile == 0].round(2).to_string(index=False))
    print(C[(C.scenario == 'P1') & (C.target == 'E[profit] all')].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
