"""Read-only: D3, LightGBM (weighting none, isotonic), P2 profit over discount grid 3%..8% step 0.5.
Mirrors experiments/profit_d3.py: point estimates averaged over the 10 repeats; CI = paired bootstrap
(1000 resamples of test loans, seed 11) on repeat 0, as in the published profit tables."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.data import full_frame, profit_p2
from creditrisk.expected_profit import ExpectedProfit
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import paired_mean_diff_ci
from creditrisk import thresholds as T

RATES = [round(0.03 + 0.005 * i, 3) for i in range(11)]
X, y = get_dataset('D3')[:2]; yv = y.values; F = full_frame().loc[X.index]
rows = []
for r in RATES:
    pr = profit_p2(F, r).values; acc = {k: [] for k in ('all', 'ep', 'youden', 'f1')}; ci = {}
    for rep in range(10):
        z = load_score('D3', rep, 'LightGBM', 'none'); tv, te = z['idx_tv'], z['idx_te']
        ep = ExpectedProfit().fit(X.iloc[tv], yv[tv], pr[tv], seed=42 + rep, n_jobs=4); mg, mb = ep.mu(X.iloc[te])
        cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
        so, st = apply_cal(cal, z['oof'].astype(float)), apply_cal(cal, z['te'].astype(float))
        a = {'all': np.ones(len(te), bool), 'ep': ((1 - st) * mg + st * mb) > 0,
             'youden': st < T.youden_threshold(yv[tv], so), 'f1': st < T.f1_max_threshold(yv[tv], so)}
        p = pr[te]
        for k in acc: acc[k].append(p[a[k]].sum() / len(p) * 1000)
        if rep == 0:
            for k in ('ep', 'youden', 'f1'):
                ci[k] = paired_mean_diff_ci(p * (a[k].astype(float) - 1.0) * 1000, 1000, seed=11)
    row = dict(rate=r, approve_all=np.mean(acc['all']))
    for k in ('ep', 'youden', 'f1'):
        m, lo, hi, _ = ci[k]
        row.update({f'{k}_profit': np.mean(acc[k]), f'{k}_diff_rep0': m, f'{k}_lo': lo, f'{k}_hi': hi, f'{k}_excl0': bool(lo > 0 or hi < 0)})
    rows.append(row); print(row, flush=True)
R = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(R.round(1).to_string(index=False))
for k in ('ep', 'youden', 'f1'):
    s = R[R[f'{k}_excl0']]
    print(k, 'smallest rate with CI excluding 0:', s.rate.iloc[0] if len(s) else None)
