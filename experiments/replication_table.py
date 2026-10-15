"""Phase 3B / task C: claim x dataset replication table and the pre-registered H8 verdict."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR as R

rows = []
for ds in ('D1', 'D2', 'D3'):
    sp = pd.read_csv(R / f"phase2_weighting_spread_{ds}.csv").set_index(['weighting', 'thr_rule'])
    cal = pd.read_csv(R / f"phase2_calibration_{ds}.csv")
    d = dict(dataset=ds)
    d['recall@0.5 none'] = sp.loc[('none', '0.5'), 'recall_mean_mean']
    d['recall@0.5 balanced'] = sp.loc[('balanced', '0.5'), 'recall_mean_mean']
    d['recall spread@0.5 balanced'] = sp.loc[('balanced', '0.5'), 'recall_spread_mean']
    d['tuned recall none'] = sp.loc[('none', 'tuned'), 'recall_mean_mean']
    d['tuned recall balanced'] = sp.loc[('balanced', 'tuned'), 'recall_mean_mean']
    d['|tuned recall none-bal|'] = abs(d['tuned recall none'] - d['tuned recall balanced'])
    d['tuned recall spread none'] = sp.loc[('none', 'tuned'), 'recall_spread_mean']
    d['tuned recall spread balanced'] = sp.loc[('balanced', 'tuned'), 'recall_spread_mean']
    d['AUC spread (balanced)'] = sp.loc[('balanced', 'tuned'), 'auc_spread_mean']
    c = cal[cal.calib != 'raw']
    d['H1 share |t*-pi|<=.05'] = (c.dev_pi_mean.abs() <= .05).mean()
    d['H1 mean |t*-pi|'] = c.dev_pi_mean.abs().mean()
    rb = cal[(cal.calib == 'raw') & (cal.weighting == 'balanced')]
    d['H2 share |t*-.5|<=.05'] = (rb.dev_half_mean.abs() <= .05).mean()
    g = c.pivot_table(index=['model', 'calib'], columns='weighting', values='t_star_mean')
    d['H3 mean |t*none-t*bal| (calibrated)'] = (g['none'] - g['balanced']).abs().mean()
    gr = cal[cal.calib == 'raw'].pivot_table(index='model', columns='weighting', values='t_star_mean')
    d['raw mean |t*none-t*bal|'] = (gr['none'] - gr['balanced']).abs().mean()
    rbm = cal[(cal.calib == 'raw') & (cal.weighting == 'balanced')]
    d['raw balanced ECE'] = rbm.ECE_mean.mean(); d['calibrated ECE'] = c.ECE_mean.mean()
    lf = R / ("phase2_leaf_sweep.csv" if ds == 'D2' else f"phase2_leaf_sweep_{ds}.csv")
    if lf.exists():
        L = pd.read_csv(lf)
        for m in ('Extra Trees', 'Random Forest'):
            a = L[(L.model == m) & (L.weighting == 'balanced')].set_index('leaf')
            d[f'{m} bal recall@.5 leaf1->20'] = f"{a.loc[1, 'Recall_fixed_0.5_mean']:.3f}->{a.loc[20, 'Recall_fixed_0.5_mean']:.3f}"
            d[f'{m} bal t* leaf1->20'] = f"{a.loc[1, 'threshold_youden_mean']:.3f}->{a.loc[20, 'threshold_youden_mean']:.3f}"
            d[f'{m} d_recall'] = a.loc[20, 'Recall_fixed_0.5_mean'] - a.loc[1, 'Recall_fixed_0.5_mean']
            d[f'{m} d_tstar'] = a.loc[20, 'threshold_youden_mean'] - a.loc[1, 'threshold_youden_mean']
    rows.append(d)
T = pd.DataFrame(rows).set_index('dataset').T
T.to_csv(R / "phase3b_replication.csv")
pd.set_option('display.width', 250)
print(T.to_string())
d3 = T['D3']
chk = {
    'H1 (>=80% of calibrated variants within .05)': float(d3['H1 share |t*-pi|<=.05']) >= .8,
    'H2 (>=70% balanced-raw models within .05 of .5)': float(d3['H2 share |t*-.5|<=.05']) >= .7,
    'H3 (mean calibrated gap <= .01)': float(d3['H3 mean |t*none-t*bal| (calibrated)']) <= .01,
    'weighting (|tuned recall none-bal| <= .02)': float(d3['|tuned recall none-bal|']) <= .02,
    'leaf purity ET (d_recall>=.30, d_t*>=.15)': float(d3['Extra Trees d_recall']) >= .30 and float(d3['Extra Trees d_tstar']) >= .15,
    'leaf purity RF (d_recall>=.30, d_t*>=.15)': float(d3['Random Forest d_recall']) >= .30 and float(d3['Random Forest d_tstar']) >= .15,
}
print("\nH8 criteria on D3:"); [print(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in chk.items()]
print("H8 verdict:", "REPLICATES" if all(chk.values()) else "DOES NOT FULLY REPLICATE")
