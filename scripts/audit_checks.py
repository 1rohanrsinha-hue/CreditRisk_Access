"""Read-only audit prints for the manuscript tables (no results are modified).
Usage: venv/bin/python scripts/audit_checks.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
R = ROOT / 'experiments' / 'results'
pd.set_option('display.width', 220)


def item1():
    print("== 1. Vintage table, 2015 default-risk k=.5 ==")
    a = pd.read_csv(R / 'phase4_term_vintage.csv'); b = pd.read_csv(R / 'phase5_term36.csv')
    q = lambda d: d[(d['T'] == 2015) & (d.term.astype(str) == '36') & (d.ranking == 'default_risk') & (d.k == .5)].iloc[0]
    qa, qb = q(a), q(b)
    print(f"mixed fit   (phase4): gain {qa.gain_per_1000:.6g}  CI [{qa.ci_lo:.6g}, {qa.ci_hi:.6g}]")
    print(f"36-month fit(phase5): gain {qb.gain_per_1000:.6g}  CI [{qb.ci_lo:.6g}, {qb.ci_hi:.6g}]")
    import oot_scores as OS
    from oot_study import fold_indices, FOLDS
    from term36_refit import fold36
    t = OS.oot_table(); yr, tm = t['yr'], t['tm']
    for T in FOLDS:
        m, s = fold_indices(T, yr, tm), fold36(T, yr, tm)
        same = all(np.array_equal(x, y) for x, y in zip(m, s))
        print(f"T={T}: mixed train/cal/test {[len(x) for x in m]}  36m train/cal/test {[len(x) for x in s]}  identical index sets: {same}")


def item3():
    print("== 3. Calibration table (Set x Weights x calibrator) ==")
    import importlib
    S = pd.concat([pd.read_csv(R / f'phase2_calibration_summary_D{i}.csv') for i in (1, 2, 3)])
    S = S[S.calib.isin(['platt', 'isotonic'])]
    print(S[['dataset', 'weighting', 'calib', 't_star', 'pi', 'abs_t_minus_pi']].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"max |t*-pi| over these 12 cells: {S.abs_t_minus_pi.max():.6f}")
    for w in ('balanced', 'none'):
        print(f"  max for isotonic+{w}: {S[(S.weighting==w)&(S.calib=='isotonic')].abs_t_minus_pi.max():.6f}")


def item4():
    print("== 4. Taiwan |t*-pi| aggregations ==")
    C = pd.read_csv(R / 'phase3e_external_TAIWAN.csv'); cell = {}
    for w in ('none', 'balanced'):
        for c in ('isotonic', 'platt'):
            g = C[(C.weighting == w) & (C.calib == c)]
            cell[(w, c)] = g.groupby('model').abs_dev_pi.mean().mean()
            print(f"{w:9s}{c:9s} mean over models of per-model repeat-mean |t*-pi| = {cell[(w, c)]:.6f}")
    print(f"H26 = (none, isotonic) = {cell[('none','isotonic')]:.6f}")
    print(f"C3  = mean over the 4 cells (2 weightings x 2 calibrators) = {np.mean(list(cell.values())):.6f}")


def item5():
    print("== 5. Nested-CV random forest AUC ==")
    d = pd.read_csv(R / 'phase3d_optuna.csv'); r = d[d.model == 'Random Forest']
    print(r[['fold', 'AUC_default', 'AUC_tuned']].to_string(index=False))
    print(f"mean default {r.AUC_default.mean():.6f}  mean tuned {r.AUC_tuned.mean():.6f}  mean diff {(r.AUC_tuned-r.AUC_default).mean():.6f}")


def item6():
    print("== 6. D3: 2010-issued loans ==")
    from creditrisk.data import full_frame, label_strict, issue_year
    f = full_frame(); y = issue_year(f)
    f10 = f[y == 2010]; print(f"2010-issued loans in file: {len(f10):,}")
    print(f10.loan_status.value_counts(dropna=False).to_string())
    s = label_strict(f10); print(f"strict-labelled (kept in D3): {len(s):,}; dropped: {len(f10)-len(s):,}")


if __name__ == '__main__':
    for fn in (item1, item3, item4, item5, item6):
        fn(); print()
