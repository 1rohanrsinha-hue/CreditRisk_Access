"""Phase 3D / task D: resampling vs weighting vs threshold moving (H15).
Stages: score (resumable, parallel by --worker) | analyze."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR
from creditrisk.repeats import run_scoring
from creditrisk.stats import summarize
from creditrisk.study import analyse
from stats_study import nb, holm

MODELS = ['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost']
METHODS = ['none', 'balanced', 'undersample', 'smote']
REPS = {'D3': 5, 'D1': 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "score":
        t0 = time.time()
        for ds, n in REPS.items():
            run_scoring(ds, list(range(n)), models=MODELS, weightings=('undersample', 'smote'),
                        n_jobs=a.n_jobs, worker=(a.worker, a.n_workers))
        print(f"[done] resampling scoring worker {a.worker} {time.time()-t0:.0f}s"); return
    out, pr = [], {}
    for ds, n in REPS.items():
        R, C = analyse(ds, list(range(n)), tag='_resamp', models=MODELS, weightings=tuple(METHODS[:2]) + ('undersample', 'smote'),
                       with_profit=False)
        R = R[R.model.isin(MODELS) & R.weighting.isin(METHODS)]; C = C[C.model.isin(MODELS) & C.weighting.isin(METHODS)]
        for calib, rule, lab in (('raw', 'fixed_0.5', 'thr0.5 (raw)'), ('raw', 'youden', 'Youden (raw)'),
                                 ('isotonic', 'youden', 'Youden (isotonic)')):
            x = R[(R.calib == calib) & (R.rule == rule)].copy(); x['setting'] = lab; x['dataset'] = ds
            out.append(x)
        Cx = C.copy(); Cx['dataset'] = ds
        pr[ds] = (R, Cx)
    allR = pd.concat(out, ignore_index=True)
    S = summarize(allR, ['dataset', 'model', 'weighting', 'setting'],
                  ['Recall', 'Precision', 'F1', 'MCC', 'ROC_AUC', 'Brier', 'threshold'])
    S.round(4).to_csv(RESULTS_DIR / "phase3d_resampling.csv", index=False)
    pd.set_option('display.width', 250)
    C3 = pr['D3'][1]
    # ----------------------------------------------------- H15
    t = (C3[C3.calib == 'raw'].groupby(['model', 'weighting']).agg(t_raw=('t_star', 'mean'), ECE=('ECE', 'mean'),
                                                                  Brier=('Brier', 'mean')).reset_index())
    t['abs_t_half'] = (t.t_raw - .5).abs()
    print("raw Youden t* and ECE by method:\n", t.pivot(index='model', columns='weighting', values='t_raw').round(3).to_string())
    i_ok = {}
    for meth in ('undersample', 'smote'):
        c = 0
        for m in MODELS:
            c += int(t[(t.model == m) & (t.weighting == meth)].abs_t_half.iloc[0] < t[(t.model == m) & (t.weighting == 'none')].abs_t_half.iloc[0])
        i_ok[meth] = c
    print("H15(i) models moving toward .5 (of 4):", i_ok)
    cal = C3[C3.calib != 'raw'].assign(d=lambda d: (d.t_star - d.pi).abs()).groupby(['model', 'weighting', 'calib']).d.mean()
    print("H15(ii) max mean |t*-pi| calibrated, methods undersample/smote:",
          round(cal.xs('undersample', level='weighting').max(), 4), round(cal.xs('smote', level='weighting').max(), 4))
    print(cal.unstack('calib').round(4).to_string())
    rows = []
    for m in MODELS:
        base = C3[(C3.model == m) & (C3.weighting == 'none') & (C3.calib == 'raw')].set_index('rep').ROC_AUC
        for meth in ('undersample', 'smote'):
            x = C3[(C3.model == m) & (C3.weighting == meth) & (C3.calib == 'raw')].set_index('rep').ROC_AUC
            d = (x - base).values; r = nb(d); r.update(model=m, method=meth)
            rows.append(r)
    T = pd.DataFrame(rows)
    T['tost_p_holm'] = holm(T.tost_p.values); T['equivalent'] = T.tost_p_holm < .05
    T['label'] = np.where(T.equivalent, 'equivalent', np.where((T.ci_lo <= 0) & (T.ci_hi >= 0), 'INCONCLUSIVE', 'not equivalent'))
    print("\nH15(iii) AUC vs unweighted (N-B, TOST ±.005, Holm over 8):\n",
          T[['model', 'method', 'mean_delta', 'ci_lo', 'ci_hi', 'tost_p_holm', 'label']].round(4).to_string(index=False))
    T.round(5).to_csv(RESULTS_DIR / "phase3d_resampling_auc_tests.csv", index=False)
    ok = (min(i_ok.values()) >= 3) and (cal.xs('undersample', level='weighting').max() < .02) and \
         (cal.xs('smote', level='weighting').max() < .02) and bool(T.equivalent.all())
    print(f"\nH15 verdict: {'SUPPORTED' if ok else 'NOT SUPPORTED'}  (i) {min(i_ok.values())>=3}  (ii) "
          f"{max(cal.xs('undersample', level='weighting').max(), cal.xs('smote', level='weighting').max())<.02}  (iii) {bool(T.equivalent.all())}")
    # threshold moving vs resampling
    D3 = S[S.dataset == 'D3']
    cmp = D3.pivot_table(index=['model', 'weighting'], columns='setting', values=['F1_mean', 'MCC_mean', 'Recall_mean']).round(3)
    print("\nD3: F1/MCC/Recall by method and threshold setting:\n", cmp.to_string())
    for ds in ('D1',):
        print(f"\n{ds} (3 repeats) F1 by method/threshold:\n",
              S[S.dataset == ds].pivot_table(index=['model', 'weighting'], columns='setting', values='F1_mean').round(3).to_string())


if __name__ == "__main__":
    main()
