"""Phase 3D / task E: scale check on the full 667k mature population (H19) and
a flagged 1.345M strict-set sensitivity (threshold/calibration only)."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR
from creditrisk.repeats import run_scoring
from creditrisk.stats import summarize
from creditrisk.study import analyse

MODELS = ['Logistic Regression', 'LightGBM', 'XGBoost', 'CatBoost']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "score":
        t0 = time.time()
        run_scoring('D3FULL', [0, 1, 2], models=MODELS, weightings=('none', 'balanced'), n_jobs=a.n_jobs)
        print(f"[stage] D3FULL scoring {time.time()-t0:.0f}s", flush=True); t1 = time.time()
        run_scoring('STRICTFULL', [0], models=['Logistic Regression', 'LightGBM'], weightings=('none', 'balanced'), n_jobs=a.n_jobs)
        print(f"[stage] STRICTFULL scoring {time.time()-t1:.0f}s", flush=True)
        return
    t0 = time.time()
    out = []
    pd.set_option('display.width', 250)
    for ds, reps, label in (('D3FULL', [0, 1, 2], 'D3 667k'), ('STRICTFULL', [0], 'strict 1.345M (sensitivity; 2016+ censored)')):
        mods = MODELS if ds == 'D3FULL' else ['Logistic Regression', 'LightGBM']
        R, C = analyse(ds, reps, models=mods, weightings=('none', 'balanced'), with_profit=False)
        R = R[R.model.isin(mods)]; C = C[C.model.isin(mods)]
        C = C.assign(abs_t_pi=lambda d: (d.t_star - d.pi).abs(), abs_t_half=lambda d: (d.t_star - .5).abs())
        S = summarize(C, ['model', 'weighting', 'calib'], ['ROC_AUC', 'Brier', 'ECE', 't_star', 'pi', 'abs_t_pi', 'abs_t_half'])
        S.insert(0, 'dataset', ds); out.append(S)
        rec = R[(R.rule.isin(['fixed_0.5', 'youden'])) & (R.calib == 'raw')].pivot_table(
            index=['model', 'weighting'], columns='rule', values='Recall')
        print(f"\n=== {label}: reps={len(reps)}, pi={C.pi.mean():.4f} ===")
        print(S[S.calib != 'raw'].pivot_table(index=['model', 'weighting'], columns='calib', values='abs_t_pi_mean').round(4).to_string())
        print(rec.round(3).to_string())
        if ds == 'D3FULL':
            tuned = rec['youden'].unstack('weighting')
            gap = (tuned['none'] - tuned['balanced']).abs()
            auc = C[(C.calib == 'raw') & (C.weighting == 'none')].groupby('model').ROC_AUC.mean()
            # D3-200k reference (10 repeats, cached)
            _, C2 = analyse('D3', list(range(10)), with_profit=True)
            ref = C2[(C2.calib == 'raw') & (C2.weighting == 'none')].groupby('model').ROC_AUC.mean()
            cmp = pd.DataFrame({'AUC_667k': auc, 'AUC_200k': ref.reindex(auc.index)}); cmp['diff'] = cmp.AUC_667k - cmp.AUC_200k
            print("\nAUC 667k vs 200k:\n", cmp.round(4).to_string())
            ha = C[C.calib != 'raw'].groupby(['model', 'weighting', 'calib']).abs_t_pi.mean().max()
            hb = gap.mean(); hc = cmp.loc[['Logistic Regression', 'LightGBM', 'XGBoost'], 'diff'].abs().max()
            print(f"\nH19 (a) max mean|t*-pi| calibrated = {ha:.4f} (<.02: {ha<.02}); (b) mean|tuned recall none-bal| = {hb:.4f} "
                  f"(<.01: {hb<.01}); (c) max |dAUC| LR/LGBM/XGB = {hc:.4f} (<.01: {hc<.01})")
            print("H19 verdict:", "SUPPORTED" if (ha < .02 and hb < .01 and hc < .01) else "NOT SUPPORTED")
            cmp.round(5).to_csv(RESULTS_DIR / "phase3d_scale_auc.csv")
    pd.concat(out).round(5).to_csv(RESULTS_DIR / "phase3d_scale.csv", index=False)
    print(f"[stage] analysis {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
