"""Phase 3C / task C: endogenous-feature ablation on D3 (6 cells; formerly "leakage ablation"):
interest_rate is known at origination but is the lender's own risk price, so it is endogenous, not leakage.
  A rate in,  fico (headline)   B rate in,  none   C rate in,  grade-derived
  D rate out, fico              E rate out, grade-derived (PARTIAL removal: embeds rate)   F rate out, none
Stages: score (resumable, parallel via --worker) | analyze."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.capacity import approve_dollar, outcome
from creditrisk.data import full_frame, profit_p2
from creditrisk.models import fit_model
from creditrisk.repeats import get_dataset, load_score, run_scoring, split
from creditrisk.stats import summarize
from creditrisk.study import analyse

MODELS = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
CELLS = {'A': 'D3', 'B': 'D3:ir1:csnone', 'C': 'D3:ir1:csgd', 'D': 'D3:ir0:csfico',
         'E': 'D3:ir0:csgd', 'F': 'D3:ir0:csnone'}
LABEL = {'A': 'rate in, fico (headline)', 'B': 'rate in, no score', 'C': 'rate in, grade-derived',
         'D': 'rate out, fico', 'E': 'rate out, grade-derived (partial removal)', 'F': 'rate out, no score'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    reps = list(range(a.repeats))
    if a.stage == "score":
        t0 = time.time()
        for cell, ds in CELLS.items():
            if cell == 'A':
                continue                      # headline cell reuses the D3 scores
            run_scoring(ds, reps, models=MODELS, weightings=('none',), n_jobs=a.n_jobs, worker=(a.worker, a.n_workers))
        print(f"[done] ablation scoring worker {a.worker} {time.time()-t0:.0f}s")
        return
    # ------------------------------------------------------------- analyze
    X0, y0, _ = get_dataset('D3'); yv = y0.values
    F = full_frame().loc[X0.index]
    funded = F.funded_amnt.values.astype(float); P = profit_p2(F, 0.03).values
    rows, cap = [], []
    for cell, ds in CELLS.items():
        R, C = analyse(ds, reps, models=MODELS, weightings=('none',), with_profit=True)
        # cell A reuses the cached D3 analysis (all models, both weightings): filter
        R = R[(R.weighting == 'none') & R.model.isin(MODELS) & (R.rep.isin(reps))]
        C = C[(C.weighting == 'none') & C.model.isin(MODELS) & (C.rep.isin(reps))]
        pr = R[(R.rule == 'youden') & (R.calib == 'raw')].set_index(['rep', 'model']).PR_AUC
        C = C.copy()
        C['PR_AUC'] = [pr.get((r.rep, r.model), np.nan) for r in C.itertuples()]
        C['abs_t_pi'] = (C.t_star - C.pi).abs()
        C['cell'] = cell
        rows.append(C)
        for rep in reps:                     # default-risk ranking, dollar capacity k=50%, P2-3%
            for m in MODELS:
                z = load_score(ds, rep, m, 'none'); te = z['idx_te']
                a_ = approve_dollar(z['te'].astype(float), funded[te], 0.5)
                o = outcome(a_, yv[te], P[te], funded[te])
                base = P[te].sum() / len(te) * 1000
                cap.append(dict(cell=cell, rep=rep, model=m, gain_dollar_k50=o['profit_per_1000'] - 0.5 * base,
                                roi=o['roi'], roi_all=P[te].sum() / funded[te].sum(), bad_dollar=o['bad_dollar']))
    C = pd.concat(rows, ignore_index=True); K = pd.DataFrame(cap)
    S = (summarize(C, ['cell', 'model', 'calib'], ['ROC_AUC', 'PR_AUC', 'Brier', 'ECE', 'pi', 't_star', 'abs_t_pi'])
         .merge(summarize(K, ['cell', 'model'], ['gain_dollar_k50', 'roi', 'bad_dollar']).assign(calib='raw'),
                on=['cell', 'model', 'calib'], how='left'))
    S['cell_label'] = S.cell.map(LABEL)
    S.round(5).to_csv(RESULTS_DIR / "phase3c_ablation.csv", index=False)
    pd.set_option('display.width', 250)
    # cell-level table
    raw = C[C.calib == 'raw'].groupby('cell')[['ROC_AUC', 'PR_AUC', 'Brier', 'ECE', 'pi']].mean()
    gap = C[C.calib != 'raw'].groupby('cell').abs_t_pi.mean().rename('mean|t*-pi| calibrated')
    gmax = C[C.calib != 'raw'].groupby(['cell', 'model', 'calib']).abs_t_pi.mean().groupby('cell').max().rename('max|t*-pi|')
    kk = K.groupby('cell')[['gain_dollar_k50', 'roi', 'bad_dollar']].mean()
    T = pd.concat([raw, gap, gmax, kk], axis=1); T['label'] = pd.Series(LABEL)
    print(T.round(4).to_string())
    T.round(5).to_csv(RESULTS_DIR / "phase3c_ablation_cells.csv")
    # H10, H11
    h10 = C[C.calib != 'raw'].groupby(['cell', 'model', 'calib']).abs_t_pi.mean()
    print(f"\nH10: max over (cell, model, calibrator) of mean|t*-pi| = {h10.max():.4f}  -> {'SUPPORTED' if h10.max() < 0.02 else 'REJECTED'}")
    auc = C[C.calib == 'raw'].groupby('cell').ROC_AUC.mean()
    gA, gF = K[K.cell == 'A'].gain_dollar_k50.mean(), K[K.cell == 'F'].gain_dollar_k50.mean()
    print(f"H11: dAUC(A-F) = {auc['A']-auc['F']:.4f} (needs > .005); gain A = {gA:,.0f}, gain F = {gF:,.0f} (needs F > A)"
          f" -> {'SUPPORTED' if (auc['A']-auc['F'] > .005 and gF > gA) else 'REJECTED'}")
    print("per-cell mean AUC:", auc.round(4).to_dict())
    # ------------------------------------------------ feature importance (SHAP, LightGBM, rep 0)
    import shap
    imp = []
    tr, tv, te = split(y0, 0)
    for cell in ('A', 'F'):
        X, y, _ = get_dataset(CELLS[cell])
        m = fit_model('LightGBM', X.iloc[tv], y.iloc[tv], 'none', 42, 4)
        enc, clf = m.named_steps['enc'], m.named_steps['clf']
        rs = np.random.RandomState(0).choice(te, 2000, replace=False)
        Z = enc.transform(X.iloc[rs])
        sv = shap.TreeExplainer(clf).shap_values(Z)
        sv = sv[1] if isinstance(sv, list) else sv
        for f, v in zip(Z.columns, np.abs(sv).mean(axis=0)):
            imp.append(dict(cell=cell, feature=f, mean_abs_shap=float(v)))
    I = pd.DataFrame(imp)
    I['share'] = I.mean_abs_shap / I.groupby('cell').mean_abs_shap.transform('sum')
    I.sort_values(['cell', 'mean_abs_shap'], ascending=[True, False]).round(5).to_csv(
        RESULTS_DIR / "phase3c_ablation_features.csv", index=False)
    for cell in ('A', 'F'):
        print(f"\ntop features, cell {cell}:\n", I[I.cell == cell].nlargest(6, 'mean_abs_shap')[['feature', 'mean_abs_shap', 'share']].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
