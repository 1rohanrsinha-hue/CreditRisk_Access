"""Phase 3D / task C: WoE scorecard LR and EBM (H18). D3, 5 repeats, with and
without interest_rate. Same protocol as the other models (5-fold OOF over
train+val + refit). Stages: score (resumable, --worker) | analyze."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedKFold
from interpret.glassbox import ExplainableBoostingClassifier
from creditrisk.config import RESULTS_DIR
from creditrisk.features import FrameEncoder
from creditrisk.repeats import get_dataset, score_path, split, load_score, BASE_SEED
from creditrisk.stats import summarize
from creditrisk.study import analyse
from creditrisk.woe import WoEScorecard
from stats_study import nb, holm

MODELS = ['WoE Scorecard', 'EBM']
CELLS = {'with_rate': 'D3', 'no_rate': 'D3:ir0:csfico'}
BOOST = ['Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
REPS = 5


def fit_predict(model, Xtr, ytr, Xte, seed, n_jobs):
    if model == 'WoE Scorecard':
        return WoEScorecard(seed=seed).fit(Xtr, ytr.values).predict_proba(Xte)[:, 1]
    enc = FrameEncoder('catboost').fit(Xtr, ytr)
    ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=n_jobs, interactions=10)
    ebm.fit(enc.transform(Xtr), ytr.values)
    return ebm.predict_proba(enc.transform(Xte))[:, 1]


def score_unit(ds, rep, model, X, y, n_jobs):
    path = score_path(ds, rep, model, 'none')
    if path.exists():
        return
    t0 = time.time(); seed = BASE_SEED + rep
    tr, tv, te = split(y, rep)
    Xtv, ytv = X.iloc[tv], y.iloc[tv]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(tv))
    for a, b in skf.split(Xtv, ytv):
        oof[b] = fit_predict(model, Xtv.iloc[a], ytv.iloc[a], Xtv.iloc[b], seed, n_jobs)
    te_s = fit_predict(model, Xtv, ytv, X.iloc[te], seed, n_jobs)
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, oof=oof.astype(np.float32), te=te_s.astype(np.float32),
                        idx_tv=tv.astype(np.int32), idx_te=te.astype(np.int32), secs=np.array([time.time() - t0]))
    os.replace(tmp, path)
    print(f"[interp] {ds} rep={rep} {model} {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "score":
        for cell, ds in CELLS.items():
            X, y, _ = get_dataset(ds)
            for rep in range(REPS):
                if rep % a.n_workers != a.worker:
                    continue
                for m in MODELS:
                    score_unit(ds, rep, m, X, y, a.n_jobs)
        print("[done] worker", a.worker); return
    pd.set_option('display.width', 250)
    out, tests = [], []
    for cell, ds in CELLS.items():
        R, C = analyse(ds, list(range(REPS)), tag='_interp', models=MODELS, weightings=('none',), with_profit=False)
        C = C[C.model.isin(MODELS)].copy()
        R = R[R.model.isin(MODELS) & (R.rule == 'youden')]
        pr = {(r.rep, r.model, r.calib): r.PR_AUC for r in R.itertuples()}
        C['PR_AUC'] = [pr.get((r.rep, r.model, r.calib), np.nan) for r in C.itertuples()]
        # booster reference from the cached scores of the same cell (D3 or ablation cell D)
        ref = {}
        for m in BOOST:
            try:
                from sklearn.metrics import roc_auc_score
                X, y, _ = get_dataset(ds)
                ref[m] = [roc_auc_score(y.values[load_score(ds, r, m, 'none')['idx_te']], load_score(ds, r, m, 'none')['te']) for r in range(REPS)]
            except FileNotFoundError:
                pass
        best = max(ref, key=lambda m: np.mean(ref[m]))
        ca = C[(C.calib == 'raw')].pivot(index='rep', columns='model', values='ROC_AUC')
        fam = []
        for m in MODELS:
            d = ca[m].values - np.array(ref[best]); r = nb(d); r.update(cell=cell, model=m, best_boosting=best,
                                                                       best_auc=np.mean(ref[best]), model_auc=ca[m].mean()); fam.append(r)
        T = pd.DataFrame(fam); T['tost_p_holm'] = holm(T.tost_p.values); T['equivalent'] = T.tost_p_holm < .05
        T['label'] = np.where(T.equivalent, 'equivalent', np.where((T.ci_lo <= 0) & (T.ci_hi >= 0), 'INCONCLUSIVE', 'not equivalent'))
        tests.append(T)
        S = summarize(C.assign(abs_t_pi=(C.t_star - C.pi).abs()), ['model', 'calib'],
                      ['ROC_AUC', 'PR_AUC', 'Brier', 'ECE', 't_star', 'pi', 'abs_t_pi'])
        S.insert(0, 'cell', cell); out.append(S)
        print(f"\n=== {cell}: best boosting ({best}) AUC {np.mean(ref[best]):.4f}; boosters {({m: round(np.mean(v),4) for m,v in ref.items()})} ===")
        print(S[S.calib != 'isotonic'].round(4).to_string(index=False))
        print(T[['model', 'mean_delta', 'ci_lo', 'ci_hi', 'tost_p_holm', 'equivalent', 'label']].round(4).to_string(index=False))
    pd.concat(out).round(5).to_csv(RESULTS_DIR / "phase3d_interp.csv", index=False)
    TT = pd.concat(tests); TT.round(5).to_csv(RESULTS_DIR / "phase3d_interp_tests.csv", index=False)
    # H18 primary = with_rate
    w = TT[TT.cell == 'with_rate'].set_index('model')
    ebm_eq = bool(w.loc['EBM', 'equivalent'])
    sc = w.loc['WoE Scorecard']
    sc_ok = (sc.mean_delta <= -0.005) and (not sc.equivalent)
    print(f"\nH18 (with rate): EBM equivalent: {ebm_eq}; WoE scorecard mean delta {sc.mean_delta:.4f} -> "
          f"{'>=.005 below' if sc_ok else ('INCONCLUSIVE' if sc.label == 'INCONCLUSIVE' else 'not >=.005 below')}")
    print("H18 verdict:", "SUPPORTED" if (ebm_eq and sc_ok) else "NOT SUPPORTED (or partially; see labels)")
    # points table (rep 0, with rate)
    X, y, _ = get_dataset('D3'); tr, tv, te = split(y, 0)
    w0 = WoEScorecard(seed=BASE_SEED).fit(X.iloc[tv], y.iloc[tv].values)
    pt = w0.points_table(top=5); pt.round(4).to_csv(RESULTS_DIR / "phase3d_scorecard_points.csv", index=False)
    print("\nscorecard top-5 features by |coef| (rep 0):", list(pt.feature.unique()))


if __name__ == "__main__":
    main()
