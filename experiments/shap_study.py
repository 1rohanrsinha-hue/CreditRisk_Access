"""Phase 4 / task B: attribution agreement across models (H21, frozen at b3240f4).
D3 (and the no-rate ablation cell for LightGBM/EBM), unweighted, raw scores,
3 repeats x 2,000 seeded test rows. Scales differ (log-odds for all models except
Random Forest, whose TreeExplainer output is on the probability scale) so ONLY
feature ranks / shares are compared. Stages: --stage run (resumable per rep) | collect."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import shap, xgboost as xgb
from catboost import Pool
from scipy.stats import spearmanr, kendalltau
from interpret.glassbox import ExplainableBoostingClassifier
from creditrisk.config import RESULTS_DIR, FIGURES_DIR, CACHE_DIR
from creditrisk.features import FrameEncoder, NOMINAL
from creditrisk.models import make_models
from creditrisk.repeats import get_dataset, split, BASE_SEED
from creditrisk.woe import WoEScorecard

SHAP_DIR = CACHE_DIR / "shap"
REPS = 3
N_ROWS = 2000
CELLS = {'with_rate': 'D3', 'no_rate': 'D3:ir0:csfico'}   # no_rate = ablation cell D (rate out, fico kept)
CELL_F = {'no_rate_no_score': 'D3:ir0:csnone'}             # ablation cell F (rate and score out)
MAIN = ['LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting', 'Random Forest (tuned)', 'Logistic Regression',
        'WoE Scorecard', 'EBM']
BOOSTERS = ['LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting']


def tuned_params(model):
    P = json.load(open(RESULTS_DIR / "optuna_params.json"))
    ds = [v for k, v in P.items() if k.startswith(model + "|")]
    out = {}
    for k in ds[0]:
        vals = [d[k] for d in ds]
        if isinstance(vals[0], str):
            out[k] = max(set(vals), key=vals.count)
        elif isinstance(vals[0], int):
            out[k] = int(round(float(np.median(vals))))
        else:
            out[k] = float(np.median(vals))
    return out


def build(model, ytr, params, n_jobs, seed):
    pipe = make_models(ytr, seed, n_jobs, None, columns=None, weighting='none')[model]
    if params:
        pipe.named_steps['clf'].set_params(**params)
    return pipe


def fold_in(df_cols, orig):
    """map encoded column name -> original feature (handles one-hot 'c=k')."""
    return [c.split('=')[0] if c.split('=')[0] in orig else c for c in df_cols]


def agg(C, cols, orig):
    m = fold_in(cols, orig)
    out = pd.DataFrame({f: C[:, [i for i, g in enumerate(m) if g == f]].sum(1) for f in orig})
    return out


def contrib(name, Xtr, ytr, Xte, seed, n_jobs):
    """Return (n x features) DataFrame of per-loan contributions on the original features, plus scale label."""
    orig = list(Xtr.columns)
    if name in ('LightGBM', 'LightGBM (tuned)'):
        pipe = build('LightGBM', ytr, tuned_params('LightGBM') if 'tuned' in name else None, n_jobs, seed).fit(Xtr, ytr)
        Z = pipe.named_steps['enc'].transform(Xte)
        C = pipe.named_steps['clf'].predict(Z, pred_contrib=True)[:, :-1]
        return agg(C, list(Z.columns), orig), 'log-odds'
    if name == 'XGBoost':
        pipe = build('XGBoost', ytr, None, n_jobs, seed).fit(Xtr, ytr)
        Z = pipe.named_steps['enc'].transform(Xte)
        C = pipe.named_steps['clf'].get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
        return agg(C, list(Z.columns), orig), 'log-odds'
    if name == 'CatBoost':
        pipe = build('CatBoost', ytr, None, n_jobs, seed).fit(Xtr, ytr)
        enc = pipe.named_steps['enc']; Z = enc.transform(Xte)
        pool = Pool(Z, cat_features=[c for c in Z.columns if c in NOMINAL])
        C = pipe.named_steps['clf'].get_feature_importance(pool, type='ShapValues')[:, :-1]
        return agg(C, list(Z.columns), orig), 'log-odds'
    if name == 'Gradient Boosting':
        pipe = build('Gradient Boosting', ytr, None, n_jobs, seed).fit(Xtr, ytr)
        Z = pipe.named_steps['enc'].transform(Xte)
        C = shap.TreeExplainer(pipe.named_steps['clf']).shap_values(Z)
        return agg(np.asarray(C), list(Z.columns), orig), 'log-odds'
    if name == 'Random Forest (tuned)':
        pipe = build('Random Forest', ytr, tuned_params('Random Forest'), n_jobs, seed).fit(Xtr, ytr)
        Z = pipe.named_steps['enc'].transform(Xte)
        C = shap.TreeExplainer(pipe.named_steps['clf']).shap_values(Z)
        C = np.asarray(C[1] if isinstance(C, list) else (C[:, :, 1] if np.ndim(C) == 3 else C))
        return agg(C, list(Z.columns), orig), 'probability'
    if name == 'Logistic Regression':
        pipe = build('Logistic Regression', ytr, None, n_jobs, seed).fit(Xtr, ytr)
        enc = pipe.named_steps['enc']
        bg = enc.transform(Xtr.sample(5000, random_state=seed))
        ex = shap.LinearExplainer(pipe.named_steps['clf'], shap.maskers.Independent(bg.values, max_samples=5000))
        Z = enc.transform(Xte)
        return agg(np.asarray(ex.shap_values(Z.values)), list(Z.columns), orig), 'log-odds'
    if name == 'WoE Scorecard':
        w = WoEScorecard(seed=seed).fit(Xtr, ytr.values)
        mu = w.transform(Xtr)[w.cols_].mean()
        Z = w.transform(Xte)[w.cols_]
        C = (Z - mu).values * w.lr_.coef_[0]
        return agg(C, list(w.cols_), orig), 'log-odds'
    if name in ('EBM', 'EBM (no interactions)'):
        enc = FrameEncoder('catboost').fit(Xtr, ytr)
        ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=n_jobs, interactions=0 if 'no inter' in name else 10)
        ebm.fit(enc.transform(Xtr), ytr.values)
        T = ebm.eval_terms(enc.transform(Xte))
        names = list(ebm.feature_names_in_)
        C = pd.DataFrame(0.0, index=range(len(Xte)), columns=orig)
        for j, tf in enumerate(ebm.term_features_):
            for f in tf:
                C[names[f]] += T[:, j] / len(tf)         # interaction terms split equally between their two features
        return C, 'log-odds'
    raise ValueError(name)


def run(n_jobs, reps):
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    for rep in reps:
        out = SHAP_DIR / f"rep{rep}.npz"
        if out.exists():
            continue
        t0 = time.time(); arrays = {}; rows = []
        for cell, ds in CELLS.items():
            X, y, _ = get_dataset(ds)
            tr, tv, te = split(y, rep)
            seed = BASE_SEED + rep
            sel = np.sort(np.random.RandomState(seed).choice(te, N_ROWS, replace=False))
            Xtr, ytr, Xte = X.iloc[tv], y.iloc[tv], X.iloc[sel]
            models = MAIN if cell == 'with_rate' else ['LightGBM', 'EBM']
            extra = ['LightGBM (tuned)', 'EBM (no interactions)'] if cell == 'with_rate' else []
            for m in models + extra:
                t1 = time.time()
                C, scale = contrib(m, Xtr, ytr, Xte, seed, n_jobs)
                share = C.abs().mean() / C.abs().mean().sum()
                for f in C.columns:
                    rows.append(dict(cell=cell, rep=rep, model=m, feature=f, mean_abs=float(C[f].abs().mean()),
                                     share=float(share[f]), scale=scale))
                if cell == 'with_rate' and m in ('LightGBM (tuned)', 'EBM', 'EBM (no interactions)'):
                    arrays[m.replace(' ', '_')] = C.values.astype(np.float32)
                print(f"[shap] rep{rep} {cell} {m} {time.time()-t1:.0f}s", flush=True)
            if cell == 'with_rate':
                arrays['sel'] = sel; arrays['cols'] = np.array(list(Xte.columns))
        pd.DataFrame(rows).to_csv(SHAP_DIR / f"rep{rep}.csv", index=False)
        np.savez_compressed(str(out) + ".tmp.npz", **arrays); os.replace(str(out) + ".tmp.npz", out)
        print(f"[shap] rep{rep} done {time.time()-t0:.0f}s", flush=True)


def run_cell_f(n_jobs, reps):
    for rep in reps:
        out = SHAP_DIR / f"rep{rep}_F.csv"
        if out.exists():
            continue
        rows = []
        for cell, ds in CELL_F.items():
            X, y, _ = get_dataset(ds); tr, tv, te = split(y, rep); seed = BASE_SEED + rep
            sel = np.sort(np.random.RandomState(seed).choice(te, N_ROWS, replace=False))
            for m in ('LightGBM', 'EBM'):
                C, scale = contrib(m, X.iloc[tv], y.iloc[tv], X.iloc[sel], seed, n_jobs)
                share = C.abs().mean() / C.abs().mean().sum()
                rows += [dict(cell=cell, rep=rep, model=m, feature=f, mean_abs=float(C[f].abs().mean()), share=float(share[f]), scale=scale) for f in C.columns]
                print(f"[shap-F] rep{rep} {m}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


def collect():
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    R = pd.concat([pd.read_csv(SHAP_DIR / f"rep{r}.csv") for r in range(REPS)] +
                  [pd.read_csv(SHAP_DIR / f"rep{r}_F.csv") for r in range(REPS) if (SHAP_DIR / f"rep{r}_F.csv").exists()])
    R.round(5).to_csv(RESULTS_DIR / "phase4_shap.csv", index=False)
    W = R[R.cell == 'with_rate']
    S = W[W.model.isin(MAIN)].groupby(['model', 'feature']).share.mean().unstack(0)[MAIN]
    S = S.loc[S.mean(1).sort_values(ascending=False).index]
    print("=== mean |contribution| share per feature (mean of 3 reps, with-rate) ===\n", (100 * S).round(1).to_string())
    # rank agreement
    pairs = []
    for r in range(REPS):
        M = W[(W.rep == r) & W.model.isin(MAIN)].pivot(index='feature', columns='model', values='mean_abs')[MAIN]
        for i, a in enumerate(MAIN):
            for b in MAIN[i + 1:]:
                pairs.append(dict(rep=r, a=a, b=b, spearman=spearmanr(M[a], M[b])[0], kendall=kendalltau(M[a], M[b])[0]))
    PR = pd.DataFrame(pairs); PM = PR.groupby(['a', 'b'])[['spearman', 'kendall']].mean().reset_index()
    PM['min_spearman_over_reps'] = PR.groupby(['a', 'b']).spearman.min().values
    PM.round(4).to_csv(RESULTS_DIR / "phase4_shap_rankcorr.csv", index=False)
    print("\n=== pairwise rank correlation (mean over reps) ===\n", PM.round(3).to_string(index=False))
    # H21
    bb = PM[PM.a.isin(BOOSTERS) & PM.b.isin(BOOSTERS)]
    bl = PM[((PM.a.isin(BOOSTERS)) & PM.b.isin(['Logistic Regression', 'WoE Scorecard'])) |
            ((PM.b.isin(BOOSTERS)) & PM.a.isin(['Logistic Regression', 'WoE Scorecard']))]
    top1 = {m: (S[m].idxmax(), [W[(W.rep == r) & (W.model == m)].set_index('feature').share.idxmax() for r in range(REPS)]) for m in MAIN}
    c1 = bool((bb.spearman >= .8).all()); c2 = bool((bl.spearman >= .6).all())
    c3 = all(top1[m][0] == 'interest_rate' and all(t == 'interest_rate' for t in top1[m][1]) for m in MAIN)
    print(f"\nH21 (1) booster pairs min Spearman {bb.spearman.min():.3f} (>= .8: {c1}); "
          f"(2) booster vs LR/WoE min {bl.spearman.min():.3f} (>= .6: {c2}); (3) interest_rate first everywhere: {c3}  {top1}")
    print("H21 verdict:", "SUPPORTED" if (c1 and c2 and c3) else "NOT SUPPORTED")
    pd.DataFrame([dict(clause='1 boosters >= .8', value=bb.spearman.min(), holds=c1),
                  dict(clause='2 booster vs LR/WoE >= .6', value=bl.spearman.min(), holds=c2),
                  dict(clause='3 interest_rate first in all', value=np.nan, holds=c3)]).round(4).to_csv(RESULTS_DIR / "phase4_h21.csv", index=False)
    # stability across reps
    st = []
    for m in MAIN:
        M = W[W.model == m].pivot(index='feature', columns='rep', values='mean_abs')
        sp = [spearmanr(M[a], M[b])[0] for a in range(REPS) for b in range(a + 1, REPS)]
        t3 = [set(M[r].nlargest(3).index) for r in range(REPS)]
        st.append(dict(model=m, mean_spearman_across_reps=np.mean(sp), top3_identical_all_reps=all(t == t3[0] for t in t3),
                       top3='|'.join(S[m].nlargest(3).index)))
    ST = pd.DataFrame(st); print("\n=== stability across the 3 repeats ===\n", ST.round(3).to_string(index=False))
    ST.round(4).to_csv(RESULTS_DIR / "phase4_shap_stability.csv", index=False)
    # with vs without rate (LightGBM, EBM)
    N = R[R.model.isin(['LightGBM', 'EBM'])]
    cmp_ = []
    for m, cell_b in [(m, c) for m in ('LightGBM', 'EBM') for c in ('no_rate', 'no_rate_no_score') if (N.cell == c).any()]:
        a = N[(N.cell == 'with_rate') & (N.model == m)].groupby('feature').share.mean()
        b = N[(N.cell == cell_b) & (N.model == m)].groupby('feature').share.mean()
        shared = [f for f in b.index if f in a.index]
        ra = (a[shared] / a[shared].sum()); rb = (b[shared] / b[shared].sum())
        print(f"\n[{cell_b}] {m}: with-rate top5 {list(a.nlargest(5).round(3).items())}\n{m}: no-rate  top5 {list(b.nlargest(5).round(3).items())}"
              f"\n   Spearman of shared-feature ranks (excl. interest_rate): {spearmanr(ra, rb[ra.index])[0]:.3f}")
        cmp_.append(dict(model=m, vs_cell=cell_b, spearman_shared_features=spearmanr(ra, rb[ra.index])[0], rate_share_with=a.get('interest_rate'),
                         top3_with='|'.join(a.nlargest(3).index), top3_without='|'.join(b.nlargest(3).index)))
    pd.DataFrame(cmp_).round(4).to_csv(RESULTS_DIR / "phase4_shap_with_without_rate.csv", index=False)
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = ['LightGBM', 'CatBoost', 'Logistic Regression', 'EBM']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharex=True)
    for ax, m in zip(axes, show):
        top = S[m].nlargest(10)[::-1]
        err = [W[(W.model == m) & (W.feature == f)].share.std() for f in top.index]
        ax.barh(top.index, 100 * top.values, xerr=100 * np.array(err), color='#4c78a8', ecolor='k', error_kw=dict(lw=.8))
        ax.set_title(m, fontsize=9); ax.set_xlabel("share of mean |contribution| (%)", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle("Top-10 features by share of mean |attribution| (D3, 3 repeats × 2,000 rows; bars = repeat SD). Scales differ; ranks only", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "phase4_shap_top10.png", dpi=300); plt.close(fig)
    Mx = np.eye(len(MAIN))
    for i, a in enumerate(MAIN):
        for j, b in enumerate(MAIN):
            if i != j:
                row = PM[((PM.a == a) & (PM.b == b)) | ((PM.a == b) & (PM.b == a))]
                Mx[i, j] = row.spearman.iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(Mx, vmin=.4, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(MAIN))); ax.set_yticks(range(len(MAIN)))
    ax.set_xticklabels(MAIN, rotation=45, ha='right', fontsize=7); ax.set_yticklabels(MAIN, fontsize=7)
    for i in range(len(MAIN)):
        for j in range(len(MAIN)):
            ax.text(j, i, f"{Mx[i, j]:.2f}", ha='center', va='center', fontsize=7, color='w' if Mx[i, j] < .75 else 'k')
    fig.colorbar(im, label="Spearman of feature ranks"); ax.set_title("Mean |attribution| rank correlation (with-rate, D3)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "phase4_shap_rankcorr.png", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "cellF", "collect"], required=True)
    ap.add_argument("--reps", type=int, nargs="*", default=list(range(REPS)))
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    {"run": lambda: run(a.n_jobs, a.reps), "cellF": lambda: run_cell_f(a.n_jobs, a.reps), "collect": collect}[a.stage]()
