"""Phase 6 / task C: H25 (frozen at e647452). The deployed configuration (tuned LightGBM with the median
Optuna parameters, isotonic calibration on 5-fold OOF scores of the training window) on the rolling-origin
OOT folds. Stages: --stage run --T <year> (resumable) | collect.
Self-contained on journal-extension: the same pipeline as creditrisk/serving.py (app-v2) is rebuilt from
make_models('LightGBM') + the median parameters."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from creditrisk.config import RESULTS_DIR, CACHE_DIR
from creditrisk.calibration import Isotonic, ece
from creditrisk.models import make_models
from oot_study import fold_indices, usable_columns, FOLDS
from term36_refit import fold36
import oot_scores as OS

OUT = CACHE_DIR / "deploy_oot"
SEEDS = (42, 43, 44); NB = 1000


def median_params(model='LightGBM'):
    P = json.load(open(RESULTS_DIR / "optuna_params.json"))
    ds = [v for k, v in P.items() if k.startswith(model + "|")]
    out = {}
    for k in ds[0]:
        vals = [d[k] for d in ds]
        out[k] = max(set(vals), key=vals.count) if isinstance(vals[0], str) else (
            int(round(float(np.median(vals)))) if isinstance(vals[0], int) else float(np.median(vals)))
    return out


def pipe(ytr, params, seed, n_jobs):
    p = make_models(ytr, seed, n_jobs, None, columns=None, weighting='none')['LightGBM']
    p.named_steps['clf'].set_params(**params)
    return p


def fit_unit(T, scope, seed, X, y, yr, tm, params, n_jobs):
    path = OUT / f"{T}_{scope}_s{seed}.npz"
    if path.exists():
        return
    tr, ca, te = (fold_indices(T, yr, tm) if scope == 'protocol' else fold36(T, yr, tm))
    cols, dropped = usable_columns(X, tr); Xc = X[cols]
    fit_i, hold_i = train_test_split(tr, test_size=0.20, random_state=seed, stratify=y[tr])
    Xf, yf = Xc.iloc[fit_i], pd.Series(y[fit_i])
    oof = np.zeros(len(fit_i))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xf, yf):
        oof[b] = pipe(yf.iloc[a], params, seed, n_jobs).fit(Xf.iloc[a], yf.iloc[a]).predict_proba(Xf.iloc[b])[:, 1]
    cal = Isotonic().fit(oof, yf.values)
    m = pipe(yf, params, seed, n_jobs).fit(Xf, yf)
    s_h, s_t = m.predict_proba(Xc.iloc[hold_i])[:, 1], m.predict_proba(Xc.iloc[te])[:, 1]
    np.savez_compressed(str(path) + ".tmp.npz", p_hold=cal.predict(s_h), s_hold=s_h, y_hold=y[hold_i],
                        p_te=cal.predict(s_t), s_te=s_t, y_te=y[te], n_fit=len(fit_i), dropped=np.array(dropped, dtype=object))
    os.replace(str(path) + ".tmp.npz", path)


def boot(Z, key_p, key_y, rng):
    """one resample (same row positions for all 3 seeds, each seed with its OWN labels: the in-time hold-out
    differs by seed) -> seed-mean AUC, ECE, Brier."""
    n = len(Z[0][key_y]); i = rng.integers(0, n, n)
    return (np.mean([roc_auc_score(z[key_y][i], z[key_p][i]) for z in Z]), np.mean([ece(z[key_y][i], z[key_p][i]) for z in Z]),
            np.mean([brier_score_loss(z[key_y][i], z[key_p][i]) for z in Z]))


def run(T, n_jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    t = OS.oot_table(); X, y, yr, tm = t['X'], t['y'], t['yr'], t['tm']; params = median_params()
    for scope in ('protocol', '36m'):
        if scope == '36m' and T == 2015:
            continue                                     # identical to the protocol fold
        for seed in SEEDS:
            t0 = time.time(); fit_unit(T, scope, seed, X, y, yr, tm, params, n_jobs)
            print(f"[deploy-oot] T={T} {scope} s{seed} {time.time()-t0:.0f}s", flush=True)
        Z = [np.load(OUT / f"{T}_{scope}_s{s}.npz", allow_pickle=True) for s in SEEDS]
        Z = [{k: z[k] for k in ('p_hold', 'y_hold', 'p_te', 'y_te', 's_hold', 's_te')} for z in Z]
        rng = np.random.default_rng(T)
        bh = np.array([boot(Z, 'p_hold', 'y_hold', rng) for _ in range(NB)]); bt = np.array([boot(Z, 'p_te', 'y_te', rng) for _ in range(NB)])
        rows = []
        for seed, z in zip(SEEDS, Z):
            for nm, p, yy in (('in_time', z['p_hold'], z['y_hold']), ('oot', z['p_te'], z['y_te'])):
                rows.append(dict(T=T, scope=scope, seed=seed, set=nm, n=len(yy), AUC=roc_auc_score(yy, p), ECE=ece(yy, p), Brier=brier_score_loss(yy, p),
                                 prevalence=float(yy.mean()), mean_pd=float(p.mean())))
        pd.DataFrame(rows).to_csv(OUT / f"seeds_{T}_{scope}.csv", index=False)
        S = pd.DataFrame(rows).groupby('set')[['AUC', 'ECE', 'Brier', 'prevalence', 'mean_pd']].mean()
        ratio = bt[:, 1] / bh[:, 1]; gap = bh[:, 0] - bt[:, 0]
        q = lambda a: tuple(np.quantile(a, [.025, .975]))
        row = dict(T=T, scope=scope, n_in_time=len(Z[0]['y_hold']), n_oot=len(Z[0]['y_te']),
                   AUC_in_time=S.loc['in_time', 'AUC'], AUC_in_ci_lo=q(bh[:, 0])[0], AUC_in_ci_hi=q(bh[:, 0])[1],
                   AUC_oot=S.loc['oot', 'AUC'], AUC_oot_ci_lo=q(bt[:, 0])[0], AUC_oot_ci_hi=q(bt[:, 0])[1],
                   AUC_gap=S.loc['in_time', 'AUC'] - S.loc['oot', 'AUC'], AUC_gap_ci_lo=q(gap)[0], AUC_gap_ci_hi=q(gap)[1],
                   ECE_in_time=S.loc['in_time', 'ECE'], ECE_in_ci_lo=q(bh[:, 1])[0], ECE_in_ci_hi=q(bh[:, 1])[1],
                   ECE_oot=S.loc['oot', 'ECE'], ECE_oot_ci_lo=q(bt[:, 1])[0], ECE_oot_ci_hi=q(bt[:, 1])[1],
                   ECE_ratio=S.loc['oot', 'ECE'] / S.loc['in_time', 'ECE'], ratio_ci_lo=q(ratio)[0], ratio_ci_hi=q(ratio)[1],
                   Brier_in_time=S.loc['in_time', 'Brier'], Brier_oot=S.loc['oot', 'Brier'],
                   prevalence_in_time=S.loc['in_time', 'prevalence'], prevalence_oot=S.loc['oot', 'prevalence'],
                   mean_pd_oot=S.loc['oot', 'mean_pd'])
        pd.DataFrame([row]).to_csv(OUT / f"summary_{T}_{scope}.csv", index=False)
        print(f"[deploy-oot] T={T} {scope}: AUC {row['AUC_in_time']:.4f}->{row['AUC_oot']:.4f}  ECE {row['ECE_in_time']:.4f}->{row['ECE_oot']:.4f}  ratio {row['ECE_ratio']:.1f}", flush=True)


def collect():
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    S = pd.concat([pd.read_csv(OUT / f"summary_{T}_{sc}.csv") for sc in ('protocol', '36m') for T in FOLDS if (OUT / f"summary_{T}_{sc}.csv").exists()])
    S.round(5).to_csv(RESULTS_DIR / "phase6_deploy_oot.csv", index=False)
    pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("seeds_*.csv"))]).round(5).to_csv(RESULTS_DIR / "phase6_deploy_oot_seeds.csv", index=False)
    print(S.round(4).T.to_string())
    P = S[S.scope == 'protocol'].set_index('T')
    c1 = int((P.ECE_ratio >= 5).sum()); c2 = bool(((P.AUC_gap >= .010) & (P.AUC_gap <= .030)).all())
    print(f"\nH25 clause 1: ECE ratio >= 5 in {c1}/3 folds ({P.ECE_ratio.round(1).to_dict()}) -> {c1 >= 2}")
    print(f"H25 clause 2: AUC gap in [.010,.030] in all folds ({P.AUC_gap.round(4).to_dict()}) -> {c2}")
    print("H25 verdict:", "SUPPORTED" if (c1 >= 2 and c2) else "NOT SUPPORTED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["run", "collect"], required=True)
    ap.add_argument("--T", type=int); ap.add_argument("--n-jobs", type=int, default=3)
    a = ap.parse_args()
    run(a.T, a.n_jobs) if a.stage == "run" else collect()
