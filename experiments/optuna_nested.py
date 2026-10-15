"""Phase 3D / task F: nested CV with Optuna (H16). Classifiers only; tuning
objective = inner 3-fold log loss on a seeded 100k subsample of the outer-training
part; 25 trials per (model, outer fold); best params refit on the full outer-training
part and scored on the outer-test fold; defaults scored on the SAME outer folds.
Usage:  --model "LightGBM" [--trials 25] [--folds 0 1 2 3 4] [--smoke]
        --stage collect   (aggregate all checkpoints)"""
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import optuna
from scipy import stats as sst
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from creditrisk.config import CACHE_DIR, RESULTS_DIR
from creditrisk.calibration import Isotonic, ece
from creditrisk.models import make_models
from creditrisk.repeats import get_dataset
from creditrisk.thresholds import youden_threshold

optuna.logging.set_verbosity(optuna.logging.WARNING)
ORDER = ['Logistic Regression', 'LightGBM', 'XGBoost', 'Random Forest', 'CatBoost']
SEED = 42


def space(model, t):
    if model == 'Logistic Regression':
        return dict(C=t.suggest_float('C', 1e-3, 1e2, log=True), penalty=t.suggest_categorical('penalty', ['l1', 'l2']),
                    solver='liblinear')
    if model == 'Random Forest':
        return dict(n_estimators=t.suggest_int('n_estimators', 100, 250, step=50), max_depth=t.suggest_int('max_depth', 4, 24),
                    min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 60, log=True),
                    max_features=t.suggest_float('max_features', 0.3, 1.0))
    if model == 'LightGBM':
        return dict(n_estimators=t.suggest_int('n_estimators', 50, 500), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    num_leaves=t.suggest_int('num_leaves', 8, 128, log=True), min_child_samples=t.suggest_int('min_child_samples', 20, 300, log=True),
                    subsample=t.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
                    colsample_bytree=t.suggest_float('colsample_bytree', 0.5, 1.0),
                    reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True), max_depth=-1)
    if model == 'XGBoost':
        return dict(n_estimators=t.suggest_int('n_estimators', 50, 400), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    max_depth=t.suggest_int('max_depth', 2, 8), min_child_weight=t.suggest_float('min_child_weight', 1, 50, log=True),
                    subsample=t.suggest_float('subsample', 0.5, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.5, 1.0),
                    reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True))
    if model == 'CatBoost':
        return dict(n_estimators=t.suggest_int('n_estimators', 100, 600), learning_rate=t.suggest_float('learning_rate', 0.02, 0.3, log=True),
                    depth=t.suggest_int('depth', 3, 8), l2_leaf_reg=t.suggest_float('l2_leaf_reg', 1, 20, log=True),
                    bootstrap_type='Bernoulli', subsample=t.suggest_float('subsample', 0.5, 1.0))
    raise ValueError(model)


def build(model, y, params, n_jobs):
    pipe = make_models(y, SEED, n_jobs, None, columns=None, weighting='none')[model]
    if params:
        pipe.named_steps['clf'].set_params(**params)
    return pipe


def fit_eval(model, Xa, ya, Xb, params, n_jobs):
    pipe = build(model, ya, params, n_jobs).fit(Xa, ya)
    return pipe.predict_proba(Xb)[:, 1]


def run_fold(model, fold, X, y, trials, n_jobs, store_dir, smoke=False):
    out = store_dir / "results" / f"{model.replace(' ', '_')}_f{fold}.json"
    if out.exists():
        return json.load(open(out))
    t0 = time.time()
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tr, te = list(outer.split(X, y))[fold]
    rs = np.random.RandomState(SEED + fold)
    sub = np.sort(rs.choice(tr, 100_000, replace=False)) if len(tr) > 100_000 else tr
    Xs, ys = X.iloc[sub], y.iloc[sub]
    inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xs, ys))

    def objective(trial):
        p = space(model, trial)
        ls = []
        for a, b in inner:
            ls.append(log_loss(ys.iloc[b], fit_eval(model, Xs.iloc[a], ys.iloc[a], Xs.iloc[b], p, n_jobs), labels=[0, 1]))
        return float(np.mean(ls))

    db = store_dir / f"{model.replace(' ', '_')}.db"
    study = optuna.create_study(study_name=f"{model}_f{fold}", storage=f"sqlite:///{db}", direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED + fold), load_if_exists=True)
    done = len([t for t in study.trials if t.state.name == 'COMPLETE'])
    t_tune = time.time()
    if done < trials:
        study.optimize(objective, n_trials=trials - done)
    tune_secs = time.time() - t_tune
    best = study.best_trial.params
    full_best = dict(best)
    if model == 'Logistic Regression':
        full_best['solver'] = 'liblinear'
    if model == 'LightGBM':
        full_best.update(subsample_freq=1, max_depth=-1)
    if model == 'CatBoost':
        full_best['bootstrap_type'] = 'Bernoulli'
    ytr, yte = y.iloc[tr], y.iloc[te]
    res = dict(model=model, fold=fold, n_trials=len(study.trials), tune_secs=tune_secs, best_params=full_best,
               best_inner_logloss=study.best_value, pi=float(ytr.mean()))
    # tuned vs default on the outer-test fold; calibrated t* on a 20% hold-out of the outer-training part
    a_idx, h_idx = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=SEED, stratify=ytr)
    for tag, params in (('tuned', full_best), ('default', None)):
        p = fit_eval(model, X.iloc[tr], ytr, X.iloc[te], params, n_jobs)
        p_h = fit_eval(model, X.iloc[tr].iloc[a_idx], ytr.iloc[a_idx], X.iloc[tr].iloc[h_idx], params, n_jobs)
        iso = Isotonic().fit(p_h, ytr.iloc[h_idx].values)
        t_star = youden_threshold(ytr.iloc[h_idx].values, iso.predict(p_h))
        res.update({f'AUC_{tag}': roc_auc_score(yte, p), f'Brier_{tag}': brier_score_loss(yte, p),
                    f'LogLoss_{tag}': log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6)), f'ECE_{tag}': ece(yte.values, p),
                    f't_star_{tag}': float(t_star), f'abs_t_pi_{tag}': float(abs(t_star - ytr.mean()))})
    res['secs_total'] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, 'w'), default=float)
    print(f"[optuna] {model} fold {fold}: tuned AUC {res['AUC_tuned']:.4f} default {res['AUC_default']:.4f} "
          f"({res['secs_total']:.0f}s; tuning {tune_secs:.0f}s)", flush=True)
    return res


def collect():
    rows = []
    for f in sorted((CACHE_DIR / "optuna" / "results").glob("*.json")):
        rows.append(json.load(open(f)))
    R = pd.DataFrame(rows); params = {f"{r['model']}|fold{r['fold']}": r['best_params'] for r in rows}
    json.dump(params, open(RESULTS_DIR / "optuna_params.json", 'w'), indent=1, default=float)
    R.drop(columns='best_params').round(5).to_csv(RESULTS_DIR / "phase3d_optuna.csv", index=False)
    pd.set_option('display.width', 250)
    g = []
    for m, d in R.groupby('model'):
        dA = (d.AUC_tuned - d.AUC_default).values; dB = (d.Brier_tuned - d.Brier_default).values; n = len(d)
        tq = sst.t.ppf(.975, n - 1) if n > 1 else np.nan
        g.append(dict(model=m, folds=n, AUC_default=d.AUC_default.mean(), AUC_tuned=d.AUC_tuned.mean(), dAUC=dA.mean(),
                      dAUC_lo=dA.mean() - tq * dA.std(ddof=1) / np.sqrt(n), dAUC_hi=dA.mean() + tq * dA.std(ddof=1) / np.sqrt(n),
                      dBrier=dB.mean(), abs_t_pi_tuned=d.abs_t_pi_tuned.mean(), abs_t_pi_default=d.abs_t_pi_default.mean(),
                      tune_secs=d.tune_secs.sum()))
    G = pd.DataFrame(g).set_index('model').reindex([m for m in ORDER if m in set(R.model)])
    print(G.round(4).to_string())
    G.round(5).to_csv(RESULTS_DIR / "phase3d_optuna_summary.csv")
    print("\nbest-param stability (CV across folds of numeric params):")
    for m, d in R.groupby('model'):
        P = pd.DataFrame([r for r in d.best_params]).select_dtypes('number')
        print(f"  {m}: " + ", ".join(f"{c}: {P[c].mean():.3g} (cv {abs(P[c].std()/P[c].mean()):.2f})" for c in P.columns if P[c].mean() != 0))
    if len(G) == 5 and (G.folds == 5).all():
        spread_t = G.AUC_tuned.max() - G.AUC_tuned.min(); spread_d = G.AUC_default.max() - G.AUC_default.min()
        a = all(G.loc[m, 'dAUC'] < .005 for m in ('LightGBM', 'XGBoost', 'CatBoost'))
        b = G.loc['Random Forest', 'dAUC'] > max(G.loc[m, 'dAUC'] for m in ('LightGBM', 'XGBoost', 'CatBoost'))
        c = spread_t <= spread_d
        print(f"\nH16 (a) boosters dAUC<.005: {a}; (b) RF gains most: {b} (RF {G.loc['Random Forest','dAUC']:.4f}); "
              f"(c) spread tuned {spread_t:.4f} <= default {spread_d:.4f}: {c}")
        print("H16 verdict:", "SUPPORTED" if (a and b and c) else "NOT SUPPORTED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None); ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", default="run")
    a = ap.parse_args()
    if a.stage == "collect":
        collect(); return
    X, y, _ = get_dataset('D3')
    store = CACHE_DIR / ("optuna_smoke" if a.smoke else "optuna")
    store.mkdir(parents=True, exist_ok=True)
    for f in a.folds:
        run_fold(a.model, f, X, y, a.trials, a.n_jobs, store, a.smoke)


if __name__ == "__main__":
    main()
