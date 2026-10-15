"""Cache out-of-time raw scores (calibration year T-1 and test year T) per
(fold, model, seed), so stats / capacity / drift studies share them.
Resumable: one npz per unit in .cache/oot_scores/."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import CACHE_DIR
from creditrisk.data import full_frame, label_strict, issue_year, term_months, parse_month, profit_p1, profit_p2
from creditrisk.features import build_frame
from creditrisk.models import MODEL_NAMES, fit_model
from oot_study import fold_indices, usable_columns, FOLDS

OUT = CACHE_DIR / "oot_scores"
MAIN5 = ['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']


def oot_table():
    """Strict-labelled rows, issue year <= 2015 (same filter/order as oot_study.make_data)."""
    d = label_strict(full_frame())
    yr = issue_year(d).values
    d = d[yr <= 2015]
    X, y, _ = build_frame(d, True, 'fico')
    out = dict(d=d, X=X, y=y.values, yr=issue_year(d).values, tm=term_months(d).values,
               month=parse_month(d.issue_d).dt.month.values,
               funded=d.funded_amnt.values.astype(float),
               P={'P1': profit_p1(d).values, 'P2_3%': profit_p2(d, 0.03).values})
    return out


def path(T, model, seed):
    return OUT / f"{T}_{model.replace(' ', '_')}_s{seed}.npz"


def load(T, model, seed):
    z = np.load(path(T, model, seed))
    return {k: z[k] for k in z.files}


def main(n_jobs=4):
    OUT.mkdir(parents=True, exist_ok=True)
    t = oot_table(); X, y, yr, tm = t['X'], t['y'], t['yr'], t['tm']
    jobs = [(m, 42) for m in MODEL_NAMES] + [(m, s) for s in (43, 44) for m in MAIN5]
    for T in FOLDS:
        tr, ca, te = fold_indices(T, yr, tm)
        cols, _ = usable_columns(X, tr); Xc = X[cols]
        for model, seed in jobs:
            p = path(T, model, seed)
            if p.exists():
                continue
            t0 = time.time()
            m = fit_model(model, Xc.iloc[tr], pd.Series(y[tr]), 'none', seed, n_jobs)
            np.savez_compressed(str(p) + ".tmp.npz", s_ca=m.predict_proba(Xc.iloc[ca])[:, 1].astype(np.float32),
                                s_te=m.predict_proba(Xc.iloc[te])[:, 1].astype(np.float32),
                                idx_tr=tr.astype(np.int32), idx_ca=ca.astype(np.int32), idx_te=te.astype(np.int32))
            import os; os.replace(str(p) + ".tmp.npz", p)
            print(f"[oot-scores] T={T} {model} s{seed} {time.time()-t0:.0f}s", flush=True)
    print("done", len(list(OUT.glob('*.npz'))), "files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-jobs", type=int, default=4)
    main(ap.parse_args().n_jobs)
