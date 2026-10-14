"""Exact replica of the app.py pipeline behind Table I.

first 100k CSV rows -> legacy 5-status label -> legacy cleaning
(legacy_fill_bug=True reproduces the pandas-3 inplace-fillna no-op)
-> LabelEncoder -> app.perform_feature_selection -> mean fill -> StandardScaler
-> stratified 80/20 split (seed 42) -> 10 app models -> Youden threshold on TEST.

app.py is imported (not modified) so models / feature selection are the real code.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, stage                       # noqa: E402
from creditrisk.data import load_first_n, legacy_clean                  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler          # noqa: E402
from sklearn.model_selection import train_test_split                    # noqa: E402
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)                  # noqa: E402

PAPER = {  # values quoted from the paper's Table I (partial)
    'Logistic Regression': dict(Accuracy=0.592, Precision=0.269, Recall=0.718, F1=0.391, AUC=0.696, Threshold=0.452),
    'CatBoost': dict(AUC=0.703, Threshold=0.519),
    'Random Forest': dict(Recall=0.668, Threshold=0.180),
}


def run(variant: str, nrows: int, seed: int = 42):
    import app as legacy_app                     # real models + feature selection
    fill_bug = variant == "legacy_fill_bug"
    with stage("load first-n rows from cache"):
        raw = load_first_n(nrows)
    with stage("legacy clean"):
        df = legacy_clean(raw, legacy_fill_bug=fill_bug)
    y = df['default']
    X = df.drop(columns=['default']).copy()
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    with stage("legacy feature selection (no-op reorder)"):
        feats = legacy_app.perform_feature_selection(X, y, n_features=16)
    X = X[feats].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=feats)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    rows = []
    for name, model in legacy_app.get_all_models().items():
        with stage(f"fit {name}"):
            model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        fpr, tpr, thr = roc_curve(yte, p)
        t = float(thr[np.argmax(tpr - fpr)])
        pred = (p >= t).astype(int)
        rows.append(dict(variant=variant, model=name,
                         Accuracy=accuracy_score(yte, pred),
                         Precision=precision_score(yte, pred, zero_division=0),
                         Recall=recall_score(yte, pred), F1=f1_score(yte, pred),
                         AUC=roc_auc_score(yte, p), Threshold=t,
                         n_train=len(Xtr), n_test=len(Xte), feat_order=",".join(feats)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["legacy_fill_bug", "proper_impute"], required=True)
    ap.add_argument("--sample-size", type=int, default=100_000, help="first-N rows (legacy=100000)")
    a = ap.parse_args()
    t0 = time.time()
    res = run(a.variant, a.sample_size)
    out = RESULTS_DIR / f"baseline_table1_{a.variant}.csv"
    res.drop(columns=['feat_order']).to_csv(out, index=False)
    print(res.drop(columns=['feat_order', 'n_train', 'n_test']).round(4).to_string(index=False))
    for m, ref in PAPER.items():
        r = res[res.model == m].iloc[0]
        print(f"vs paper {m}: " + ", ".join(f"{k} {r[k]:.3f} (paper {v:.3f}, d={r[k]-v:+.3f})" for k, v in ref.items()))
    print(f"total {time.time()-t0:.0f}s -> {out}")
