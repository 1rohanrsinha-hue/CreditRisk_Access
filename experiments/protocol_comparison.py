"""Threshold-selection protocol comparison (Youden's J) on an untouched test set.

D1 = first 100k rows, legacy 5-status label, legacy features (grade-derived credit_score)
D2 = seeded random sample, strict label (Fully Paid vs Charged Off/Default), headline
     features (fico mean replaces the grade-derived credit_score)

Models are trained once on the train split (64%). The threshold is then chosen by:
  A_test   tuned on the TEST set (legacy behaviour; optimistic)
  B_val    tuned on the validation set (16%)
  C_oof    tuned on 5-fold out-of-fold predictions over train+val (80%)
  fixed_0.5 reference, no tuning
and every protocol is evaluated on the same test set.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import Config, RESULTS_DIR, stage               # noqa: E402
from creditrisk.data import load_first_n, label_legacy, load_dev_sample  # noqa: E402
from creditrisk.features import build_frame, selection_stability        # noqa: E402
from creditrisk.models import make_models, oof_proba, MODEL_NAMES                  # noqa: E402
from creditrisk.splits import split_indices                             # noqa: E402
from creditrisk.thresholds import youden_threshold                      # noqa: E402
from creditrisk.evaluate import metrics_at                              # noqa: E402


def get_dataset(name, cfg, n):
    if name == "D1":
        raw = label_legacy(load_first_n(n or 100_000))
        return build_frame(raw, use_interest_rate=True, credit_score="grade_derived")
    raw = load_dev_sample(n or cfg.sample_size, cfg.seed, "strict")
    return build_frame(raw, use_interest_rate=cfg.use_interest_rate, credit_score="fico")


def run(name, cfg, n, oof_cap, only):
    with stage(f"{name}: load+features"):
        X, y, meta = get_dataset(name, cfg, n)
    tr, va, te = split_indices(y, cfg.seed, cfg.test_size, cfg.val_size)
    tv = np.concatenate([tr, va])
    print(f"{name}: n={len(y)} prevalence={y.mean():.3f} train/val/test={len(tr)}/{len(va)}/{len(te)} "
          f"features={list(X.columns)}", flush=True)
    models = make_models(y.iloc[tr], cfg.seed, cfg.n_jobs, cfg.select_k, columns=list(X.columns))
    rows = []
    Xtr, ytr, Xva, yva, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], X.iloc[te], y.iloc[te]
    # optional OOF subsample (stratified) to bound runtime
    tv_idx = tv
    if oof_cap and len(tv) > oof_cap:
        rng = np.random.RandomState(cfg.seed)
        tv_idx = np.sort(rng.choice(tv, oof_cap, replace=False))
        print(f"{name}: OOF stage uses {oof_cap} of {len(tv)} train+val rows", flush=True)
    Xtv, ytv = X.iloc[tv_idx], y.iloc[tv_idx]
    skf = StratifiedKFold(cfg.oof_folds, shuffle=True, random_state=cfg.seed)
    for mname in MODEL_NAMES:
        if only and mname not in only:
            continue
        pipe = models[mname]
        with stage(f"{name}: {mname} fit"):
            pipe.fit(Xtr, ytr)
        p_va, p_te = pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]
        with stage(f"{name}: {mname} OOF"):
            oof, sel = oof_proba(mname, Xtv, ytv, skf, cfg.seed, cfg.n_jobs, cfg.select_k)
        stab = selection_stability(sel)
        thr = {"A_test": youden_threshold(yte, p_te),
               "B_val": youden_threshold(yva, p_va),
               "C_oof": youden_threshold(ytv, oof),
               "fixed_0.5": 0.5}
        for proto, t in thr.items():
            r = metrics_at(yte, p_te, t)
            r.update(dataset=name, model=mname, protocol=proto, n_train=len(tr),
                     n_val=len(va), n_test=len(te), prevalence=float(y.mean()),
                     oof_n=len(tv_idx), seed=cfg.seed, sel_stability=stab)
            rows.append(r)
        # incremental save (partial results survive crashes)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"protocol_{name}_partial.csv", index=False)
    df = pd.DataFrame(rows)
    # gaps versus protocol A (same model)
    A = df[df.protocol == "A_test"].set_index("model")
    for m in ['Recall', 'Precision', 'F1', 'MCC', 'Threshold']:
        df[f"d_{m}_vs_A"] = df[m] - df["model"].map(A[m])
    cols = ['dataset', 'model', 'protocol', 'Threshold', 'Recall', 'Precision', 'F1', 'Accuracy',
            'ROC_AUC', 'PR_AUC', 'MCC', 'Brier'] + [c for c in df.columns if c.startswith("d_")] + \
           ['n_train', 'n_val', 'n_test', 'prevalence', 'oof_n', 'seed', 'sel_stability']
    return df[cols]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["D1", "D2"], required=True)
    ap.add_argument("--sample-size", type=int, default=0,
                    help="D1: first-N rows (default 100000); D2: random sample size (default 200000)")
    ap.add_argument("--oof-cap", type=int, default=0, help="cap train+val rows used for the OOF stage")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = Config(seed=a.seed, n_jobs=a.n_jobs)
    t0 = time.time()
    df = run(a.dataset, cfg, a.sample_size, a.oof_cap, a.models)
    out = RESULTS_DIR / f"protocol_comparison_{a.dataset}.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s -> {out}")
