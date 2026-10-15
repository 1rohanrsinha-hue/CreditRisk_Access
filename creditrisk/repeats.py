"""Resumable repeated-split scoring.

For each (dataset, repeat, model, weighting[, leaf]) we run the calibration
protocol's expensive part ONCE and checkpoint raw scores to .cache/scores/:
  1. stratified 64/16/20 split (split seed = base_seed + repeat)
  2. 5-fold OOF raw scores over train+val, fresh pipeline per fold
     (class weights recomputed from each fold's training labels)
  3. refit on all of train+val, score the untouched test set
Everything downstream (calibration, thresholds, metrics, profit) is cheap and
lives in creditrisk/study.py.
"""
import os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import CACHE_DIR, SCORES_DIR, stage
from .data import (load_first_n, load_dev_sample, label_legacy, realized_profit)
from .features import build_frame
from .models import MODEL_NAMES, fit_model
from .splits import split_indices

BASE_SEED = 42
DEV_N = 200_000


def d3_population():
    """Mature vintages: strict label; issue years 2010-2013 (any term) plus
    2014 and 2015 36-month loans. Excludes pre-2010 (policy-exception regime),
    60-month 2014+ and all vintages >= 2016 (censored)."""
    from .data import full_frame, label_strict, issue_year, term_months
    d = label_strict(full_frame())
    yr, tm = issue_year(d), term_months(d)
    m = ((yr >= 2010) & (yr <= 2013)) | (yr.isin([2014, 2015]) & (tm == 36))
    return d[m]


def d3_sample(n: int = DEV_N):
    pop = d3_population()
    return pop.sample(n=min(n, len(pop)), random_state=BASE_SEED).sort_index()


def _parse(name: str):
    """'D3', or an endogenous-feature ablation cell like 'D3:ir0:csgd' (interest rate out,
    grade-derived credit score). cs: fico | gd (grade-derived) | none."""
    base, *opts = name.split(':')
    ir, cs = True, 'fico'
    for o in opts:
        if o == 'ir0': ir = False
        elif o == 'ir1': ir = True
        elif o.startswith('cs'): cs = {'csfico': 'fico', 'csgd': 'grade_derived', 'csnone': 'none'}[o]
        else: raise ValueError(o)
    return base, ir, cs


def get_dataset(name: str):
    """D3: seeded 200k sample of mature vintages (see d3_population).
    D1: first 100k rows, legacy label, legacy features (grade-derived score).
    D2: seeded random 200k labelled rows, strict label, headline features.
    Returns (X, y, profit) ; profit is realized profit (meaningful for D2)."""
    name, ir_flag, cs_flag = _parse(name)
    if name == "D3FULL":                        # full 667k mature population
        raw = d3_population().sort_index()
        X, y, meta = build_frame(raw, ir_flag, cs_flag)
    elif name == "STRICTFULL":                  # full 1.345M strict set (sensitivity, censored 2016+)
        from .data import full_frame, label_strict
        raw = label_strict(full_frame()).sort_index()
        X, y, meta = build_frame(raw, ir_flag, cs_flag)
    elif name == "D1":
        raw = label_legacy(load_first_n(100_000))
        X, y, meta = build_frame(raw, True, "grade_derived")
    elif name == "D3":
        raw = d3_sample()
        X, y, meta = build_frame(raw, ir_flag, cs_flag)
    elif name == "D2":
        raw = load_dev_sample(DEV_N, BASE_SEED, "strict")
        X, y, meta = build_frame(raw, ir_flag, cs_flag)
    else:
        raise ValueError(name)
    profit = realized_profit(raw.loc[X.index])
    return X, y, profit


def split(y, rep):
    tr, va, te = split_indices(y, BASE_SEED + rep, 0.20, 0.20)
    return tr, np.concatenate([tr, va]), te


def _slug(s):
    return s.replace(' ', '_').replace('(', '').replace(')', '')


def score_path(ds, rep, model, weighting, leaf=None):
    d = SCORES_DIR / ds.replace(':', '_')
    d.mkdir(parents=True, exist_ok=True)
    return d / f"r{rep:02d}__{_slug(model)}__{weighting}__leaf{leaf or 0}.npz"


def score_variant(X, y, ds, rep, model, weighting, leaf=None, n_jobs=2):
    path = score_path(ds, rep, model, weighting, leaf)
    if path.exists():
        return path
    t0 = time.time()
    tr, tv, te = split(y, rep)
    Xtv, ytv = X.iloc[tv], y.iloc[tv]
    skf = StratifiedKFold(5, shuffle=True, random_state=BASE_SEED + rep)
    oof = np.zeros(len(tv))
    seed = BASE_SEED + rep
    for a, b in skf.split(Xtv, ytv):
        m = fit_model(model, Xtv.iloc[a], ytv.iloc[a], weighting, seed, n_jobs, leaf)
        oof[b] = m.predict_proba(Xtv.iloc[b])[:, 1]
    m = fit_model(model, Xtv, ytv, weighting, seed, n_jobs, leaf)
    te_s = m.predict_proba(X.iloc[te])[:, 1]
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, oof=oof.astype(np.float32), te=te_s.astype(np.float32),
                        idx_tv=tv.astype(np.int32), idx_te=te.astype(np.int32),
                        secs=np.array([time.time() - t0]))
    os.replace(tmp, path)
    return path


def run_scoring(ds, reps, models=None, weightings=('none', 'balanced'),
                leaves=(None,), n_jobs=2, worker=(0, 1)):
    """Compute (and checkpoint) scores. `worker=(i, n)` processes repeats with
    rep % n == i so several processes can share the same cache."""
    models = models or MODEL_NAMES
    X, y, _ = get_dataset(ds)
    for rep in reps:
        if rep % worker[1] != worker[0]:
            continue
        for w in weightings:
            for model in models:
                for leaf in leaves:
                    if leaf is not None and model not in ('Random Forest', 'Extra Trees'):
                        continue
                    p = score_path(ds, rep, model, w, leaf)
                    if p.exists():
                        continue
                    t0 = time.time()
                    score_variant(X, y, ds, rep, model, w, leaf, n_jobs)
                    print(f"[score] {ds} rep={rep} {model} w={w} leaf={leaf} "
                          f"{time.time()-t0:.0f}s", flush=True)


def load_score(ds, rep, model, weighting, leaf=None):
    z = np.load(score_path(ds, rep, model, weighting, leaf))
    return {k: z[k] for k in z.files}
