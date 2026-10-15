"""The 10 models, each wrapped in a leak-free Pipeline (encoder + estimator).
Class weights are derived from the training labels (not hardcoded). Weighting
follows the legacy scheme (GB/AdaBoost/MLP unweighted) so protocol comparisons
isolate the threshold effect; the weighting ablation is Phase 2."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .features import FrameEncoder, NOMINAL

MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest',
               'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost',
               'AdaBoost', 'Extra Trees', 'Neural Network']


WEIGHTED_BY_SAMPLE_WEIGHT = ('Gradient Boosting', 'AdaBoost', 'Neural Network')
# legacy scheme: these 7 are balanced, GB/AdaBoost/MLP unweighted
_LEGACY_BALANCED = True


def _cw(weighting):
    # 'legacy' behaves like 'balanced' for models with a class_weight param
    return 'balanced' if weighting in ('legacy', 'balanced') else None


def _spw(weighting, spw):
    return spw if weighting in ('legacy', 'balanced') else 1.0


def make_models(y_train, seed=42, n_jobs=4, select_k=None, columns=None,
                weighting='legacy', leaf=None):
    """Return {name: Pipeline}. y_train used only for imbalance ratio."""
    y_train = np.asarray(y_train)
    pos = y_train.sum()
    spw = float((len(y_train) - pos) / max(pos, 1))
    cat_cols = [c for c in NOMINAL if columns is None or c in columns]
    # weighting: 'legacy' (paper scheme), 'balanced', 'none'
    cw = {'legacy': None, 'balanced': 'balanced', 'none': None}[weighting]
    bal = weighting == 'balanced'
    leaf_kw = {} if leaf is None else {'min_samples_leaf': int(leaf)}
    enc = lambda mode: FrameEncoder(mode=mode, select_k=select_k)  # noqa: E731

    spec = {
        'Logistic Regression': ('onehot_scaled', LogisticRegression(
            max_iter=1000, random_state=seed, class_weight=_cw(weighting))),
        'Decision Tree': ('ordinal', DecisionTreeClassifier(
            max_depth=10, random_state=seed, class_weight=_cw(weighting))),
        'Random Forest': ('ordinal', RandomForestClassifier(
            n_estimators=100, random_state=seed, class_weight=_cw(weighting), n_jobs=n_jobs, **leaf_kw)),
        'Gradient Boosting': ('ordinal', GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed)),
        'XGBoost': ('ordinal', XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed,
            eval_metric='logloss', scale_pos_weight=_spw(weighting, spw), n_jobs=n_jobs)),
        'LightGBM': ('category', LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed,
            verbose=-1, class_weight=_cw(weighting), n_jobs=n_jobs)),
        'CatBoost': ('catboost', CatBoostClassifier(
            n_estimators=100, learning_rate=0.1, depth=5, random_seed=seed,
            verbose=False, auto_class_weights=('Balanced' if _cw(weighting) else None), thread_count=n_jobs,
            allow_writing_files=False, cat_features=cat_cols)),
        'AdaBoost': ('ordinal', AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=seed)),
        'Extra Trees': ('ordinal', ExtraTreesClassifier(
            n_estimators=100, random_state=seed, class_weight=_cw(weighting), n_jobs=n_jobs, **leaf_kw)),
        'Neural Network': ('onehot_scaled', MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed,
            early_stopping=True)),
    }
    return {n: Pipeline([('enc', enc(m)), ('clf', c)]) for n, (m, c) in spec.items()}


def oof_proba(name, X, y, skf, seed=42, n_jobs=4, select_k=None):
    """Out-of-fold P(default) with a fresh pipeline per fold (CatBoost with
    cat_features cannot be sklearn-cloned). Class weights are recomputed from
    each fold's training labels. Returns (oof, per-fold selected raw features)."""
    oof = np.zeros(len(y))
    selected = []
    for tr, te in skf.split(X, y):
        pipe = make_models(y.iloc[tr], seed, n_jobs, select_k, columns=list(X.columns))[name]
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        selected.append(pipe.named_steps['enc'].selected_)
    return oof, selected


def fit_model(name, X, y, weighting='balanced', seed=42, n_jobs=4, leaf=None):
    """Fresh pipeline fit on (X, y). Balanced weighting for GB/AdaBoost/MLP via
    sample_weight (sklearn 1.8 MLPClassifier.fit supports sample_weight)."""
    from sklearn.utils.class_weight import compute_sample_weight
    pipe = make_models(y, seed, n_jobs, None, columns=list(X.columns),
                       weighting=weighting, leaf=leaf)[name]
    if weighting == 'balanced' and name in WEIGHTED_BY_SAMPLE_WEIGHT:
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight('balanced', y))
    else:
        pipe.fit(X, y)
    return pipe
