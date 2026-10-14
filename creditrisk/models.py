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


def make_models(y_train, seed=42, n_jobs=4, select_k=None, columns=None):
    """Return {name: Pipeline}. y_train used only for imbalance ratio."""
    y_train = np.asarray(y_train)
    pos = y_train.sum()
    spw = float((len(y_train) - pos) / max(pos, 1))
    cat_cols = [c for c in NOMINAL if columns is None or c in columns]
    enc = lambda mode: FrameEncoder(mode=mode, select_k=select_k)  # noqa: E731

    spec = {
        'Logistic Regression': ('onehot_scaled', LogisticRegression(
            max_iter=1000, random_state=seed, class_weight='balanced')),
        'Decision Tree': ('ordinal', DecisionTreeClassifier(
            max_depth=10, random_state=seed, class_weight='balanced')),
        'Random Forest': ('ordinal', RandomForestClassifier(
            n_estimators=100, random_state=seed, class_weight='balanced', n_jobs=n_jobs)),
        'Gradient Boosting': ('ordinal', GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed)),
        'XGBoost': ('ordinal', XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed,
            eval_metric='logloss', scale_pos_weight=spw, n_jobs=n_jobs)),
        'LightGBM': ('category', LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed,
            verbose=-1, class_weight='balanced', n_jobs=n_jobs)),
        'CatBoost': ('catboost', CatBoostClassifier(
            n_estimators=100, learning_rate=0.1, depth=5, random_seed=seed,
            verbose=False, auto_class_weights='Balanced', thread_count=n_jobs,
            allow_writing_files=False, cat_features=cat_cols)),
        'AdaBoost': ('ordinal', AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=seed)),
        'Extra Trees': ('ordinal', ExtraTreesClassifier(
            n_estimators=100, random_state=seed, class_weight='balanced', n_jobs=n_jobs)),
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
