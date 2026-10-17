"""Loaders and leak-free preprocessing for the external credit datasets (Phase 7).

Each loader returns (X, y, spec): X raw typed features (missing values left for the
in-fold pipeline), y in {0,1} with 1 = bad/default, spec = ColumnSpec with the column
types. Encoding is decided from the data cards (docs/data_cards/):
  * nominal   -> one-hot (LR) / integer codes (RF, GB, XGBoost) / pandas category
                 (LightGBM) / native strings (CatBoost); categories learned on train only
  * ordinal   -> numeric codes in the documented order (treated as numeric)
  * numeric   -> median-imputed, scaled for LR
No target-derived features. Imputation, encoding and scaling live inside the Pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from .config import ROOT
from .features import FrameEncoder
from .models import make_models

EXT_DIR = ROOT / "external_data"
EXT_MODELS = ['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting']


@dataclass
class ColumnSpec:
    nominal: list = field(default_factory=list)
    ordinal: list = field(default_factory=list)
    numeric: list = field(default_factory=list)


GERMAN_COLS = ['checking_status', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
               'savings', 'employment_since', 'installment_rate', 'personal_status_sex',
               'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
               'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker']
# A7 employment duration is the only attribute with a clean monotone order (A71 unemployed < ... < A75 >= 7y).
GERMAN_EMP_ORDER = {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}


def load_german(path=None):
    """UCI Statlog German Credit (german.data). Target coded 1 = good, 2 = bad -> y = 1 if bad."""
    d = pd.read_csv(path or EXT_DIR / "german.data", sep=' ', header=None, names=GERMAN_COLS + ['target'])
    assert set(d.target.unique()) == {1, 2}
    y = (d.pop('target') == 2).astype(int)
    d['employment_since'] = d['employment_since'].map(GERMAN_EMP_ORDER).astype('float64')
    num = ['duration_months', 'credit_amount', 'installment_rate', 'residence_since', 'age',
           'existing_credits', 'people_liable']
    for c in num:
        d[c] = d[c].astype('float64')
    nominal = [c for c in GERMAN_COLS if c not in num + ['employment_since']]
    for c in nominal:
        d[c] = d[c].astype(str)
    return d, y, ColumnSpec(nominal=nominal, ordinal=['employment_since'], numeric=num)


TAIWAN_PAY = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']


def load_taiwan(path=None):
    """UCI Default of Credit Card Clients (xls, header on row 2). y = 'default payment next month'.
    Features cover April-September 2005; the label is the October payment: no post-outcome columns.
    ID is dropped. PAY_* (-2..9: repayment status, documented order) are treated as ordinal/numeric;
    EDUCATION and MARRIAGE contain undocumented codes (0, 5, 6 / 0) so they stay nominal with codes as given."""
    d = pd.read_excel(path or EXT_DIR / "taiwan.xls", header=1)
    y = d.pop('default payment next month').astype(int)
    d = d.drop(columns=['ID'])
    nominal = ['SEX', 'EDUCATION', 'MARRIAGE']
    for c in nominal:
        d[c] = d[c].astype(str)
    num = [c for c in d.columns if c not in nominal + TAIWAN_PAY]
    for c in num + TAIWAN_PAY:
        d[c] = d[c].astype('float64')
    return d, y, ColumnSpec(nominal=nominal, ordinal=TAIWAN_PAY, numeric=num)


LOADERS = {'GERMAN': load_german, 'TAIWAN': load_taiwan}
# test fraction; Nadeau-Bengio ratio n_test / n_train follows from it
TEST_SIZE = {'GERMAN': 0.30, 'TAIWAN': 0.20}
N_REPEATS = {'GERMAN': 30, 'TAIWAN': 10}


class ExtEncoder(FrameEncoder):
    """FrameEncoder with a per-dataset nominal list (the base class is tied to Lending Club columns).
    Fit on training rows only; everything else (transform) is inherited."""
    def __init__(self, mode="ordinal", nominal=()):
        super().__init__(mode=mode, select_k=None)
        self.nominal = nominal

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.nominal_ = [c for c in self.nominal if c in X.columns]
        self.numeric_ = [c for c in X.columns if c not in self.nominal_]
        self.medians_ = X[self.numeric_].median().fillna(0.0)
        self.modes_ = {c: (X[c].mode()[0] if X[c].notna().any() else 'Unknown') for c in self.nominal_}
        self.cats_ = {c: sorted(X[c].fillna(self.modes_[c]).astype(str).unique()) for c in self.nominal_}
        self.selected_ = list(X.columns)
        Xn = X[self.numeric_].fillna(self.medians_[self.numeric_])
        self.mean_, self.std_ = Xn.mean(), Xn.std(ddof=0).replace(0, 1.0)
        return self


def ext_pipeline(name, spec, y_train, weighting, seed, n_jobs=2):
    """Same estimators and default hyperparameters as the main study (creditrisk.models),
    wrapped with the dataset's encoder. Class weights come from y_train only."""
    pipe = make_models(y_train, seed, n_jobs, None, columns=None, weighting=weighting)[name]
    pipe.steps[0] = ('enc', ExtEncoder(mode=pipe.named_steps['enc'].mode, nominal=tuple(spec.nominal)))
    if name == 'CatBoost':
        pipe.named_steps['clf'].set_params(cat_features=list(spec.nominal))
    return pipe


def fit_ext(name, X, y, spec, weighting, seed, n_jobs=2):
    pipe = ext_pipeline(name, spec, y, weighting, seed, n_jobs)
    if weighting == 'balanced' and name == 'Gradient Boosting':
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight('balanced', y))
    else:
        pipe.fit(X, y)
    return pipe
