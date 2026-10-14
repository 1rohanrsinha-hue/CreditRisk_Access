"""Leak-free feature construction: raw frame -> typed X; encoders live in sklearn
transformers that are fit on the training fold only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif

from .config import META_COLS
from .data import _RENAME, grade_derived_credit_score

EMP_ORDER = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
             '6 years', '7 years', '8 years', '9 years', '10+ years']
EMP_MAP = {k: i for i, k in enumerate(EMP_ORDER)}   # 0..10; 'n/a'/NaN -> missing
NOMINAL = ['home_ownership', 'loan_purpose']
NUMERIC_BASE = ['loan_amount', 'annual_income', 'debt_to_income', 'delinquencies',
                'num_credit_lines', 'revolving_utilization', 'total_credit_lines',
                'public_records', 'credit_inquiries', 'mortgage_accounts', 'collections']


def build_frame(df: pd.DataFrame, use_interest_rate: bool = True,
                credit_score: str = "fico"):
    """Return (X, y, meta). X holds raw (un-imputed) application-time features.
    credit_score: 'fico' (headline, fico_range mean), 'grade_derived' (legacy,
    leaky), 'none'. Missing values are left for the in-fold pipeline."""
    X = pd.DataFrame(index=df.index)
    for old, new in _RENAME.items():
        if new in NUMERIC_BASE or new in NOMINAL:
            X[new] = df[old]
    X['annual_income'] = df['annual_inc']
    X['employment_length'] = df['emp_length'].map(EMP_MAP).astype('float64')  # NaN = missing
    X['emp_length_missing'] = X['employment_length'].isna().astype(float)
    ir = df['int_rate'].astype(str).str.replace('%', '').astype(float)
    if use_interest_rate:
        X['interest_rate'] = ir
    if credit_score == "fico":
        X['fico_mean'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    elif credit_score == "grade_derived":
        X['credit_score'] = grade_derived_credit_score(df['grade'], ir)
    X = X[[c for c in X.columns]]
    y = df['default'].astype(int)
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    return X, y, meta


class FrameEncoder(BaseEstimator, TransformerMixin):
    """Imputation + encoding + (optional) scaling, fit on train only.

    mode: 'onehot_scaled' (LR/MLP) | 'ordinal' (sklearn trees/boosters)
          | 'category' (LightGBM pandas-category) | 'catboost' (string cats).
    select_k: keep top-k raw features by in-fold ANOVA F (None = keep all).
    """
    def __init__(self, mode="ordinal", select_k=None):
        self.mode = mode
        self.select_k = select_k

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.nominal_ = [c for c in NOMINAL if c in X.columns]
        self.numeric_ = [c for c in X.columns if c not in self.nominal_]
        self.medians_ = X[self.numeric_].median()
        self.modes_ = {c: (X[c].mode()[0] if X[c].notna().any() else 'Unknown')
                       for c in self.nominal_}
        self.cats_ = {c: sorted(X[c].fillna(self.modes_[c]).astype(str).unique())
                      for c in self.nominal_}
        keep = list(X.columns)
        if self.select_k is not None and y is not None and self.select_k < len(keep):
            Z = self._ordinal(X, keep_all=True)
            f, _ = f_classif(Z.fillna(0), np.asarray(y))
            f = np.nan_to_num(f)
            keep = list(Z.columns[np.argsort(-f)[:self.select_k]])
        self.selected_ = keep
        self.nominal_ = [c for c in self.nominal_ if c in keep]
        self.numeric_ = [c for c in self.numeric_ if c in keep]
        Xn = X[self.numeric_].fillna(self.medians_[self.numeric_])
        self.mean_ = Xn.mean()
        self.std_ = Xn.std(ddof=0).replace(0, 1.0)
        return self

    def _ordinal(self, X, keep_all=False):
        Z = X.copy()
        for c in [c for c in NOMINAL if c in Z.columns]:
            cats = self.cats_[c]
            Z[c] = (Z[c].fillna(self.modes_[c]).astype(str)
                    .map({v: i for i, v in enumerate(cats)}).fillna(-1))
        return Z

    def transform(self, X):
        X = pd.DataFrame(X)
        num = X[self.numeric_].fillna(self.medians_[self.numeric_]).astype('float64')
        if self.mode == "onehot_scaled":
            parts = [(num - self.mean_) / self.std_]
            for c in self.nominal_:
                v = X[c].fillna(self.modes_[c]).astype(str)
                parts.append(pd.DataFrame({f"{c}={k}": (v == k).astype(float)
                                           for k in self.cats_[c]}, index=X.index))
            return pd.concat(parts, axis=1)
        out = num.copy()
        for c in self.nominal_:
            v = X[c].fillna(self.modes_[c]).astype(str)
            if self.mode == "ordinal":
                out[c] = v.map({k: i for i, k in enumerate(self.cats_[c])}).fillna(-1).astype(float)
            elif self.mode == "category":
                out[c] = pd.Categorical(v, categories=self.cats_[c])
            elif self.mode == "catboost":
                out[c] = v.where(v.isin(self.cats_[c]), self.modes_[c])
            else:
                raise ValueError(self.mode)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.transform(pd.DataFrame(columns=self.selected_)).columns) \
            if False else np.array(self.selected_)

    def categorical_columns(self):
        return list(self.nominal_)


def selection_stability(fold_sets):
    """Mean pairwise Jaccard of selected raw-feature sets across folds."""
    from itertools import combinations
    sets = [set(s) for s in fold_sets]
    if len(sets) < 2:
        return 1.0
    return float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)]))
