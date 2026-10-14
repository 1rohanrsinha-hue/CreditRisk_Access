"""Data loading (parquet cache), labels and the legacy cleaning replica."""
import numpy as np
import pandas as pd

from .config import (CACHE_COLS, CSV_PATH, PARQUET_PATH, CACHE_DIR, stage,
                     LEGACY_DEFAULT_STATUSES, STRICT_DEFAULT, STRICT_GOOD)


def build_cache(force: bool = False):
    """Read the 1.6GB CSV once (usecols) and persist it as parquet."""
    if PARQUET_PATH.exists() and not force:
        return PARQUET_PATH
    CACHE_DIR.mkdir(exist_ok=True)
    with stage("read full CSV"):
        df = pd.read_csv(CSV_PATH, usecols=CACHE_COLS, low_memory=False)
    with stage("write parquet"):
        # row order of the CSV is preserved (needed for the legacy first-100k)
        df.to_parquet(PARQUET_PATH, index=False)
    return PARQUET_PATH


def load_full() -> pd.DataFrame:
    build_cache()
    return pd.read_parquet(PARQUET_PATH)


def load_first_n(n: int = 100_000) -> pd.DataFrame:
    """First n CSV rows (what legacy app.py reads with nrows=n)."""
    return load_full().iloc[:n].copy()


def label_legacy(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['loan_status'].notna()].copy()
    df['default'] = df['loan_status'].isin(LEGACY_DEFAULT_STATUSES).astype(int)
    return df


def label_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Fully Paid (0) vs Charged Off/Default (1). Everything else is dropped."""
    df = df[df['loan_status'].isin(STRICT_GOOD + STRICT_DEFAULT)].copy()
    df['default'] = df['loan_status'].isin(STRICT_DEFAULT).astype(int)
    return df


def load_dev_sample(n: int = 200_000, seed: int = 42, label: str = "strict") -> pd.DataFrame:
    """Seeded random sample of n *labelled* rows from the full file."""
    df = load_full()
    df = label_strict(df) if label == "strict" else label_legacy(df)
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.sort_index()


# --------------------------------------------------------------------------
# Replica of app.py:clean_lending_club_data (legacy features) with the
# pandas-3 inplace-fillna no-op reproduced on purpose via `legacy_fill_bug`.
# --------------------------------------------------------------------------
_NUM = ['loan_amount', 'interest_rate', 'annual_income', 'debt_to_income',
        'delinquencies', 'num_credit_lines', 'revolving_utilization',
        'total_credit_lines', 'credit_score', 'public_records',
        'credit_inquiries', 'mortgage_accounts', 'collections']
_CAT = ['employment_length', 'home_ownership', 'loan_purpose']
_FINAL = ['loan_amount', 'interest_rate', 'annual_income', 'employment_length',
          'home_ownership', 'loan_purpose', 'debt_to_income', 'delinquencies',
          'num_credit_lines', 'revolving_utilization', 'total_credit_lines',
          'credit_score', 'public_records', 'credit_inquiries',
          'mortgage_accounts', 'collections', 'default']
_RENAME = {'loan_amnt': 'loan_amount', 'annual_inc': 'annual_income',
           'dti': 'debt_to_income', 'delinq_2yrs': 'delinquencies',
           'open_acc': 'num_credit_lines', 'total_acc': 'total_credit_lines',
           'purpose': 'loan_purpose', 'home_ownership': 'home_ownership', 'pub_rec': 'public_records',
           'inq_last_6mths': 'credit_inquiries', 'mort_acc': 'mortgage_accounts',
           'collections_12_mths_ex_med': 'collections'}
GRADE_SCORES = {'A': 750, 'B': 700, 'C': 650, 'D': 600, 'E': 550, 'F': 500, 'G': 450}


def grade_derived_credit_score(grade: pd.Series, int_rate: pd.Series) -> pd.Series:
    cs = grade.map(GRADE_SCORES) - (int_rate - 12) * 5
    return cs.clip(300, 850)


def legacy_clean(df: pd.DataFrame, legacy_fill_bug: bool = True) -> pd.DataFrame:
    """Legacy cleaning. legacy_fill_bug=True reproduces what pandas 3.0.2 does
    to the original code (median/mode fills silently do nothing, everything is
    zero-filled). False = the fills the author intended (median / mode)."""
    d = label_legacy(df)
    out = pd.DataFrame(index=d.index)
    out['interest_rate'] = d['int_rate'].astype(str).str.replace('%', '').astype(float)
    out['revolving_utilization'] = d['revol_util'].astype(str).str.replace('%', '').astype(float)
    out['employment_length'] = d['emp_length'].map(
        {'< 1 year': '< 1 year', '1 year': '1 year', '2 years': '2 years',
         '3 years': '3 years', '4 years': '4 years', '5 years': '5 years',
         '6 years': '6 years', '7 years': '7 years', '8 years': '8 years',
         '9 years': '9 years', '10+ years': '10+ years', 'n/a': '< 1 year'})
    out['credit_score'] = grade_derived_credit_score(d['grade'], out['interest_rate'])
    for old, new in _RENAME.items():
        out[new] = d[old]
    out['default'] = d['default']
    out = out[_FINAL].copy()
    if not legacy_fill_bug:
        for c in _NUM:
            if out[c].isnull().any():
                out[c] = out[c].fillna(out[c].median())
        for c in _CAT:
            if out[c].isnull().any():
                m = out[c].mode()
                out[c] = out[c].fillna(m[0] if not m.empty else 'Unknown')
    out = out.fillna(0)          # legacy fallback; no-op when properly imputed
    return out
