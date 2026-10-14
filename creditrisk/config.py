"""Central configuration, seeds and stage timing for the journal extension."""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "accepted_2007_to_2018Q4.csv"
CACHE_DIR = ROOT / ".cache"
PARQUET_PATH = CACHE_DIR / "lc_full.parquet"
RESULTS_DIR = ROOT / "experiments" / "results"

# Columns the legacy app.py loads (19) + extras needed for the extension.
LEGACY_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'emp_length', 'home_ownership', 'annual_inc', 'loan_status',
    'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
    'pub_rec', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mort_acc',
]
EXTRA_COLS = ['issue_d', 'fico_range_low', 'fico_range_high',
              'funded_amnt', 'total_pymnt', 'recoveries']
CACHE_COLS = LEGACY_COLS + EXTRA_COLS

LEGACY_DEFAULT_STATUSES = [
    'Charged Off', 'Default', 'Late (31-120 days)',
    'In Grace Period', 'Does not meet the credit policy. Status:Charged Off',
]
STRICT_DEFAULT = ['Charged Off', 'Default']
STRICT_GOOD = ['Fully Paid']

# Columns kept aligned with rows but NEVER used as model inputs.
META_COLS = ['issue_d', 'funded_amnt', 'total_pymnt', 'recoveries', 'loan_status']


@dataclass
class Config:
    seed: int = 42
    sample_size: int = 200_000          # dev sample (random, from full file)
    test_size: float = 0.20             # of all rows
    val_size: float = 0.20              # of train+val  -> 64/16/20 overall
    oof_folds: int = 5
    n_jobs: int = 4
    legacy_fill_bug: bool = False       # only the baseline reproduction sets True
    label: str = "strict"               # "strict" | "legacy"
    # leakage switches
    use_interest_rate: bool = True
    credit_score: str = "fico"          # "fico" | "grade_derived" | "none"
    select_k: int | None = None         # None = keep all raw features


@contextmanager
def stage(name: str):
    t0 = time.time()
    print(f"[stage] {name} ...", flush=True)
    try:
        yield
    finally:
        print(f"[stage] {name} done in {time.time() - t0:.1f}s", flush=True)
