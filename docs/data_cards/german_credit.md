# Data card: Statlog German Credit Data

- **Source name:** UCI Machine Learning Repository, 'Statlog (German Credit Data)' (Hofmann)
- **Source URL (citation):** FILL IN  (direct-download URL used by `scripts/fetch_external.sh`)
- **Licence:** FILL IN (not verified)
- **Access date:** FILL IN
- **File:** `external_data/german.data` (git-ignored); sha256 prefix `b21f3d81db807125`
- **Rows:** 1,000   **Feature columns:** 20   (nominal 12, ordinal 1, numeric 7)
- **Positive class (bad/default) rate:** 0.3000  (300 positives)
- **Label definition (verified in code):** raw column 21 is coded 1 = good (700), 2 = bad (300); here y = 1 if raw == 2 (bad credit risk)
- **Missing values:** 0 cells in total
- **Exact duplicate rows (features + label):** 0
- **Post-outcome leak check:** none found. All 20 attributes are application-time (checking account, duration, amount, history, purpose, savings, employment, ...); no repayment-performance columns. The 'credit history' attribute (A3) describes earlier loans, not this one.
- **Encoding decisions:** A7 (employment duration) is the only attribute with a monotone documented order and is coded 0-4 (ordinal, numeric). Checking status (A1: 'no checking account' breaks the order), savings (A6: 'unknown' breaks the order), credit history, purpose, personal status/sex, other debtors, property, other installment plans, housing, job, telephone and foreign worker are nominal (one-hot / category / native). Numeric: duration, amount, installment rate, residence-since, age, existing credits, people liable (small-integer attributes kept numeric).
- **No cash-flow data:** the files have no loan amounts repaid, recoveries or cost fields, so there is **no profit analysis** for this dataset.
- **No time variable:** random stratified splits only; no out-of-time evaluation is possible.
- **Small-sample note:** 1,000 rows and 300 positives; 30 repeated stratified 70/30 splits give about 210 training and 90 test positives per repeat. Repeat-level estimates are noisy and repeats share data (corrected resampled tests are used).
- **Cost matrix:** the source documents a 5:1 cost for misclassifying bad as good; it is **not used** here (no profit analysis).
- **Sensitive attributes:** personal status/sex, age and foreign-worker are included as in the standard benchmark; this is not a fairness-reviewed feature set.

## Column types

- nominal: checking_status, credit_history, purpose, savings, personal_status_sex, other_debtors, property, other_installment_plans, housing, job, telephone, foreign_worker
- ordinal (numeric codes in documented order): employment_since
- numeric: duration_months, credit_amount, installment_rate, residence_since, age, existing_credits, people_liable
