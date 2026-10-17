# Data card: Default of Credit Card Clients (Taiwan)

- **Source name:** UCI Machine Learning Repository, 'Default of Credit Card Clients' (Yeh and Lien, 2009)
- **Source URL (citation):** FILL IN  (direct-download URL used by `scripts/fetch_external.sh`)
- **Licence:** FILL IN (not verified)
- **Access date:** FILL IN
- **File:** `external_data/taiwan.xls` (git-ignored); sha256 prefix `30c6be3abd8dcfd3`
- **Rows:** 30,000   **Feature columns:** 23   (nominal 3, ordinal 6, numeric 14)
- **Positive class (bad/default) rate:** 0.2212  (6,636 positives)
- **Label definition (verified in code):** column 'default payment next month' (1 = default in the next month, 0 = no); used as is, y = 1 for default
- **Missing values:** 0 cells in total
- **Exact duplicate rows (features + label):** 35
- **Post-outcome leak check:** none found by construction: features are limit, demographics, repayment status (PAY_0, PAY_2..PAY_6, April-September 2005) and bill/payment amounts for the same six months; the label is the October payment. Note PAY_0 is the September status (the source names it PAY_0, not PAY_1). ID is dropped.
- **Encoding decisions:** PAY_* (-2 no consumption, -1 paid in full, 0 revolving credit, 1..9 months of delay) have a documented order and are coded as given (ordinal, numeric). SEX is binary nominal. EDUCATION has codes 0-6 of which 0, 5 and 6 are undocumented and 4 = 'others', so it stays nominal; MARRIAGE (0 undocumented) stays nominal. All other columns numeric.
- **No cash-flow data:** the files have no loan amounts repaid, recoveries or cost fields, so there is **no profit analysis** for this dataset.
- **No time variable:** random stratified splits only; no out-of-time evaluation is possible.
- **Population caveat:** one Taiwanese card issuer, one six-month window; the default rate (22%) is far above Lending Club's. Clients, not loans.
- **Sensitive attributes:** SEX, EDUCATION, MARRIAGE and AGE are included as in the standard benchmark.

## Column types

- nominal: SEX, EDUCATION, MARRIAGE
- ordinal (numeric codes in documented order): PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
- numeric: LIMIT_BAL, AGE, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
