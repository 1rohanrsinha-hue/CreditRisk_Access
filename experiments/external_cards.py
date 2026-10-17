"""Phase 7 / task A: data cards for the external datasets (computed, not hand-typed numbers)."""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.external import load_german, load_taiwan, EXT_DIR

OUT = ROOT / "docs" / "data_cards"
OUT.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def common(name, source, file, X, y, spec, label_txt, leak_txt, enc_txt, extra):
    n, p = X.shape
    miss = X.isna().sum()
    dup = int(pd.concat([X, y.rename('y')], axis=1).duplicated().sum())
    lines = [f"# Data card: {name}", "",
             f"- **Source name:** {source}",
             "- **Source URL (citation):** FILL IN  (direct-download URL used by `scripts/fetch_external.sh`)",
             "- **Licence:** FILL IN (not verified)",
             "- **Access date:** FILL IN",
             f"- **File:** `external_data/{file}` (git-ignored); sha256 prefix `{sha(EXT_DIR / file)}`",
             f"- **Rows:** {n:,}   **Feature columns:** {p}   (nominal {len(spec.nominal)}, ordinal {len(spec.ordinal)}, numeric {len(spec.numeric)})",
             f"- **Positive class (bad/default) rate:** {y.mean():.4f}  ({int(y.sum()):,} positives)",
             f"- **Label definition (verified in code):** {label_txt}",
             f"- **Missing values:** {int(miss.sum())} cells in total" + (f" ({', '.join(f'{c}: {int(v)}' for c, v in miss[miss > 0].items())})" if miss.sum() else ""),
             f"- **Exact duplicate rows (features + label):** {dup}",
             f"- **Post-outcome leak check:** {leak_txt}",
             f"- **Encoding decisions:** {enc_txt}",
             "- **No cash-flow data:** the files have no loan amounts repaid, recoveries or cost fields, so there is **no profit analysis** for this dataset.",
             "- **No time variable:** random stratified splits only; no out-of-time evaluation is possible."]
    lines += extra
    lines += ["", "## Column types", "",
              f"- nominal: {', '.join(spec.nominal)}", f"- ordinal (numeric codes in documented order): {', '.join(spec.ordinal) or '-'}",
              f"- numeric: {', '.join(spec.numeric)}", ""]
    return "\n".join(lines)


def main():
    X, y, s = load_german()
    raw = pd.read_csv(EXT_DIR / "german.data", sep=' ', header=None)
    vc = raw[20].value_counts().to_dict()
    txt = common("Statlog German Credit Data", "UCI Machine Learning Repository, 'Statlog (German Credit Data)' (Hofmann)", "german.data", X, y, s,
                 f"raw column 21 is coded 1 = good ({vc.get(1)}), 2 = bad ({vc.get(2)}); here y = 1 if raw == 2 (bad credit risk)",
                 "none found. All 20 attributes are application-time (checking account, duration, amount, history, purpose, savings, employment, ...); no repayment-performance columns. The 'credit history' attribute (A3) describes earlier loans, not this one.",
                 "A7 (employment duration) is the only attribute with a monotone documented order and is coded 0-4 (ordinal, numeric). Checking status (A1: 'no checking account' breaks the order), savings (A6: 'unknown' breaks the order), credit history, purpose, personal status/sex, other debtors, property, other installment plans, housing, job, telephone and foreign worker are nominal (one-hot / category / native). Numeric: duration, amount, installment rate, residence-since, age, existing credits, people liable (small-integer attributes kept numeric).",
                 ["- **Small-sample note:** 1,000 rows and 300 positives; 30 repeated stratified 70/30 splits give about 210 training and 90 test positives per repeat. Repeat-level estimates are noisy and repeats share data (corrected resampled tests are used).",
                  "- **Cost matrix:** the source documents a 5:1 cost for misclassifying bad as good; it is **not used** here (no profit analysis).",
                  "- **Sensitive attributes:** personal status/sex, age and foreign-worker are included as in the standard benchmark; this is not a fairness-reviewed feature set."])
    (OUT / "german_credit.md").write_text(txt)
    X, y, s = load_taiwan()
    txt = common("Default of Credit Card Clients (Taiwan)", "UCI Machine Learning Repository, 'Default of Credit Card Clients' (Yeh and Lien, 2009)", "taiwan.xls", X, y, s,
                 "column 'default payment next month' (1 = default in the next month, 0 = no); used as is, y = 1 for default",
                 "none found by construction: features are limit, demographics, repayment status (PAY_0, PAY_2..PAY_6, April-September 2005) and bill/payment amounts for the same six months; the label is the October payment. Note PAY_0 is the September status (the source names it PAY_0, not PAY_1). ID is dropped.",
                 "PAY_* (-2 no consumption, -1 paid in full, 0 revolving credit, 1..9 months of delay) have a documented order and are coded as given (ordinal, numeric). SEX is binary nominal. EDUCATION has codes 0-6 of which 0, 5 and 6 are undocumented and 4 = 'others', so it stays nominal; MARRIAGE (0 undocumented) stays nominal. All other columns numeric.",
                 ["- **Population caveat:** one Taiwanese card issuer, one six-month window; the default rate (22%) is far above Lending Club's. Clients, not loans.",
                  "- **Sensitive attributes:** SEX, EDUCATION, MARRIAGE and AGE are included as in the standard benchmark."])
    (OUT / "taiwan_default.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
