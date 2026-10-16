"""Top-k risk-increasing reason codes from per-loan additive contributions
(EBM term scores or TreeSHAP; positive = raises log-odds of default).
Plain-language labels are a lookup table; NO claim is made that these satisfy
adverse-action-notice requirements (a legal question outside this study)."""
import numpy as np

LABELS = {
    'interest_rate': "Interest rate assigned to the loan is high",
    'fico_mean': "Credit score is lower than comparable borrowers",
    'debt_to_income': "Debt payments are high relative to income",
    'annual_income': "Annual income is low relative to comparable borrowers",
    'loan_amount': "Requested loan amount is high",
    'credit_inquiries': "Number of recent credit inquiries is high",
    'delinquencies': "Recent delinquencies on credit accounts",
    'public_records': "Derogatory public records on file",
    'collections': "Accounts have been sent to collections",
    'num_credit_lines': "Number of open credit accounts is unusual",
    'total_credit_lines': "Total number of credit accounts is unusual",
    'mortgage_accounts': "Number of mortgage accounts",
    'employment_length': "Length of employment is short",
    'emp_length_missing': "Employment length not reported",
    'home_ownership': "Housing status (rent/own/mortgage)",
    'loan_purpose': "Stated purpose of the loan",
}


def label(f):
    return LABELS.get(f, f)


def top_codes(C, cols, k=3, exclude=()):
    """C: (n x p) contributions; returns a list (length n) of feature-name lists, the up-to-k
    largest POSITIVE contributions among non-excluded features, strongest first."""
    cols = np.asarray(cols); C = np.asarray(C, float).copy()
    for e in exclude:
        C[:, list(cols).index(e)] = -np.inf
    order = np.argsort(-C, axis=1)[:, :k]
    out = []
    for i in range(len(C)):
        out.append([cols[j] for j in order[i] if C[i, j] > 0])
    return out


def overlap_stats(A, B):
    """Per-loan overlap count, Jaccard and the number of codes available on both sides."""
    ov = np.array([len(set(a) & set(b)) for a, b in zip(A, B)])
    jac = np.array([len(set(a) & set(b)) / max(1, len(set(a) | set(b))) for a, b in zip(A, B)])
    avail = np.array([min(len(a), len(b)) for a, b in zip(A, B)])
    return ov, jac, avail
