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


# ---------------------------------------------------------------------------------------------
# Direction-aware wording (Phase 5). The sentence depends on whether the applicant's value is
# above or below the training MEDIAN of that feature, or on the category itself; a missing value
# gets its own sentence. This describes the value relative to typical borrowers; it is not a
# causal statement and NOT a regulatory adverse-action notice.
DIRECTION_LABELS = {
    'interest_rate':      ("The interest rate on this loan is higher than typical", "The interest rate on this loan is lower than typical"),
    'fico_mean':          ("Credit score is higher than typical (but still adds risk in combination with other factors)", "Credit score is lower than typical"),
    'debt_to_income':     ("Debt payments are high relative to income", "Debt payments are low relative to income (but add risk in combination with other factors)"),
    'annual_income':      ("Annual income is higher than typical (but adds risk in combination with other factors)", "Annual income is lower than typical"),
    'loan_amount':        ("The requested loan amount is higher than typical", "The requested loan amount is lower than typical"),
    'credit_inquiries':   ("Number of recent credit inquiries is higher than typical", "Number of recent credit inquiries is lower than typical"),
    'delinquencies':      ("More delinquencies in the credit history than typical", "Fewer delinquencies than typical (but adds risk in combination with other factors)"),
    'public_records':     ("More derogatory public records than typical", "Fewer public records than typical (but adds risk in combination with other factors)"),
    'collections':        ("More accounts sent to collections than typical", "Fewer accounts sent to collections than typical (but adds risk in combination with other factors)"),
    'num_credit_lines':   ("Number of open credit accounts is higher than typical", "Number of open credit accounts is lower than typical"),
    'total_credit_lines': ("Total number of credit accounts is higher than typical", "Total number of credit accounts is lower than typical"),
    'mortgage_accounts':  ("Number of mortgage accounts is higher than typical", "Number of mortgage accounts is lower than typical"),
    'employment_length':  ("Length of employment is longer than typical (but adds risk in combination with other factors)", "Length of employment is shorter than typical"),
}
CATEGORY_LABELS = {
    'loan_purpose': "Loan purpose ({v}) is associated with higher risk",
    'home_ownership': "Housing status ({v}) is associated with higher risk",
}
MISSING_LABELS = {
    'emp_length_missing': "Employment length was not reported",
    'employment_length': "Employment length was not reported",
}


def direction_label(feature, value, median=None):
    """Plain-language label for a risk-increasing contribution of `feature` at `value`."""
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature].format(v=str(value).replace('_', ' '))
    if feature == 'emp_length_missing':
        return MISSING_LABELS['emp_length_missing']
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING_LABELS.get(feature, f"{feature} was not reported")
    if feature in DIRECTION_LABELS:
        hi, lo = DIRECTION_LABELS[feature]
        return hi if (median is None or value >= median) else lo
    return LABELS.get(feature, feature)
