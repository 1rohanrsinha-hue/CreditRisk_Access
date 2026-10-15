"""Threshold rules. Convention: flag a loan as 'default' when score >= t;
a lending policy approves when score < t. Every rule is picked on OOF
(train+val) scores and then evaluated on the untouched test set."""
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def youden_threshold(y, s) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def f1_max_threshold(y, s) -> float:
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def gmean_threshold(y, s) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    g = np.sqrt(tpr * (1 - fpr))
    return float(thr[int(np.argmax(g))])


def recall_target_threshold(y, s, target=0.70) -> float:
    """Highest threshold whose recall on (y, s) is >= target."""
    fpr, tpr, thr = roc_curve(y, s)
    k = int(np.argmax(tpr >= target))
    return float(thr[k])


def prevalence_threshold(y_train) -> float:
    return float(np.mean(y_train))


def elkan_threshold(c_fp, c_fn) -> float:
    """Elkan/Bayes-optimal cutoff on *calibrated* P(default): flag if
    p*c_fn > (1-p)*c_fp  <=>  p > c_fp / (c_fp + c_fn)."""
    return float(c_fp / (c_fp + c_fn))


def profit_max_threshold(s, profit) -> float:
    """argmax over t of sum(profit[score < t]) (approve low-risk loans).
    Exact search over unique score boundaries; ties are never split."""
    s = np.asarray(s, float); profit = np.asarray(profit, float)
    o = np.argsort(s, kind='stable'); ss = s[o]
    cum = np.concatenate([[0.0], np.cumsum(profit[o])])        # cum[k] = approve k lowest
    n = len(ss)
    ok = np.ones(n + 1, bool)
    ok[1:n] = ss[:-1] < ss[1:]                                 # boundary between distinct scores
    cum_masked = np.where(ok, cum, -np.inf)
    k = int(np.argmax(cum_masked))
    return float(ss[k]) if k < n else float(ss[-1]) + 1e-9     # k=0 -> reject all (t = min score)


def policy_outcome(s, t, y, profit=None):
    """Approve if s < t. Returns profit per 1000 applicants, acceptance rate and
    bad rate among approved."""
    ap = np.asarray(s) < t
    n = len(ap)
    out = {'accept_rate': float(ap.mean()),
           'bad_rate_approved': float(np.mean(np.asarray(y)[ap])) if ap.any() else np.nan}
    if profit is not None:
        out['profit_per_1000'] = float(np.asarray(profit)[ap].sum() / n * 1000)
    return out


def youden_flatness(y, s, tol=0.005):
    """(J_max, width of the threshold interval with J >= J_max - tol)."""
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    jm = float(j.max())
    keep = (j >= jm - tol) & np.isfinite(thr)
    if not keep.any():
        return jm, 0.0
    return jm, float(thr[keep].max() - thr[keep].min())
