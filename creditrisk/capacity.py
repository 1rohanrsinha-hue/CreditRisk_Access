"""Capacity-constrained approval helpers (count- and dollar-based) + metrics."""
import numpy as np


def approve_count(key, k):
    """Approve the k share of applicants with the lowest `key`."""
    n_ap = int(round(k * len(key)))
    a = np.zeros(len(key), bool)
    a[np.argsort(key, kind='stable')[:n_ap]] = True
    return a


def approve_dollar(key, funded, k):
    """Approve in ascending key order until cumulative funded dollars reach
    k x total dollars (the last loan that fits is the final one)."""
    o = np.argsort(key, kind='stable')
    cum = np.cumsum(np.asarray(funded, float)[o])
    n_ap = int(np.searchsorted(cum, k * cum[-1], side='right'))
    a = np.zeros(len(key), bool)
    a[o[:n_ap]] = True
    return a


def outcome(a, y, profit, funded):
    n = len(a)
    f = np.asarray(funded, float)
    if not a.any():
        return dict(profit_per_1000=0.0, roi=np.nan, bad_count=np.nan, bad_dollar=np.nan, accept_dollar_share=0.0)
    return dict(profit_per_1000=float(profit[a].sum() / n * 1000),
                roi=float(profit[a].sum() / f[a].sum()),
                bad_count=float(np.mean(y[a])),
                bad_dollar=float((f[a] * y[a]).sum() / f[a].sum()),
                accept_dollar_share=float(f[a].sum() / f.sum()))
