"""Review tiers on calibrated PD, anchored on the training prevalence pi.
These are REVIEW tiers, not accept/reject cutoffs: on mature vintages no threshold rule
beat approving everyone (Phase 3B, H5), so no tier boundary is a decision to decline."""
import numpy as np
import pandas as pd

LOW_MULT, HIGH_MULT = 0.75, 1.5
TIER_NAMES = ['Tier 1: lower predicted risk', 'Tier 2: typical predicted risk', 'Tier 3: higher predicted risk']
TIER_DEFINITIONS = ['PD < 0.75 x prevalence', '0.75 x prevalence <= PD <= 1.5 x prevalence', 'PD > 1.5 x prevalence']


def tier_edges(pi):
    return LOW_MULT * pi, HIGH_MULT * pi


def assign_tier(pd_, pi):
    """Index 0/1/2 for every PD in [0, 1]; the three tiers partition the whole interval."""
    p = np.asarray(pd_, float)
    if np.isnan(p).any():
        raise ValueError("NaN probability cannot be assigned a tier")
    lo, hi = tier_edges(pi)
    return np.where(p < lo, 0, np.where(p > hi, 2, 1))


def tier_report(y, p, pi, n_boot=1000, seed=0):
    """Share, observed default rate (bootstrap CI) and calibration gap per tier."""
    y = np.asarray(y, float); p = np.asarray(p, float); t = assign_tier(p, pi)
    rng = np.random.default_rng(seed); rows = []
    for k in range(3):
        m = np.where(t == k)[0]
        if len(m) == 0:
            rows.append(dict(tier=TIER_NAMES[k], definition=TIER_DEFINITIONS[k], n=0)); continue
        ym, pm = y[m], p[m]
        bs = [ym[i].mean() for i in (rng.integers(0, len(m), len(m)) for _ in range(n_boot))]
        gap = [(ym[i].mean() - pm[i].mean()) for i in (rng.integers(0, len(m), len(m)) for _ in range(n_boot))]
        rows.append(dict(tier=TIER_NAMES[k], definition=TIER_DEFINITIONS[k], n=len(m), share=len(m) / len(y),
                         observed_default_rate=ym.mean(), obs_ci_lo=np.quantile(bs, .025), obs_ci_hi=np.quantile(bs, .975),
                         mean_pd=pm.mean(), gap_obs_minus_pd=ym.mean() - pm.mean(),
                         gap_ci_lo=np.quantile(gap, .025), gap_ci_hi=np.quantile(gap, .975),
                         calibrated_within_ci=bool(np.quantile(gap, .025) <= 0 <= np.quantile(gap, .975))))
    return pd.DataFrame(rows)
