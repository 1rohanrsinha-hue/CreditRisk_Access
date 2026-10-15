"""Bootstrap CIs, paired bootstrap, repeat summaries."""
import numpy as np
import pandas as pd


def bootstrap_ci(fn, n, n_boot=1000, seed=0, alpha=0.05, chunk=100):
    """CI of scalar fn(idx) over bootstrap resamples of range(n)."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = fn(rng.integers(0, n, n))
    return float(np.mean(vals)), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_mean_diff_ci(d, n_boot=1000, seed=0, alpha=0.05, chunk=50):
    """Paired bootstrap of mean(d) where d_i = per-row difference between two
    policies/models on the same rows. Returns (mean, lo, hi, p_two_sided)."""
    d = np.asarray(d, float); n = len(d)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(0, n_boot, chunk):
        k = min(chunk, n_boot - len(means))
        idx = rng.integers(0, n, (k, n))
        means.extend(d[idx].mean(axis=1))
    means = np.asarray(means)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)), float(min(p, 1.0))


def boot_youden(y, s, n_boot=1000, seed=0, nbins=1000):
    """Bootstrap distribution of Youden's t* using a fixed quantile binning of
    the scores (fast; resolution = bin width)."""
    y = np.asarray(y); s = np.asarray(s, float); n = len(y)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, nbins + 1)))
    nb = len(edges)
    b = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, nb - 1)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        pos = np.bincount(b[idx], weights=y[idx], minlength=nb)
        tot = np.bincount(b[idx], minlength=nb).astype(float)
        neg = tot - pos
        P, N = max(pos.sum(), 1), max(neg.sum(), 1)
        tpr = np.cumsum(pos[::-1])[::-1] / P          # flag if score >= edge_k
        fpr = np.cumsum(neg[::-1])[::-1] / N
        out[i] = edges[int(np.argmax(tpr - fpr))]
    return out


def summarize(df, by, cols):
    """mean and sd across repeats (rows) for each group in `by`."""
    g = df.groupby(by)[cols]
    m, s = g.mean().add_suffix("_mean"), g.std(ddof=1).add_suffix("_sd")
    n = g.size().rename("n_rep")
    return pd.concat([m, s, n], axis=1).reset_index()
