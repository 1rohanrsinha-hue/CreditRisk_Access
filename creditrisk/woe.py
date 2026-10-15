"""Weight-of-Evidence scorecard: quantile-binned numerics + nominal categories,
WoE with additive smoothing, logistic regression on the WoE features."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import NOMINAL


class WoEScorecard:
    def __init__(self, n_bins=10, alpha=0.5, C=1.0, seed=42):
        self.n_bins, self.alpha, self.C, self.seed = n_bins, alpha, C, seed

    # ---- binning
    def _bin_numeric(self, c, v):
        e = self.edges_[c]
        b = np.searchsorted(e, v, side='right')            # 0..len(e)
        b = np.where(np.isnan(v), len(e) + 1, b)           # missing bin
        return b.astype(int)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y); X = X.copy()
        self.nominal_ = [c for c in NOMINAL if c in X.columns]
        self.numeric_ = [c for c in X.columns if c not in self.nominal_]
        self.edges_, self.woe_, self.labels_ = {}, {}, {}
        G, Bd = (y == 0).sum(), (y == 1).sum()
        for c in self.numeric_:
            v = X[c].values.astype(float)
            q = np.unique(np.nanquantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.edges_[c] = q
            b = self._bin_numeric(c, v)
            self.woe_[c], self.labels_[c] = self._woe(b, y, G, Bd, len(q) + 2), \
                [f"<= {q[0]:.4g}"] + [f"({q[i-1]:.4g}, {q[i]:.4g}]" for i in range(1, len(q))] + [f"> {q[-1]:.4g}", "missing"]
        for c in self.nominal_:
            cats = sorted(X[c].fillna('missing').astype(str).unique())
            self.edges_[c] = cats
            b = pd.Categorical(X[c].fillna('missing').astype(str), categories=cats).codes
            self.woe_[c], self.labels_[c] = self._woe(b, y, G, Bd, len(cats)), cats
        Z = self.transform(X)
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed).fit(Z, y)
        self.cols_ = list(Z.columns)
        return self

    def _woe(self, b, y, G, Bd, nb):
        good = np.bincount(b[y == 0], minlength=nb).astype(float)
        bad = np.bincount(b[y == 1], minlength=nb).astype(float)
        a = self.alpha
        return np.log((good + a) / (G + a * nb)) - np.log((bad + a) / (Bd + a * nb))

    def transform(self, X):
        out = {}
        for c in self.numeric_:
            b = self._bin_numeric(c, X[c].values.astype(float))
            out[c] = self.woe_[c][np.clip(b, 0, len(self.woe_[c]) - 1)]
        for c in self.nominal_:
            cats = self.edges_[c]
            b = pd.Categorical(X[c].fillna('missing').astype(str), categories=cats).codes
            w = self.woe_[c]
            out[c] = np.where(b >= 0, w[np.clip(b, 0, len(w) - 1)], 0.0)       # unseen category -> neutral
        return pd.DataFrame(out, index=X.index)

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.transform(X)[self.cols_])

    def points_table(self, top=5, pdo=50, base_points=600, base_odds=1 / 19):
        """Standard scorecard scaling: factor = PDO/ln 2; points = -(beta_j * WoE_bj) * factor
        (WoE is ln(good/bad); LR models P(default)). Intercept folded into base points."""
        factor = pdo / np.log(2)
        coef = pd.Series(self.lr_.coef_[0], index=self.cols_)
        rows = []
        for c in coef.abs().sort_values(ascending=False).index[:top]:
            for lab, w in zip(self.labels_[c], self.woe_[c]):
                rows.append(dict(feature=c, bin=lab, woe=w, coef=coef[c], points=-coef[c] * w * factor))
        return pd.DataFrame(rows)
