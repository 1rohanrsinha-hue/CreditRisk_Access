"""Score-dependent cost baseline: approve iff (1-p)*mu_good(x) + p*mu_bad(x) > 0
with p a calibrated P(default) and mu_good/mu_bad LightGBM regressors of
realized profit on repaid / defaulted loans (application-time features only)."""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from .features import FrameEncoder


def _reg(seed, n_jobs):
    return Pipeline([('enc', FrameEncoder(mode='category')),
                     ('reg', LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                           min_child_samples=100, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.9, random_state=seed,
                                           n_jobs=n_jobs, verbose=-1))])


class ExpectedProfit:
    def fit(self, X, y, profit, seed=42, n_jobs=4):
        y = np.asarray(y); profit = np.asarray(profit)
        self.good = _reg(seed, n_jobs).fit(X[y == 0], profit[y == 0])
        self.bad = _reg(seed, n_jobs).fit(X[y == 1], profit[y == 1])
        return self

    def mu(self, X):
        return self.good.predict(X), self.bad.predict(X)

    def expected(self, X, p):
        mg, mb = self.mu(X)
        return (1 - np.asarray(p)) * mg + np.asarray(p) * mb
