"""Calibrators and calibration metrics."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    """Logistic regression on logit(score) (near-unregularised)."""
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(s).reshape(-1, 1), y)
        return self

    def predict(self, s):
        return self.m.predict_proba(_logit(s).reshape(-1, 1))[:, 1]


class Isotonic:
    def fit(self, s, y):
        self.m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(s, y)
        return self

    def predict(self, s):
        return self.m.predict(s)


def make_calibrators(s_oof, y_oof):
    return {'raw': None, 'platt': Platt().fit(s_oof, y_oof), 'isotonic': Isotonic().fit(s_oof, y_oof)}


def apply_cal(cal, s):
    return np.asarray(s, float) if cal is None else cal.predict(s)


def ece(y, p, n_bins=15):
    """Expected calibration error with equal-frequency bins."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    order = np.argsort(p, kind='stable')
    chunks = np.array_split(order, n_bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def reliability_curve(y, p, n_bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float)
    order = np.argsort(p, kind='stable')
    chunks = [c for c in np.array_split(order, n_bins) if len(c)]
    return (np.array([p[c].mean() for c in chunks]), np.array([y[c].mean() for c in chunks]))


def prob_metrics(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return {'Brier': brier_score_loss(y, p), 'LogLoss': log_loss(y, p), 'ECE': ece(y, p)}
