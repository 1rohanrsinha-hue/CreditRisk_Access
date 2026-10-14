"""Threshold selection rules. Phase 1 only needs Youden's J."""
import numpy as np
from sklearn.metrics import roc_curve


def youden_threshold(y_true, proba) -> float:
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])
