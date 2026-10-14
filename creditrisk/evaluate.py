"""Threshold-dependent and threshold-free metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, brier_score_loss)


def metrics_at(y, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        'Accuracy': accuracy_score(y, pred),
        'Precision': precision_score(y, pred, zero_division=0),
        'Recall': recall_score(y, pred, zero_division=0),
        'F1': f1_score(y, pred, zero_division=0),
        'MCC': matthews_corrcoef(y, pred),
        'ROC_AUC': roc_auc_score(y, proba),
        'PR_AUC': average_precision_score(y, proba),
        'Brier': brier_score_loss(y, proba),
        'Threshold': float(thr),
    }
