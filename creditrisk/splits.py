"""Stratified 64/16/20 train/val/test split, row-aligned with metadata."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(y, seed=42, test_size=0.20, val_size=0.20):
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    tr, va = train_test_split(tv, test_size=val_size, random_state=seed,
                              stratify=np.asarray(y)[tv])
    return tr, va, te
