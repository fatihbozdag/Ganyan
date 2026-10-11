"""Missing measurements contribute zero after within-race standardisation."""
import numpy as np


def zscore_missing(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.zeros_like(values)
    if valid.any():
        std = values[valid].std()
        if std > 1e-9:
            result[valid] = (values[valid] - values[valid].mean()) / std
    return result
