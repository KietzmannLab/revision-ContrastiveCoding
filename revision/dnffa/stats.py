"""Small statistics helpers."""

import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_rel


def finite_mask(*arrays):
    """Boolean mask of entries that are finite in every array."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def valid_pearsonr(x, y, min_valid=750):
    """Pearson r over finite entries, or NaN if fewer than ``min_valid`` remain."""
    valid = finite_mask(x, y)
    if np.sum(valid) <= min_valid:
        return np.nan
    return stats.pearsonr(x[valid], y[valid])[0]


def pairwise_ttest_with_first(model_data, alpha=0.01):
    """Paired t-tests of every entry in ``model_data`` against the first one."""
    n = len(model_data)
    significant_pairs = []
    p_values = np.ones(n)
    for i in range(1, n):
        _, p = ttest_rel(model_data[0], model_data[i])
        p_values[i] = p
        if p < alpha:
            significant_pairs.append((0, i))
    return significant_pairs, p_values
