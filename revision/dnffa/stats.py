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


# ---------------------------------------------------------------------------
# Single-subject permutation tests: swap the two models' predictions per image
# ---------------------------------------------------------------------------

def _swap_masks(rng, n_items, n_perm, batch_size):
    """Boolean (n_items, batch) masks, True where an item's model labels are swapped."""
    for start in range(0, n_perm, batch_size):
        yield rng.random((n_items, min(batch_size, n_perm - start))) < 0.5


def _corr_columns(x, y_centered):
    """Pearson r of each column of ``x`` with a centered vector ``y``."""
    xc = x - x.mean(axis=0)
    return (y_centered @ xc) / np.sqrt((xc ** 2).sum(axis=0) * (y_centered ** 2).sum())


def permutation_test_univar(pred_a, pred_b, true, n_perm=10000, seed=0, batch_size=1000):
    """Test r(pred_a, true) > r(pred_b, true) by swapping the two predictions per image.

    ``pred_a``, ``pred_b`` and ``true`` hold one value per image (the ROI-mean response).
    Returns the observed difference r_a - r_b, the null differences and the one-sided
    p-value (1 + #{null >= observed}) / (1 + n_perm).
    """
    pred_a, pred_b, true = (np.asarray(v, dtype=float) for v in (pred_a, pred_b, true))
    tc = true - true.mean()
    observed = (_corr_columns(pred_a[:, None], tc) - _corr_columns(pred_b[:, None], tc))[0]

    rng = np.random.default_rng(seed)
    null = []
    for swap in _swap_masks(rng, len(true), n_perm, batch_size):
        perm_a = np.where(swap, pred_b[:, None], pred_a[:, None])
        perm_b = np.where(swap, pred_a[:, None], pred_b[:, None])
        null.append(_corr_columns(perm_a, tc) - _corr_columns(perm_b, tc))
    null = np.concatenate(null)
    return observed, null, (1 + np.sum(null >= observed)) / (1 + n_perm)


def _row_zscore(x):
    """Rows scaled so that z_i . z_j is the Pearson r of rows i and j."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean(axis=1, keepdims=True)
    return xc / np.sqrt((xc ** 2).sum(axis=1, keepdims=True))


def permutation_test_rsa(pred_a, pred_b, true, n_perm=10000, seed=0, batch_size=1000):
    """Test r(RDM_a, RDM_true) > r(RDM_b, RDM_true) by swapping the two predictions per image.

    ``pred_a``, ``pred_b`` and ``true`` are (images, voxels) response patterns; RDMs are
    correlation distances as in ``pdist(..., 'correlation')``. Each permutation gives image i
    the other model's predicted pattern with probability 0.5 and recomputes both RDMs.

    The RDM of a permuted prediction only contains entries of the three cross-model RDMs
    D_aa, D_bb and D_ab (D_xy[i, j] = 1 - r(pred_x[i], pred_y[j])). Every sum the Pearson r
    needs (sum d, sum d^2, sum d * t) is therefore a quadratic form in the swap mask s and is
    computed for a whole batch of permutations with matrix products instead of building
    each permuted RDM. Same values as the direct computation, up to floating-point error.

    Returns (observed r_a - r_b, null differences, one-sided p) as ``permutation_test_univar``,
    or NaNs if a predicted pattern is constant across voxels (its correlation distances are
    undefined).
    """
    za, zb = _row_zscore(pred_a), _row_zscore(pred_b)
    if not (np.all(np.isfinite(za)) and np.all(np.isfinite(zb))):
        return np.nan, np.full(n_perm, np.nan), np.nan

    n = za.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    true_rdm = 1 - _row_zscore(true) @ _row_zscore(true).T
    tc = np.where(off_diag, true_rdm - true_rdm[np.triu_indices(n, 1)].mean(), 0)
    tc_ss = (tc ** 2).sum() / 2
    n_pairs = n * (n - 1) / 2

    rdms = {'aa': 1 - za @ za.T, 'bb': 1 - zb @ zb.T, 'ab': 1 - za @ zb.T}
    # per summed quantity (d, d^2, d * t) the weight matrices of the three RDMs; diagonal never counts
    weights = [{key: np.where(off_diag, f(d), 0) for key, d in rdms.items()}
               for f in (lambda d: d, np.square, lambda d: d * tc)]

    def diff(swap):
        """r_a - r_b for each column of the swap mask.

        With k = 1 - s marking images that keep their own model's pattern, the permuted RDM of
        model a sums to k'W_aa k / 2 + s'W_bb s / 2 + k'W_ab s, and model b's to
        s'W_aa s / 2 + k'W_bb k / 2 + s'W_ab k. Substituting s = 1 - k, all of them follow from
        W @ k and the row / column sums of W.
        """
        keep = 1 - swap.astype(float)
        sums_a, sums_b = [], []
        for w in weights:
            q = {key: np.einsum('ip,ip->p', keep, w[key] @ keep) for key in w}  # k'W k
            row = {key: w[key].sum(axis=1) @ keep for key in w}                   # k'W 1
            col = w['ab'].sum(axis=0) @ keep                                      # 1'W_ab k
            total = {key: w[key].sum() for key in w}
            # s'W s = 1'W 1 - 2 k'W 1 + k'W k for the symmetric W_aa, W_bb
            swapped = {key: total[key] - 2 * row[key] + q[key] for key in ('aa', 'bb')}
            sums_a.append(0.5 * q['aa'] + 0.5 * swapped['bb'] + row['ab'] - q['ab'])
            sums_b.append(0.5 * swapped['aa'] + 0.5 * q['bb'] + col - q['ab'])

        def corr(s_d, s_d2, s_dt):
            return s_dt / np.sqrt((s_d2 - s_d ** 2 / n_pairs) * tc_ss)

        return corr(*sums_a) - corr(*sums_b)

    observed = diff(np.zeros((n, 1), dtype=bool))[0]
    rng = np.random.default_rng(seed)
    null = np.concatenate([diff(swap) for swap in _swap_masks(rng, n, n_perm, batch_size)])
    return observed, null, (1 + np.sum(null >= observed)) / (1 + n_perm)
