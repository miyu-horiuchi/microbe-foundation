"""Maximum Mean Discrepancy two-sample test with a permutation-based p-value.

The permutation test yields a valid p-value with controlled type-I error — the
population-level drift verifier. Kernel is RBF with a median-heuristic bandwidth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel


def median_gamma(X: np.ndarray) -> float:
    """RBF gamma from the median-heuristic: gamma = 1 / median(pairwise sq. dist)."""
    X = np.asarray(X, dtype=float)
    d2 = pdist(X, metric="sqeuclidean")
    med = float(np.median(d2)) if d2.size else 0.0
    if med <= 0:
        med = 1.0
    return 1.0 / med


def _mmd2_from_gram(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """Unbiased MMD^2 estimate from precomputed kernel blocks."""
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def mmd_permutation_test(
    X_ref: np.ndarray,
    X_test: np.ndarray,
    *,
    gamma: float | None = None,
    n_perm: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (observed MMD^2, permutation p-value) for X_ref vs X_test."""
    X_ref = np.asarray(X_ref, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    Z = np.vstack([X_ref, X_test])
    if gamma is None:
        gamma = median_gamma(Z)
    K = rbf_kernel(Z, Z, gamma=gamma)
    m = len(X_ref)
    N = len(Z)

    def mmd2(idx_x: np.ndarray, idx_y: np.ndarray) -> float:
        Kxx = K[np.ix_(idx_x, idx_x)]
        Kyy = K[np.ix_(idx_y, idx_y)]
        Kxy = K[np.ix_(idx_x, idx_y)]
        return _mmd2_from_gram(Kxx, Kyy, Kxy)

    observed = mmd2(np.arange(m), np.arange(m, N))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        if mmd2(perm[:m], perm[m:]) >= observed:
            count += 1
    p_value = (1 + count) / (1 + n_perm)
    return observed, p_value
