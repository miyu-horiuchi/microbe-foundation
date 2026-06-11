"""MMD permutation test: valid type-I under the null, power under a real shift."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.mmd import median_gamma, mmd_permutation_test


def test_median_gamma_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    assert median_gamma(X) > 0


def test_null_gives_large_pvalue():
    """Same distribution -> not flagged as different (p well above 0.05)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))
    Y = rng.normal(size=(80, 5))
    mmd2, p = mmd_permutation_test(X, Y, n_perm=200, seed=0)
    assert p > 0.05


def test_shift_gives_small_pvalue():
    """A clear mean shift -> low p-value (drift detected)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 5))
    Y = rng.normal(size=(80, 5)) + 3.0
    mmd2, p = mmd_permutation_test(X, Y, n_perm=200, seed=0)
    assert p < 0.05
    assert mmd2 > 0


def test_deterministic_with_seed():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    Y = rng.normal(size=(40, 3)) + 1.0
    a = mmd_permutation_test(X, Y, n_perm=100, seed=7)
    b = mmd_permutation_test(X, Y, n_perm=100, seed=7)
    assert a == b
