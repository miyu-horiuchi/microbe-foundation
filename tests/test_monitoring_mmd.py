"""MMD permutation test: valid type-I under the null, power under a real shift."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.mmd import median_gamma, mmd_permutation_test


def test_median_gamma_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    assert median_gamma(X) > 0


def test_null_type_i_is_calibrated():
    """Under the null (same distribution), the rejection rate stays near alpha.

    A single null draw rejecting at p<0.05 is expected ~5% of the time and proves
    nothing; the meaningful property is that across many independent null draws the
    fraction with p<0.05 hovers around the nominal 0.05 — i.e. the permutation test
    controls the type-I error rate.
    """
    n_draws = 200
    rejections = 0
    for s in range(n_draws):
        rng = np.random.default_rng(1000 + s)
        X = rng.normal(size=(80, 5))
        Y = rng.normal(size=(80, 5))
        _, p = mmd_permutation_test(X, Y, n_perm=100, seed=0)
        rejections += p < 0.05
    rate = rejections / n_draws
    assert rate < 0.12  # generous upper bound around the nominal 0.05


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
