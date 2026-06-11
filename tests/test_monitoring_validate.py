"""Validation harness: OOD scores separate out-of-clade from in-clade genomes."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend
from microbe_model.monitoring.validate import error_correlation, evaluate_backend


def test_evaluate_backend_separates_clusters():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, 5))
    heldin = rng.normal(size=(100, 5))          # same distribution -> low OOD
    heldout = rng.normal(size=(100, 5)) + 8.0   # novel clade -> high OOD
    auroc, _, _ = evaluate_backend(
        train, heldout, heldin, EuclideanBackend(k=10), max_reference=None
    )
    assert auroc > 0.9


def test_error_correlation_monotonic():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    errors = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert error_correlation(scores, errors) > 0.99
