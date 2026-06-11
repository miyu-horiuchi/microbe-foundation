"""assess_drift: no_drift under the null, data_drift under a real shift."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend
from microbe_model.monitoring.drift import assess_drift, DriftReport
from microbe_model.monitoring.reference import ReferenceManifold


def _manifold(seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(300, 4))
    m = ReferenceManifold(backend=EuclideanBackend(k=10), max_reference=None).fit(ref)
    return m, ref, rng


def test_no_drift_on_same_distribution():
    m, ref, rng = _manifold()
    test = rng.normal(size=(80, 4))
    report = assess_drift(m, ref, test, n_perm=200, seed=0)
    assert isinstance(report, DriftReport)
    assert report.classification == "no_drift"


def test_data_drift_on_shift():
    m, ref, rng = _manifold()
    test = rng.normal(size=(80, 4)) + 6.0
    report = assess_drift(m, ref, test, n_perm=200, seed=0)
    assert report.classification == "data_drift"
    assert report.ood_rate > 0.5
    assert "Concept drift" in report.recommendation
