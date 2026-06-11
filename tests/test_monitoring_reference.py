"""ReferenceManifold: standardize, calibrate threshold, score, persist."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend
from microbe_model.monitoring.reference import ReferenceManifold


def _fit_simple(seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(300, 4))
    return ReferenceManifold(backend=EuclideanBackend(k=10), max_reference=None).fit(ref), rng


def test_flags_far_points_as_ood():
    m, rng = _fit_simple()
    far = rng.normal(size=(15, 4)) + 10.0
    assert m.is_ood(far).all()


def test_does_not_flag_inliers_en_masse():
    m, rng = _fit_simple()
    inliers = rng.normal(size=(100, 4))
    # threshold is the 95th pct of reference scores -> roughly <=10% inliers flagged
    assert m.is_ood(inliers).mean() < 0.25


def test_anchor_subsampling_caps_reference():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(500, 4))
    m = ReferenceManifold(backend=EuclideanBackend(k=5), max_reference=100, seed=0).fit(ref)
    # backend was fit on at most max_reference anchors
    assert m.backend._nn.n_samples_fit_ == 100


def test_save_load_roundtrip(tmp_path):
    m, rng = _fit_simple()
    X = rng.normal(size=(10, 4))
    before = m.ood_score(X)
    path = tmp_path / "manifold.pkl"
    m.save(path)
    loaded = ReferenceManifold.load(path)
    after = loaded.ood_score(X)
    assert np.allclose(before, after)
