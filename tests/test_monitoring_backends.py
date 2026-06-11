"""Distance backends: OOD score rises with distance from the reference support."""
from __future__ import annotations

import numpy as np

from microbe_model.monitoring.backends import EuclideanBackend


def test_euclidean_scores_far_higher_than_near():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(200, 4))
    backend = EuclideanBackend(k=10).fit(ref)
    near = rng.normal(size=(20, 4))            # same distribution
    far = rng.normal(size=(20, 4)) + 8.0       # shifted far away
    assert backend.score(far).mean() > backend.score(near).mean()


def test_euclidean_score_shape():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(100, 3))
    backend = EuclideanBackend(k=5).fit(ref)
    X = rng.normal(size=(7, 3))
    s = backend.score(X)
    assert s.shape == (7,)


from microbe_model.monitoring.backends import DiffusionBackend


def test_diffusion_nystrom_roundtrip():
    """transform() on the reference reproduces the fitted reference coordinates."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(150, 5))
    backend = DiffusionBackend(n_components=6, k=10).fit(ref)
    coords = backend.transform(ref)
    assert np.allclose(coords, backend.coords_ref_, atol=1e-8)


def test_diffusion_scores_far_higher_than_near():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(200, 5))
    backend = DiffusionBackend(n_components=6, k=10).fit(ref)
    near = rng.normal(size=(20, 5))
    far = rng.normal(size=(20, 5)) + 8.0
    assert backend.score(far).mean() > backend.score(near).mean()
