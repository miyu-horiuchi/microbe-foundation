"""Pure-helper tests for cosine family-collapse diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import cosine_family_collapse as cfc


def test_normalize_rows_handles_zero_vectors():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = cfc.normalize_rows(x)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])


def test_centroid_matrix_normalizes_family_centroids():
    x = cfc.normalize_rows(np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
    ]))
    fams, centroids = cfc.centroid_matrix(x, np.array(["b", "a", "a"]))
    assert fams.tolist() == ["a", "b"]
    assert centroids.shape == (2, 2)
    assert np.allclose(np.linalg.norm(centroids, axis=1), [1.0, 1.0])


def test_nearest_centroid_cosine_identifies_closest_family():
    centroids = cfc.normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
    query = cfc.normalize_rows(np.array([[0.9, 0.1], [0.2, 0.8]]))
    sims = cfc.nearest_centroid_cosine(query, centroids)
    assert sims.shape == (2,)
    assert sims[0] > 0.99
    assert sims[1] > 0.97


def test_cosine_knn_transfer_returns_probabilities_and_similarities():
    x_train = cfc.normalize_rows(np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ]))
    y_train = np.array([1, 1, 0, 0])
    x_test = cfc.normalize_rows(np.array([[1.0, 0.05], [0.05, 1.0]]))
    prob, top1, topk = cfc.cosine_knn_transfer(x_train, y_train, x_test, k=2)
    assert prob.tolist() == [1.0, 0.0]
    assert np.all(top1 > 0.99)
    assert np.all(topk > 0.95)


def test_geometry_verdicts():
    assert cfc.geometry_verdict(0.8, -0.2) == "coverage/geometry-limited"
    assert cfc.geometry_verdict(0.55, 0.0) == "label not local in cosine space"
    assert cfc.geometry_verdict(0.7, -0.2) == "mixed: cosine support matters"
