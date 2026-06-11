"""Pluggable distance backends for OOD scoring.

Each backend exposes the same interface so the validation harness can A/B them:
    fit(X_ref) -> self
    score(X)   -> np.ndarray   # per-row OOD score, higher = more out-of-distribution

EuclideanBackend scores in standardized embedding space (baseline).
DiffusionBackend scores in diffusion-map coordinates (curvature-aware contender).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

from .mmd import median_gamma


class DistanceBackend(Protocol):
    def fit(self, X_ref: np.ndarray) -> "DistanceBackend": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


class EuclideanBackend:
    """Mean distance to the k nearest reference points in standardized space."""

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def fit(self, X_ref: np.ndarray) -> "EuclideanBackend":
        X_ref = np.asarray(X_ref, dtype=float)
        k = min(self.k, len(X_ref))
        self._nn = NearestNeighbors(n_neighbors=k).fit(X_ref)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dist, _ = self._nn.kneighbors(X)
        return dist.mean(axis=1)


class DiffusionBackend:
    """OOD score = mean k-NN distance in diffusion-map coordinates.

    Diffusion coordinates are the discrete heat-kernel embedding of the
    reference manifold. Out-of-sample points are placed via Nyström extension:
    for any X, coords(X) = row_normalize(rbf(X, X_ref)) @ phi, which reproduces
    the reference coordinates exactly when X == X_ref.
    """

    def __init__(self, n_components: int = 10, k: int = 10, gamma: float | None = None) -> None:
        self.n_components = n_components
        self.k = k
        self.gamma = gamma

    def fit(self, X_ref: np.ndarray) -> "DiffusionBackend":
        X_ref = np.asarray(X_ref, dtype=float)
        self._X_ref = X_ref
        self.gamma_ = self.gamma if self.gamma is not None else median_gamma(X_ref)

        W = rbf_kernel(X_ref, X_ref, gamma=self.gamma_)
        d = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        A = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]  # symmetric normalized affinity

        ncomp = min(self.n_components + 1, len(X_ref) - 1)
        vals, vecs = eigsh(A, k=ncomp, which="LM")
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # Drop the trivial leading (constant) component.
        self.lambdas_ = vals[1:]
        self.phi_ = vecs[:, 1:] * d_inv_sqrt[:, None]  # right eigenvectors of the random walk

        self.coords_ref_ = self._coords(X_ref)
        k = min(self.k, len(X_ref))
        self._nn = NearestNeighbors(n_neighbors=k).fit(self.coords_ref_)
        return self

    def _coords(self, X: np.ndarray) -> np.ndarray:
        Wx = rbf_kernel(X, self._X_ref, gamma=self.gamma_)
        Px = Wx / Wx.sum(axis=1, keepdims=True)
        return Px @ self.phi_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._coords(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray) -> np.ndarray:
        coords = self._coords(np.asarray(X, dtype=float))
        dist, _ = self._nn.kneighbors(coords)
        return dist.mean(axis=1)
