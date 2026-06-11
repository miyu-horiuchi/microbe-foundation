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
        n_ref = len(X_ref)
        # Need at least n_components + 2 points: one trivial eigenvector is dropped,
        # and eigsh requires k < n. Fail loudly rather than return degenerate coords.
        if n_ref < self.n_components + 2:
            raise ValueError(
                f"DiffusionBackend needs at least n_components+2={self.n_components + 2} "
                f"reference points; got {n_ref}."
            )
        self._X_ref = X_ref
        self.gamma_ = self.gamma if self.gamma is not None else median_gamma(X_ref)

        W = rbf_kernel(X_ref, X_ref, gamma=self.gamma_)
        # Row sums are >= 1 because the RBF self-affinity (diagonal) is exp(0) = 1;
        # the maximum() guard documents that invariant and survives a future refactor
        # that might zero the diagonal.
        d = np.maximum(W.sum(axis=1), 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        A = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]  # symmetric normalized affinity

        ncomp = min(self.n_components + 1, n_ref - 1)
        vals, vecs = eigsh(A, k=ncomp, which="LM")
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # Drop the trivial leading (constant) component.
        self.lambdas_ = vals[1:]
        # Random-walk (Coifman-Lafon) normalization: phi_ are RIGHT eigenvectors of
        # M = D^-1 W (phi = D^-1/2 v). With P = row-normalized affinity, coords = P @ phi
        # gives the t=1 diffusion coordinate consistently for reference AND out-of-sample
        # points (exact round-trip follows from M phi = lambda phi). This is the
        # random-walk variant, not the symmetric-normalized Nystrom — intentionally so.
        self.phi_ = vecs[:, 1:] * d_inv_sqrt[:, None]

        self.coords_ref_ = self._coords(X_ref)
        k = min(self.k, n_ref)
        self._nn = NearestNeighbors(n_neighbors=k).fit(self.coords_ref_)
        return self

    def _coords(self, X: np.ndarray) -> np.ndarray:
        Wx = rbf_kernel(X, self._X_ref, gamma=self.gamma_)
        row_sums = Wx.sum(axis=1, keepdims=True)
        # A point so far from every reference that all RBF affinities underflow to 0
        # has row_sum == 0; substitute 1 to avoid 0/0 = NaN. score() flags these rows
        # as maximally OOD via the same zero-affinity test, so the placeholder coords
        # are never trusted.
        safe = np.where(row_sums == 0.0, 1.0, row_sums)
        Px = Wx / safe
        return Px @ self.phi_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._coords(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        # Rows with zero affinity to the entire reference are maximally out-of-
        # distribution; their diffusion coords are ill-defined, so score them directly
        # as +inf instead of relying on a NaN coordinate (NaN > threshold is False,
        # which would silently mark the most extreme OOD point as in-distribution).
        zero_affinity = rbf_kernel(X, self._X_ref, gamma=self.gamma_).sum(axis=1) == 0.0
        coords = self._coords(X)
        scores = np.full(len(X), np.inf)
        if (~zero_affinity).any():
            dist, _ = self._nn.kneighbors(coords[~zero_affinity])
            scores[~zero_affinity] = dist.mean(axis=1)
        return scores
