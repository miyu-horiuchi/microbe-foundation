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
from sklearn.neighbors import NearestNeighbors


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
