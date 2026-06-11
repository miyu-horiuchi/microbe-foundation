"""ReferenceManifold: the fitted reference distribution for OOD scoring.

Standardizes embeddings, optionally subsamples anchor points (so diffusion
eigendecomposition stays tractable on ~19k genomes), delegates scoring to a
distance backend, and calibrates an OOD threshold from reference self-scores.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

from .backends import DistanceBackend, EuclideanBackend


@dataclass
class ReferenceManifold:
    backend: DistanceBackend = field(default_factory=EuclideanBackend)
    max_reference: int | None = 4000
    threshold_quantile: float = 0.95
    seed: int = 0
    mean_: np.ndarray | None = field(default=None, repr=False)
    scale_: np.ndarray | None = field(default=None, repr=False)
    threshold_: float | None = None

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """Apply the reference mean/scale to X. Public so other modules (e.g. the
        drift classifier's MMD test) can put samples in the same space as the
        fitted backend without reaching into internals."""
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    # Backwards-compatible internal alias.
    _standardize = standardize

    def fit(self, X_ref: np.ndarray) -> "ReferenceManifold":
        X_ref = np.asarray(X_ref, dtype=float)
        self.mean_ = X_ref.mean(axis=0)
        scale = X_ref.std(axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale

        Xs = self._standardize(X_ref)
        anchors = Xs
        if self.max_reference is not None and len(Xs) > self.max_reference:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(Xs), self.max_reference, replace=False)
            anchors = Xs[idx]

        self.backend.fit(anchors)
        ref_scores = self.backend.score(anchors)
        self.threshold_ = float(np.quantile(ref_scores, self.threshold_quantile))
        return self

    def ood_score(self, X: np.ndarray) -> np.ndarray:
        return self.backend.score(self._standardize(X))

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        return self.ood_score(X) > self.threshold_

    def save(self, path) -> None:
        # pickle: the fitted state holds sklearn NearestNeighbors objects with no
        # clean JSON form. These are the user's own locally-fitted artifacts, not
        # untrusted input — only load manifolds you produced yourself.
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "ReferenceManifold":
        # Trust boundary: only load a manifold file you created (see save()).
        with open(path, "rb") as fh:
            return pickle.load(fh)
