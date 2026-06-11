"""Tier 1 drift classifier.

Combines a per-genome OOD rate (Tier 0 scores vs the calibrated threshold) with a
population-level MMD permutation p-value. Flags data drift only; concept drift
(P(Y|X)) is unobservable from embeddings and is surfaced as a recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mmd import mmd_permutation_test
from .reference import ReferenceManifold

DATA_DRIFT_REC = (
    "Data drift detected (P(X) shift). Characterize the new distribution, measure "
    "OOD rates, and confirm downstream performance is affected before retraining. "
    "Concept drift (P(Y|X)) is NOT observable from embeddings; acquire fresh labels "
    "to assess it."
)
NO_DRIFT_REC = (
    "No distributional shift detected at the chosen alpha. Embedding monitoring "
    "cannot rule out concept drift (P(Y|X)); only fresh labels can."
)


@dataclass
class DriftReport:
    n_test: int
    ood_rate: float
    ood_threshold: float
    mmd2: float
    p_value: float
    classification: str
    recommendation: str


def _subsample(X: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    """Return at most n_max rows of X, sampled without replacement."""
    if len(X) <= n_max:
        return X
    idx = rng.choice(len(X), n_max, replace=False)
    return X[idx]


def assess_drift(
    manifold: ReferenceManifold,
    X_ref_sample: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 0.05,
    ood_rate_threshold: float = 0.2,
    n_perm: int = 200,
    max_mmd_sample: int = 2000,
    seed: int = 0,
) -> DriftReport:
    """Assess whether X_test has drifted from the reference distribution.

    The MMD permutation test builds a full (n_ref+n_test)^2 Gram matrix, so both
    inputs are subsampled to at most ``max_mmd_sample`` rows before the test to keep
    memory bounded at production scale. The per-genome OOD rate uses all of X_test.
    """
    X_test = np.asarray(X_test, dtype=float)
    X_ref_sample = np.asarray(X_ref_sample, dtype=float)
    scores = manifold.ood_score(X_test)
    ood_rate = float((scores > manifold.threshold_).mean())

    rng = np.random.default_rng(seed)
    mmd_ref = _subsample(X_ref_sample, max_mmd_sample, rng)
    mmd_test = _subsample(X_test, max_mmd_sample, rng)
    mmd2, p_value = mmd_permutation_test(
        manifold.standardize(mmd_ref),
        manifold.standardize(mmd_test),
        n_perm=n_perm,
        seed=seed,
    )

    drifted = (p_value < alpha) or (ood_rate > ood_rate_threshold)
    classification = "data_drift" if drifted else "no_drift"
    recommendation = DATA_DRIFT_REC if drifted else NO_DRIFT_REC
    return DriftReport(
        n_test=len(X_test),
        ood_rate=ood_rate,
        ood_threshold=float(manifold.threshold_),
        mmd2=mmd2,
        p_value=p_value,
        classification=classification,
        recommendation=recommendation,
    )
