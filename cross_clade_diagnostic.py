"""Cross-clade collapse diagnostic.

Returns a verdict on WHY family-level trait transfer collapses: insufficient
training-clade coverage (fixable without fine-tuning) vs a frozen-representation
wall (only encoder fine-tuning can help). Two diagnostics:

  A. Train-family-diversity curve: test macro-F1 vs number of training families.
  B. Cross-clade k-NN label transfer: do a novel genome's nearest training-family
     neighbours carry the label?

Reuses the 640-d ESM-2 features, the family split, and a LogisticRegression probe.
CPU-only; does not modify or fine-tune the encoder.

Usage:
    python cross_clade_diagnostic.py
    python cross_clade_diagnostic.py --out-dir paper/tables --fig-dir paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ood_error_analysis import binary_trait_labels

TRAITS = ["sporulation", "motility", "catalase"]
DIVERSITY_KS = [5, 10, 20, 40, 80]   # plus "all" appended at run time
DIVERSITY_SEEDS = [0, 1, 2]
K_NN = 10


def sample_families(families, k: int, seed: int) -> list:
    """Return k distinct families, drawn deterministically; all if k >= available."""
    uniq = sorted(set(families))
    if k >= len(uniq):
        return uniq
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(uniq), k, replace=False)
    return [uniq[i] for i in sorted(idx)]


def knn_majority_predict(X_ref, y_ref, X_query, k: int = K_NN) -> np.ndarray:
    """Predicted positive-probability = fraction of the k nearest X_ref that are positive."""
    X_ref = np.asarray(X_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    kk = min(k, len(X_ref))
    nn = NearestNeighbors(n_neighbors=kk).fit(X_ref)
    _, idx = nn.kneighbors(X_query)
    return y_ref[idx].mean(axis=1)
