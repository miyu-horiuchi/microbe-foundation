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


def _probe(X_train, y_train, X_test):
    """Standardize on train, fit a balanced LogisticRegression, return P(test=1)."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train), y_train)
    return clf.predict_proba(scaler.transform(X_test))[:, 1]


def _macro_f1(y_true, prob) -> float:
    return float(f1_score(y_true, (np.asarray(prob) > 0.5).astype(int), average="macro", zero_division=0))


def _chance_f1(y_true) -> float:
    """Macro-F1 of always predicting the training-majority class."""
    majority = 1 if np.mean(y_true) >= 0.5 else 0
    pred = np.full(len(y_true), majority)
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def knn_transfer(X_train, y_train, X_test, y_test, k: int = K_NN) -> dict:
    """Cross-clade k-NN label transfer vs the trained probe and a chance baseline.

    Standardizes embedding space on the training genomes; neighbours are drawn only
    from training-family genomes (no leakage).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    scaler = StandardScaler().fit(X_train)
    knn_prob = knn_majority_predict(scaler.transform(X_train), y_train, scaler.transform(X_test), k=k)
    probe_prob = _probe(X_train, y_train, X_test)
    return {
        "knn_f1": _macro_f1(y_test, knn_prob),
        "knn_auroc": float(roc_auc_score(y_test, knn_prob)),
        "probe_f1": _macro_f1(y_test, probe_prob),
        "probe_auroc": float(roc_auc_score(y_test, probe_prob)),
        "chance_f1": _chance_f1(y_test),
        "n_test": int(len(y_test)),
        "pos_rate": float(y_test.mean()),
    }


def family_diversity_curve(X_train, y_train, fam_train, X_test, y_test, ks, seeds) -> list:
    """Test macro-F1 as a function of the number of distinct training families.

    For each (k, seed): sample k families, restrict training to genomes in them,
    train a probe, evaluate on the fixed test set. Single-class subsets are skipped
    (with a printed note).
    """
    fam_train = np.asarray(fam_train)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    rows = []
    for k in ks:
        for seed in seeds:
            fams = sample_families(list(fam_train), k, seed)
            sel = np.isin(fam_train, list(fams))
            ytr = y_train[sel]
            if len(np.unique(ytr)) < 2:
                print(f"  skip k={k} seed={seed}: single-class subset")
                continue
            prob = _probe(X_train[sel], ytr, X_test)
            rows.append({
                "k_families": int(k),
                "seed": int(seed),
                "n_train": int(sel.sum()),
                "test_f1": _macro_f1(y_test, prob),
            })
    return rows
