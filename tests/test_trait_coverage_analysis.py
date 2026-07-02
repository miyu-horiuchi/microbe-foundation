"""Trait coverage analysis: gini, neighbour-positive-rate, recall stratification."""
from __future__ import annotations

import numpy as np

import trait_coverage_analysis as tca


def test_gini_uniform_is_zero():
    assert tca.gini([5, 5, 5, 5]) < 1e-9


def test_gini_concentrated_is_high():
    g = tca.gini([0, 0, 0, 100])
    assert g > 0.7


def test_neighbor_positive_rate_all_positive_neighbors():
    # train: positives near +5, negatives near -5; test point at +5 -> neighbours positive
    X_train = np.array([[5.0], [5.1], [4.9], [-5.0], [-5.1], [-4.9]])
    y_train = np.array([1, 1, 1, 0, 0, 0])
    X_test = np.array([[5.0], [-5.0]])
    npr = tca.neighbor_positive_rate(X_train, y_train, X_test, k=3)
    assert npr[0] == 1.0   # near the positive cluster
    assert npr[1] == 0.0   # near the negative cluster


def test_recall_by_bin_separates_coverage():
    # genomes with positive neighbours get predicted; those without don't
    npr = np.array([0.0, 0.0, 0.5, 0.5])
    y_test = np.array([1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.8])  # only high-npr ones cross 0.5
    rows = tca.recall_by_bin(npr, y_test, proba, bins=[0.0, 0.001, 0.1, 0.3, 1.01])
    by_bin = {r["bin"]: r for r in rows}
    # the [0, 0.001) bin holds the no-neighbour positives -> recall 0
    zero_bin = next(r for r in rows if r["n_positives"] == 2 and r["recall_on_positives"] == 0.0)
    assert zero_bin is not None
    # a higher bin holds covered positives -> recall 1
    assert any(r["recall_on_positives"] == 1.0 for r in rows)


def test_recall_by_bin_nan_when_no_positives():
    npr = np.array([0.0, 0.0])
    y_test = np.array([0, 0])
    proba = np.array([0.1, 0.2])
    rows = tca.recall_by_bin(npr, y_test, proba, bins=[0.0, 0.001, 1.01])
    r0 = rows[0]
    assert r0["n_positives"] == 0
    assert r0["recall_on_positives"] != r0["recall_on_positives"]  # NaN
