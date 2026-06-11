"""OOD->error analysis: label materialization and per-trait evaluation plumbing."""
from __future__ import annotations

import numpy as np
import pandas as pd

import ood_error_analysis as mod


def test_binary_trait_labels_handles_nulls_and_truthiness():
    col = pd.Series([True, False, None, 1.0, 0.0])
    labels, mask = mod.binary_trait_labels(col)
    assert list(mask) == [True, True, False, True, True]
    # null row's label is masked out; truthy/falsey mapped to 1.0/0.0
    assert list(labels) == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_evaluate_trait_returns_metrics_with_sane_ranges():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(60, 4))
    y_train = (X_train[:, 0] > 0).astype(float)
    X_test = rng.normal(size=(40, 4))
    y_test = (X_test[:, 0] > 0).astype(float)
    result = mod.evaluate_trait(X_train, y_train, X_test, y_test, X_train, k=5)
    assert set(result) == {"n_test", "pos_rate", "auroc", "spearman_ood_error", "p_value"}
    assert result["n_test"] == 40
    assert 0.0 <= result["auroc"] <= 1.0
    assert -1.0 <= result["spearman_ood_error"] <= 1.0


def test_evaluate_trait_returns_none_for_single_class():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(30, 3))
    y_train = np.zeros(30)  # only one class
    X_test = rng.normal(size=(10, 3))
    y_test = np.array([0.0, 1.0] * 5)
    assert mod.evaluate_trait(X_train, y_train, X_test, y_test, X_train) is None
