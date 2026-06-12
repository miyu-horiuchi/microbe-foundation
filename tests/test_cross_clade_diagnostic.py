"""Cross-clade diagnostic: pure helpers and synthetic end-to-end plumbing."""
from __future__ import annotations

import numpy as np

import cross_clade_diagnostic as mod


def test_sample_families_deterministic_distinct_subset():
    fams = ["a", "b", "c", "d", "e", "a", "b"]  # duplicates collapse
    out = mod.sample_families(fams, k=3, seed=0)
    assert len(out) == 3
    assert len(set(out)) == 3
    assert set(out) <= {"a", "b", "c", "d", "e"}
    assert mod.sample_families(fams, k=3, seed=0) == out  # deterministic


def test_sample_families_caps_at_available():
    fams = ["a", "b"]
    assert sorted(mod.sample_families(fams, k=10, seed=1)) == ["a", "b"]


def test_knn_majority_predict_all_positive_and_all_negative():
    X_ref = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    y_ref = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    # query near the positive cluster -> ~1.0; near the negative cluster -> ~0.0
    q = np.array([[0.05], [10.05]])
    probs = mod.knn_majority_predict(X_ref, y_ref, q, k=3)
    assert probs.shape == (2,)
    assert probs[0] == 1.0
    assert probs[1] == 0.0
