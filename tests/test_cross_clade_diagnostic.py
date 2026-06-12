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


def test_knn_transfer_separable_beats_chance():
    rng = np.random.default_rng(0)
    # two well-separated clusters, label = cluster
    X_train = np.vstack([rng.normal(0, 0.3, (60, 4)), rng.normal(6, 0.3, (60, 4))])
    y_train = np.r_[np.zeros(60), np.ones(60)]
    X_test = np.vstack([rng.normal(0, 0.3, (20, 4)), rng.normal(6, 0.3, (20, 4))])
    y_test = np.r_[np.zeros(20), np.ones(20)]
    res = mod.knn_transfer(X_train, y_train, X_test, y_test, k=5)
    assert set(res) >= {"knn_f1", "knn_auroc", "probe_f1", "probe_auroc", "chance_f1"}
    assert res["knn_auroc"] > 0.9
    assert res["knn_f1"] > res["chance_f1"]


def test_diversity_curve_returns_row_per_k_seed():
    rng = np.random.default_rng(1)
    n_fam = 12
    fams = np.repeat([f"fam{i}" for i in range(n_fam)], 10)
    X_train = rng.normal(size=(len(fams), 4))
    y_train = (X_train[:, 0] > 0).astype(float)
    X_test = rng.normal(size=(40, 4))
    y_test = (X_test[:, 0] > 0).astype(float)
    rows = mod.family_diversity_curve(X_train, y_train, list(fams), X_test, y_test,
                                      ks=[3, 6], seeds=[0, 1])
    # 2 ks x 2 seeds = 4 rows (no degenerate single-class subsets here)
    assert len(rows) == 4
    for r in rows:
        assert r["k_families"] in (3, 6)
        assert 0.0 <= r["test_f1"] <= 1.0
        assert r["n_train"] > 0


def test_verdict_labels_match_signals():
    # diversity rising + knn good -> coverage; flat + poor -> wall
    assert mod.verdict(diversity_rising=True, knn_good=True) == "coverage-limited"
    assert mod.verdict(diversity_rising=False, knn_good=False) == "representation-wall"
    assert mod.verdict(diversity_rising=True, knn_good=False) == "mixed"


def test_is_rising_detects_monotone_gain():
    # mean test_f1 grows with k -> rising
    rows = [{"k_families": 5, "test_f1": 0.4}, {"k_families": 5, "test_f1": 0.42},
            {"k_families": 80, "test_f1": 0.6}, {"k_families": 80, "test_f1": 0.62}]
    assert mod.is_rising(rows) is True
    flat = [{"k_families": 5, "test_f1": 0.5}, {"k_families": 80, "test_f1": 0.5}]
    assert mod.is_rising(flat) is False
