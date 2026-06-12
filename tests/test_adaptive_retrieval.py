"""Novelty-adaptive retrieval head: pure helpers and synthetic end-to-end plumbing."""
from __future__ import annotations

import numpy as np

import adaptive_retrieval as mod


def test_normalize_novelty_monotone_and_bounded():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([0.5, 2.5, 5.0])
    t = mod.normalize_novelty(ref, q)
    assert t.shape == (3,)
    assert np.all((t >= 0.0) & (t <= 1.0))
    # least novel -> ~0, most novel -> 1, and monotone non-decreasing in the score
    assert t[0] == 0.0
    assert t[2] == 1.0
    assert t[0] <= t[1] <= t[2]


def test_adaptive_alpha_endpoints_and_clip():
    t = np.array([0.0, 0.5, 1.0])
    a = mod.adaptive_alpha(t, alpha_lo=0.2, alpha_hi=0.8)
    assert np.isclose(a[0], 0.2)
    assert np.isclose(a[1], 0.5)
    assert np.isclose(a[2], 0.8)
    # ramp that would exceed [0,1] is clipped
    a2 = mod.adaptive_alpha(np.array([0.0, 1.0]), alpha_lo=-0.5, alpha_hi=1.5)
    assert a2[0] == 0.0 and a2[1] == 1.0


def test_tune_ramp_trusts_probe_when_novel():
    # Two novelty regimes. Low-novelty (t=0): knn is correct, probe wrong.
    # High-novelty (t=1): probe is correct, knn wrong.
    # Optimal ramp should keep alpha low when t=0 and high when t=1 -> alpha_hi > alpha_lo.
    t = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    knn = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8])  # right when t=0, wrong when t=1
    probe = np.array([0.1, 0.2, 0.9, 0.8, 0.9, 0.8, 0.1, 0.2])  # wrong when t=0, right when t=1
    lo, hi = mod.tune_ramp(probe, knn, t, y)
    assert lo in mod.RAMP_GRID and hi in mod.RAMP_GRID
    assert hi > lo


def test_evaluate_adaptive_keys_and_ranges():
    rng = np.random.default_rng(0)

    def make(n):
        X = np.vstack([rng.normal(0, 0.4, (n, 5)), rng.normal(5, 0.4, (n, 5))])
        y = np.r_[np.zeros(n), np.ones(n)]
        return X, y

    Xtr, ytr = make(40)
    Xva, yva = make(15)
    Xte, yte = make(15)
    res = mod.evaluate_adaptive(Xtr, ytr, Xva, yva, Xte, yte, k=5)
    expected = {
        "alpha_lo", "alpha_hi", "alpha_global", "probe_f1", "probe_auroc",
        "global_f1", "global_auroc", "adaptive_f1", "adaptive_auroc",
        "delta_vs_global_f1", "delta_vs_probe_f1", "n_test", "pos_rate_test",
    }
    assert expected <= set(res)
    assert res["alpha_lo"] in mod.RAMP_GRID and res["alpha_hi"] in mod.RAMP_GRID
    for key in ("probe_f1", "global_f1", "adaptive_f1",
                "probe_auroc", "global_auroc", "adaptive_auroc"):
        assert 0.0 <= res[key] <= 1.0
    assert res["adaptive_auroc"] > 0.9
