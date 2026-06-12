"""Retrieval-augmented head: pure helpers and synthetic end-to-end plumbing."""
from __future__ import annotations

import numpy as np
import pandas as pd

import retrieval_head as mod


def test_blend_endpoints_and_midpoint():
    probe = np.array([0.9, 0.2, 0.7])
    knn = np.array([0.1, 0.8, 0.3])
    assert np.allclose(mod.blend(probe, knn, 1.0), probe)
    assert np.allclose(mod.blend(probe, knn, 0.0), knn)
    assert np.allclose(mod.blend(probe, knn, 0.5), (probe + knn) / 2)


def test_tune_alpha_prefers_knn_when_knn_better():
    # y matches knn exactly, probe is anti-correlated -> low alpha wins.
    y_val = np.array([1, 1, 0, 0])
    knn_val = np.array([0.9, 0.8, 0.1, 0.2])
    probe_val = np.array([0.1, 0.2, 0.9, 0.8])
    alpha = mod.tune_alpha(probe_val, knn_val, y_val)
    assert alpha in mod.ALPHA_GRID
    assert alpha < 0.5


def test_tune_alpha_prefers_probe_when_knn_harmful():
    # probe is correct, knn is anti-correlated -> mixing knn in flips predictions,
    # so only a high alpha keeps the blend correct at the 0.5 threshold.
    y_val = np.array([1, 1, 0, 0])
    probe_val = np.array([0.95, 0.85, 0.05, 0.15])
    knn_val = np.array([0.05, 0.15, 0.95, 0.85])
    alpha = mod.tune_alpha(probe_val, knn_val, y_val)
    assert alpha > 0.5


def test_prepare_trait_tvt_shapes_and_counts():
    feats = np.arange(20 * 3, dtype=float).reshape(20, 3)
    tr = pd.DataFrame(
        {
            "trait": [True, False] * 8 + [None, None, True, False],
            "fsplit": (["train"] * 8 + ["val"] * 4 + ["test"] * 4 + ["train"] * 4),
            "row": list(range(20)),
        }
    )
    Xtr, ytr, Xva, yva, Xte, yte = mod.prepare_trait_tvt(feats, tr, "trait")
    # 2 of the last 4 rows are unlabeled (None) -> dropped by the mask.
    assert Xtr.shape[1] == 3
    assert len(ytr) == Xtr.shape[0]
    assert len(yva) == Xva.shape[0] == 4
    assert len(yte) == Xte.shape[0] == 4
    assert set(np.unique(ytr)) <= {0.0, 1.0}


def test_evaluate_retrieval_keys_and_ranges():
    rng = np.random.default_rng(0)
    # separable clusters so probe and knn both work; label = cluster
    def make(n):
        X = np.vstack([rng.normal(0, 0.4, (n, 5)), rng.normal(5, 0.4, (n, 5))])
        y = np.r_[np.zeros(n), np.ones(n)]
        return X, y

    Xtr, ytr = make(40)
    Xva, yva = make(15)
    Xte, yte = make(15)
    res = mod.evaluate_retrieval(Xtr, ytr, Xva, yva, Xte, yte, k=5)
    expected = {
        "alpha_star", "probe_f1", "probe_auroc", "knn_f1", "knn_auroc",
        "blend_f1", "blend_auroc", "delta_f1", "n_test", "pos_rate_test",
    }
    assert expected <= set(res)
    assert res["alpha_star"] in mod.ALPHA_GRID
    for key in ("probe_f1", "knn_f1", "blend_f1", "probe_auroc", "knn_auroc", "blend_auroc"):
        assert 0.0 <= res[key] <= 1.0
    # On a cleanly separable problem the blend should be strong.
    assert res["blend_auroc"] > 0.9
    # Blend tuned on val cannot do worse than probe-alone by construction of the grid
    # (alpha=1 is always a candidate), modulo val/test gap; allow a small slack.
    assert res["delta_f1"] >= -0.1
