"""Checkpoint retrieval blend (real-system cross-clade)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import checkpoint_retrieval as cr


def test_blend_alpha_one_is_model_alone():
    model_test = np.array([0.9, 0.1, 0.8, 0.2])
    knn_test = np.array([0.0, 1.0, 0.0, 1.0])  # adversarial
    out = cr.blend(model_test, knn_test, 1.0)
    assert np.allclose(out, model_test)


def test_evaluate_trait_perfect_model_high_f1():
    rng = np.random.default_rng(0)
    d = 6
    n = 40
    X_ref = rng.normal(size=(n, d))
    y_ref = (X_ref[:, 0] > 0).astype(int)
    X_val = rng.normal(size=(20, d))
    y_val = (X_val[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(20, d))
    y_test = (X_test[:, 0] > 0).astype(int)
    # model is near-perfect on val and test
    model_val = np.where(y_val == 1, 0.95, 0.05)
    model_test = np.where(y_test == 1, 0.95, 0.05)
    r = cr.evaluate_trait(model_val, y_val, X_val, model_test, y_test, X_test,
                          X_ref, y_ref)
    assert r["model_f1"] > 0.9
    assert 0.0 <= r["alpha_star"] <= 1.0
    assert r["n_ref"] == n and r["n_test"] == 20


def _write_inputs(tmp_path, n_per_family=20):
    """Build a synthetic features.npz + splits/traits/preds parquets.

    Family-disjoint train/val/test; trait label = sign of feature 0 so k-NN and a
    'model' that reads feature 0 both succeed -> alignment + plumbing are exercised.
    """
    rng = np.random.default_rng(1)
    rows = []
    feats = []
    bid = 0
    for split, fams in [("train", ["A", "B"]), ("val", ["C"]), ("test", ["D"])]:
        for f in fams:
            for _ in range(n_per_family):
                x = rng.normal(size=8)
                label = int(x[0] > 0)
                rows.append({"bacdive_id": bid, "family_split": split,
                             "family": f, "trait_label": label})
                feats.append(x)
                bid += 1
    feats = np.array(feats)
    ids = np.array([r["bacdive_id"] for r in rows])

    feat_path = tmp_path / "feats.npz"
    np.savez(feat_path, bacdive_ids=ids, features=feats)

    df = pd.DataFrame(rows)
    splits_path = tmp_path / "splits.parquet"
    df[["bacdive_id", "family_split"]].to_parquet(splits_path)
    traits_path = tmp_path / "traits.parquet"
    df.rename(columns={"trait_label": "motility"})[["bacdive_id", "motility"]].to_parquet(traits_path)

    # model predictions on val+test: a good model reading feature 0, shuffled order
    pred_rows = []
    for r in rows:
        if r["family_split"] in ("val", "test"):
            x0 = feats[r["bacdive_id"]][0]
            p = 1 / (1 + np.exp(-3 * x0))  # logistic of feature 0
            pred_rows.append({"bacdive_id": r["bacdive_id"], "split": r["family_split"],
                              "trait": "motility", "true_label": r["trait_label"], "pred": p})
    preds = pd.DataFrame(pred_rows).sample(frac=1.0, random_state=2).reset_index(drop=True)
    preds_path = tmp_path / "preds.parquet"
    preds.to_parquet(preds_path)
    return preds_path, feat_path, splits_path, traits_path


def test_run_end_to_end(tmp_path):
    preds_path, feat_path, splits_path, traits_path = _write_inputs(tmp_path)
    results = cr.run(preds_path, feat_path, splits_path, traits_path)
    assert "motility" in results
    r = results["motility"]
    # both the model and k-NN read the informative feature -> all should be strong
    assert r["model_f1"] > 0.7
    assert r["knn_f1"] > 0.6
    assert r["blend_f1"] >= min(r["model_f1"], r["knn_f1"]) - 1e-6
    assert "19" not in str(r)  # sanity: r is a metrics dict, not markdown


def test_run_alignment_is_by_id_not_order(tmp_path):
    """Shuffled preds must still align to embeddings by bacdive_id (not row order)."""
    preds_path, feat_path, splits_path, traits_path = _write_inputs(tmp_path)
    results = cr.run(preds_path, feat_path, splits_path, traits_path)
    # If alignment were positional, the shuffled preds would destroy model_f1.
    assert results["motility"]["model_f1"] > 0.7


def test_to_markdown_has_table_header(tmp_path):
    preds_path, feat_path, splits_path, traits_path = _write_inputs(tmp_path)
    md = cr.to_markdown(cr.run(preds_path, feat_path, splits_path, traits_path))
    assert "Table 19" in md
    assert "Blend F1" in md and "alpha*" in md
