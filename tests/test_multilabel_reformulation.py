"""Multi-label reformulation pure helpers: target collection and aggregation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import multilabel_reformulation as ml


def test_collect_metabolite_ternary():
    dicts = [
        {"indole": True, "acetoin": False, "h2s": None},
        {"indole": True, "acetoin": True, "h2s": None},
        {"indole": False, "acetoin": None},
    ]
    bids = ["a", "b", "c"]
    pos, neg = ml.collect_metabolite(dicts, bids)
    assert pos["indole"] == ["a", "b"]
    assert neg["indole"] == ["c"]
    assert pos["acetoin"] == ["b"]
    assert neg["acetoin"] == ["a"]
    assert "h2s" not in pos and "h2s" not in neg  # only None -> never recorded


def test_collect_metabolite_ignores_non_dict():
    pos, neg = ml.collect_metabolite([None, {"x": True}], ["a", "b"])
    assert pos["x"] == ["b"]


def test_collect_medium_presence():
    arrays = [np.array(["65", "9"]), np.array(["65"]), np.array(["92"])]
    bids = ["a", "b", "c"]
    pres, all_bids = ml.collect_medium(arrays, bids)
    assert pres["65"] == {"a", "b"}
    assert pres["9"] == {"a"}
    assert pres["92"] == {"c"}
    assert set(all_bids) == {"a", "b", "c"}


def test_aggregate_macro_micro():
    per_label = {
        "l1": {"auroc": 0.80, "f1": 0.6, "yte": np.array([1, 0, 1, 0]),
               "pred": np.array([1, 0, 1, 1])},
        "l2": {"auroc": 0.90, "f1": 0.8, "yte": np.array([1, 1, 0, 0]),
               "pred": np.array([1, 1, 0, 0])},
    }
    agg = ml.aggregate(per_label)
    assert agg["n_labels"] == 2
    assert abs(agg["macro_auroc"] - 0.85) < 1e-9
    assert abs(agg["macro_f1"] - 0.7) < 1e-9
    assert 0.0 <= agg["micro_f1"] <= 1.0


def test_aggregate_empty():
    assert ml.aggregate({})["n_labels"] == 0


def test_to_markdown_header_and_baseline():
    results = {"metabolite_production": {"per_label": {}, "n_labels": 0}}
    md = ml.to_markdown(results)
    assert "Table 25" in md
    assert "0.102" in md  # collapse baseline shown even when no labels
