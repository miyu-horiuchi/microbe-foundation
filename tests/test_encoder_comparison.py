"""Encoder-comparison pure helpers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import encoder_comparison as ec


def test_common_ids_intersection():
    assert ec.common_ids([["1", "2", "3"], ["2", "3", "4"], ["3", "2", "9"]]) == ["2", "3"]


def test_delta_row_signs():
    per_enc = {
        "esm2_150M": {"f1": 0.50, "auroc": 0.70, "n_test": 100, "pos_rate": 0.1},
        "bacformer": {"f1": 0.58, "auroc": 0.79, "n_test": 100, "pos_rate": 0.1},
    }
    r = ec.delta_row("motility", per_enc, "esm2_150M", "bacformer", "coverage-limited")
    assert abs(r["d_f1"] - 0.08) < 1e-9
    assert abs(r["d_auroc"] - 0.09) < 1e-9
    assert r["esm2_150M_f1"] == 0.50
    assert r["bacformer_auroc"] == 0.79


def test_parse_encoders_default_and_custom():
    assert ec.parse_encoders(None) == dict(ec.DEFAULT_ENCODERS)
    custom = ec.parse_encoders(["a=x.npz", "b=y.npz"])
    assert custom == {"a": "x.npz", "b": "y.npz"}
    assert list(custom.keys())[0] == "a"  # first = baseline


def test_to_markdown_header_and_mean():
    rows = [ec.delta_row("catalase", {
        "esm2_150M": {"f1": 0.9, "auroc": 0.95, "n_test": 50, "pos_rate": 0.5},
        "bacformer": {"f1": 0.92, "auroc": 0.96, "n_test": 50, "pos_rate": 0.5},
    }, "esm2_150M", "bacformer", "solved")]
    md = ec.to_markdown(rows, "esm2_150M", "bacformer", 1234)
    assert "Table 23" in md
    assert "1234 shared genomes" in md
    assert "Mean" in md
