"""Coverage panel summary-row flattening."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import coverage_panel as cp


def _fake_result(trait="motility"):
    return {
        "trait": trait,
        "test_pos_rate": 0.06,
        "auroc": 0.76,
        "macro_f1": 0.58,
        "auroc_minus_f1": 0.18,
        "positive_gini_over_families": 0.90,
        "top5_family_share_of_positives": 0.43,
        "recall_by_neighbor_positive_rate": [
            {"bin": "[0, 0.001)", "n_genomes": 1378, "n_positives": 37, "recall_on_positives": 0.108},
            {"bin": "[0.001, 0.1)", "n_genomes": 0, "n_positives": 0, "recall_on_positives": float("nan")},
            {"bin": "[0.1, 0.3)", "n_genomes": 523, "n_positives": 70, "recall_on_positives": 0.514},
            {"bin": "[0.3, 1.01)", "n_genomes": 257, "n_positives": 25, "recall_on_positives": 0.720},
        ],
    }


def test_summary_row_extracts_endpoints():
    row = cp.summary_row(_fake_result())
    assert row["trait"] == "motility"
    assert row["recall_no_coverage"] == 0.108
    assert row["n_pos_no_coverage"] == 37
    assert row["recall_high_coverage"] == 0.720
    assert row["n_pos_high_coverage"] == 25
    assert row["auroc"] == 0.76


def test_summary_row_nan_recall_becomes_none():
    r = _fake_result()
    r["recall_by_neighbor_positive_rate"][0]["recall_on_positives"] = float("nan")
    row = cp.summary_row(r)
    assert row["recall_no_coverage"] is None


def test_to_markdown_has_table21_header():
    md = cp.to_markdown([cp.summary_row(_fake_result())])
    assert "Table 21" in md
    assert "Recall (no cov)" in md
    assert "`motility`" in md
