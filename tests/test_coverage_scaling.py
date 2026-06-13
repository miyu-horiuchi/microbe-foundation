"""Coverage-scaling pure helpers: log2 slope and curve summary."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import coverage_scaling as cs


def test_log2_slope_linear_in_log2():
    # f1 = 0.1 * log2(k) + 0.2  -> slope should be 0.1
    ks = [5, 10, 20, 40, 80]
    means = [0.1 * np.log2(k) + 0.2 for k in ks]
    assert abs(cs.log2_slope(ks, means) - 0.1) < 1e-9


def test_log2_slope_flat_is_zero():
    assert abs(cs.log2_slope([5, 10, 20], [0.4, 0.4, 0.4])) < 1e-9


def test_log2_slope_degenerate_single_point():
    assert cs.log2_slope([10], [0.5]) == 0.0


def test_summarize_curve_endpoints_and_gain():
    rows = [
        {"k_families": 5, "seed": 0, "n_train": 100, "f1": 0.30, "auroc": 0.60},
        {"k_families": 5, "seed": 1, "n_train": 100, "f1": 0.34, "auroc": 0.62},
        {"k_families": 80, "seed": 0, "n_train": 100, "f1": 0.50, "auroc": 0.78},
        {"k_families": 80, "seed": 1, "n_train": 100, "f1": 0.54, "auroc": 0.80},
    ]
    s = cs.summarize_curve(rows)
    assert s["ks"] == [5, 80]
    assert abs(s["f1_lo"] - 0.32) < 1e-9
    assert abs(s["f1_hi"] - 0.52) < 1e-9
    assert abs(s["f1_gain"] - 0.20) < 1e-9
    assert abs(s["auroc_gain"] - 0.18) < 1e-9
    assert s["f1_slope_per_2x"] > 0


def test_to_markdown_header():
    out = {"motility": cs.summarize_curve([
        {"k_families": 5, "seed": 0, "n_train": 50, "f1": 0.3, "auroc": 0.6},
        {"k_families": 40, "seed": 0, "n_train": 50, "f1": 0.5, "auroc": 0.75},
    ])}
    out["motility"]["n_families_available"] = 120
    out["motility"]["rows"] = []
    out["motility"]["fixed"] = None
    out["motility"]["fixed_budget"] = 0
    md = cs.to_markdown(out)
    assert "Table 22" in md
    assert "F1 / 2x" in md
    assert "`motility`" in md
