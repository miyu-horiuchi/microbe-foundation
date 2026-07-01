"""Pure-helper tests for the baseline-regressor experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import baseline_regressors as br


def test_parse_ranks_supports_full_alias():
    assert br.parse_ranks("6, 10, full") == (6, 10, None)
    assert br.rank_label(None) == "full"
    assert br.rank_label(25) == "25"


def test_top_multilabel_targets_uses_observed_rows_as_mask():
    df = pd.DataFrame({
        "cultivation_medium": [
            ["LB", "M9"],
            ["LB"],
            ["M9"],
            None,
            [],
        ]
    })
    targets = br.top_multilabel_targets(
        df, "cultivation_medium", top_k=2, min_positive=1
    )

    names = [t[0] for t in targets]
    assert names == ["cultivation_medium:LB", "cultivation_medium:M9"]
    _, y_lb, mask_lb, count_lb = targets[0]
    assert count_lb == 2
    assert y_lb.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert mask_lb.tolist() == [True, True, True, False, False]


def test_collect_targets_adds_binary_and_media_targets():
    df = pd.DataFrame({
        "motility": [True, False, None, True],
        "cultivation_medium": [["LB"], ["M9"], ["LB", "M9"], None],
    })
    targets = br.collect_targets(
        df,
        ["motility"],
        top_media=1,
        min_positive=1,
    )

    assert [t.name for t in targets] == ["motility", "cultivation_medium:LB"]
    assert targets[0].group == "binary_trait"
    assert targets[0].mask.tolist() == [True, True, False, True]
    assert np.array_equal(targets[0].y, np.array([1.0, 0.0, 0.0, 1.0]))


def test_model_grid_caps_polynomial_to_low_rank():
    grid = br.model_grid((6, 10, 25, 50, None), max_poly_rank=25)
    assert ("chance", None) in grid
    assert ("logistic_l2", None) in grid
    assert ("poly2_logistic", 25) in grid
    assert ("poly2_logistic", 50) not in grid
    assert ("poly2_logistic", None) not in grid


def test_metrics_row_contains_rare_target_metrics():
    row = br.metrics_row(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert row["macro_f1"] == 1.0
    assert row["auroc"] == 1.0
    assert row["auprc"] == 1.0
