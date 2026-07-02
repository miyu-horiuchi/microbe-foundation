"""Tests for the capacity-ladder / fusion / stacking module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import baseline_capacity_ladder as cl


def test_apply_onehot_maps_unseen_and_missing_to_zero_row():
    train = pd.Series(["a", "b", "a", "b"])
    cats = cl.build_onehot(train)
    assert cats == ["a", "b"]
    test = pd.Series(["a", "c", None])
    mat = cl.apply_onehot(test, cats)
    assert mat.shape == (3, 2)
    assert mat[0].tolist() == [1.0, 0.0]   # known "a"
    assert mat[1].tolist() == [0.0, 0.0]   # unseen "c" -> zero row
    assert mat[2].tolist() == [0.0, 0.0]   # missing -> zero row


def test_build_onehot_top_k_keeps_most_frequent():
    train = pd.Series(["a", "a", "a", "b", "b", "c"])
    assert cl.build_onehot(train, top_k=2) == ["a", "b"]


def test_detect_plateau_finds_first_flat_run():
    # rises then flattens: gains 0.05, 0.05, 0.003, 0.001, 0.000, 0.000
    metrics = [0.70, 0.75, 0.80, 0.803, 0.804, 0.804, 0.804, 0.804]
    out = cl.detect_plateau(metrics, eps=0.005)
    assert out["lower_bound"] == 0.70
    assert out["plateaued"] is True
    assert out["plateau_index"] == 4      # first rung of the 2-in-a-row flat run
    assert abs(out["plateau_value"] - 0.804) < 1e-9


def test_detect_plateau_reports_not_plateaued_when_still_climbing():
    metrics = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    out = cl.detect_plateau(metrics, eps=0.005)
    assert out["plateaued"] is False
    assert out["plateau_index"] == len(metrics) - 1


def test_escalation_verdict_three_cases_plus_unknown():
    assert cl.escalation_verdict(0.80, 0.95, plateaued=False) == "keep-scaling-simple"
    assert cl.escalation_verdict(0.80, 0.81, plateaued=True) == "NO — coverage-limited"
    assert cl.escalation_verdict(0.80, 0.95, plateaued=True) == "YES — capacity-limited"
    assert cl.escalation_verdict(0.80, float("nan"), plateaued=True) == "unknown"


def test_load_ceilings_reads_knn_auroc(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text("trait,knn_auroc,probe_auroc\nmotility,0.695,0.728\nsporulation,0.924,0.935\n")
    c = cl.load_ceilings(csv)
    assert abs(c["sporulation"] - 0.924) < 1e-9
    assert abs(c["motility"] - 0.695) < 1e-9


def test_build_extra_matrix_taxonomy_is_train_fit():
    sub = pd.DataFrame({
        "phylum": ["P1", "P1", "P2", "P3"],
        "class":  ["C1", "C1", "C2", "C3"],
        "order":  ["O1", "O1", "O2", "O3"],
    })
    train_mask = np.array([True, True, False, False])
    mat = cl.build_extra_matrix(sub, "taxonomy", train_mask)
    # only P1/C1/O1 seen in train -> 3 columns; unseen test rows are all-zero
    assert mat.shape == (4, 3)
    assert mat[0].tolist() == [1.0, 1.0, 1.0]
    assert mat[2].tolist() == [0.0, 0.0, 0.0]
    assert mat[3].tolist() == [0.0, 0.0, 0.0]
