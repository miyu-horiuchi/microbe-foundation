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
