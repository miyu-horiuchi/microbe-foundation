"""PU-correction helpers: Elkan-Noto conditional, threshold tuning, PU metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import cultivation_medium_pu as pu


def test_en_conditional_c_one_gives_zero():
    # If labeling is complete (c=1), no unlabeled example is a hidden positive.
    q = pu.en_conditional_pos(np.array([0.2, 0.5, 0.9]), 1.0)
    assert np.allclose(q, 0.0)


def test_en_conditional_monotonic_and_bounded():
    # c=0.8, modest scores stay below the [0,1] clip so monotonicity is visible.
    q = pu.en_conditional_pos(np.array([0.1, 0.3, 0.6]), 0.8)
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] < q[1] < q[2]  # higher score -> higher hidden-positive posterior


def test_f1_from_pr():
    assert pu.f1_from_pr(0.0, 0.0) == 0.0
    assert abs(pu.f1_from_pr(0.5, 0.5) - 0.5) < 1e-9
    assert abs(pu.f1_from_pr(1.0, 1.0) - 1.0) < 1e-9


def test_naive_prf_perfect():
    s = np.array([1, 1, 0, 0])
    pred = np.array([True, True, False, False])
    prec, rec, f1 = pu.naive_prf(s, pred)
    assert (prec, rec, f1) == (1.0, 1.0, 1.0)


def test_tune_threshold_recovers_separating_value():
    s = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t = pu.tune_threshold(s, p)
    # any threshold in (0.2, 0.8] gives perfect F1; tuner must pick one of them
    assert 0.2 < t <= 0.8


def test_pu_raises_f1_when_unlabeled_are_hidden_positives():
    # Confident model: unlabeled high-score examples are likely hidden positives.
    s = np.array([1, 1, 0, 0, 0])
    p = np.array([0.95, 0.9, 0.85, 0.05, 0.04])
    pred = p >= 0.5  # predicts the 0.85 unlabeled as positive -> naive counts it FP
    c = 0.5
    _, _, f1_naive = pu.naive_prf(s, pred)
    _, _, f1_pu = pu.pu_prf(s, p, pred, c)
    assert f1_pu > f1_naive  # PU correction credits the likely-hidden positive
