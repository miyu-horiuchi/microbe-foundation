"""Tests for the GPU cost estimator's pure model."""
import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "estimate_cost", Path(__file__).resolve().parent.parent / "scripts" / "estimate_cost.py"
)
estimate_cost = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(estimate_cost)
estimate = estimate_cost.estimate


def test_run_counts_full_matrix():
    e = estimate(19592, 39_000_000, n_poolings=3, n_splits=3, seeds=5, epochs=40)
    # 3*3*5 = 45 pooling/split/seed + 3*5 = 15 family-balanced
    assert e["n_runs_a"] == 45
    assert e["n_runs_b"] == 15
    assert e["n_runs"] == 60


def test_no_balanced_drops_section_b():
    e = estimate(100, 0, n_poolings=2, n_splits=2, seeds=3, epochs=1, balanced=False)
    assert e["n_runs_b"] == 0
    assert e["n_runs"] == 2 * 2 * 3


def test_extract_hours_scale_with_throughput():
    fast = estimate(10, 3_600_000, n_poolings=1, n_splits=1, seeds=1, epochs=1,
                    extract=True, throughput=1000.0)
    slow = estimate(10, 3_600_000, n_poolings=1, n_splits=1, seeds=1, epochs=1,
                    extract=True, throughput=500.0)
    assert math.isclose(fast["extract_h"], 1.0)        # 3.6M / 1000 / 3600
    assert math.isclose(slow["extract_h"], 2.0)        # half the throughput -> 2x time


def test_extract_skipped_when_flag_off():
    e = estimate(10, 9_999_999, n_poolings=1, n_splits=1, seeds=1, epochs=1, extract=False)
    assert e["extract_h"] == 0.0


def test_cost_is_hours_times_rate():
    e = estimate(1000, 0, n_poolings=1, n_splits=1, seeds=1, epochs=10,
                 sec_per_genome_epoch=0.01, train_frac=1.0, rate=2.0, balanced=False)
    # 1 run * 10 epochs * 1000 genomes * 0.01s / 3600 GPU-hr
    assert e["n_runs"] == 1
    assert math.isclose(e["total_h"], 10 * 1000 * 0.01 / 3600.0)
    assert math.isclose(e["total_cost"], e["total_h"] * 2.0)
