"""Failure-mode classification + run aggregation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import failure_mode_analysis as fm


def test_solved_when_family_high():
    assert fm.classify_failure_mode(0.95, 0.92) == "solved"
    assert fm.classify_failure_mode(0.72, 0.71) == "solved"


def test_label_ceiling_when_species_low():
    # poor even in-distribution -> not a generalization problem
    assert fm.classify_failure_mode(0.19, 0.14) == "label-ceiling"
    assert fm.classify_failure_mode(0.10, 0.08) == "label-ceiling"


def test_coverage_limited_on_big_decay():
    assert fm.classify_failure_mode(0.51, 0.19) == "coverage-limited"
    assert fm.classify_failure_mode(0.766, 0.589) == "coverage-limited"


def test_moderate_flat_small_decay():
    assert fm.classify_failure_mode(0.643, 0.637) == "moderate-flat"
    assert fm.classify_failure_mode(0.465, 0.440) == "moderate-flat"


def test_incomplete_when_missing():
    assert fm.classify_failure_mode(None, 0.5) == "incomplete"
    assert fm.classify_failure_mode(0.5, None) == "incomplete"


def test_quality_prefers_f1_then_acc_then_none():
    assert fm._quality({"metrics": {"acc": 0.9, "f1": 0.6}}) == 0.6
    assert fm._quality({"metrics": {"acc": 0.9}}) == 0.9
    assert fm._quality({"metrics": {"rmse": 0.1}}) is None


def _write(d, split, seed, per_head):
    payload = {"split_level": split, "per_head": per_head}
    (d / f"run-{split}-s{seed}.json").write_text(json.dumps(payload))


def test_load_split_means_averages_seeds(tmp_path):
    for seed, f1 in [(1, 0.80), (2, 0.84)]:
        _write(tmp_path, "species", seed,
               {"motility": {"metrics": {"acc": 0.9, "f1": f1}, "head_type": "binary"}})
    _write(tmp_path, "family", 1,
           {"motility": {"metrics": {"acc": 0.7, "f1": 0.50}, "head_type": "binary"}})
    means = fm.load_split_means(str(tmp_path), prefix="run")
    assert abs(means["species"]["motility"] - 0.82) < 1e-9
    assert abs(means["family"]["motility"] - 0.50) < 1e-9


def test_build_rows_classifies_and_orders(tmp_path):
    _write(tmp_path, "species", 1,
           {"motility": {"metrics": {"f1": 0.76}}, "catalase": {"metrics": {"f1": 0.95}}})
    _write(tmp_path, "family", 1,
           {"motility": {"metrics": {"f1": 0.55}}, "catalase": {"metrics": {"f1": 0.92}}})
    rows = fm.build_rows(fm.load_split_means(str(tmp_path), prefix="run"))
    modes = {r["trait"]: r["mode"] for r in rows}
    assert modes["catalase"] == "solved"
    assert modes["motility"] == "coverage-limited"
    # coverage-limited sorts before solved
    assert rows[0]["trait"] == "motility"
