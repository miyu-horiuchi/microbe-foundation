"""Seed aggregation: mean / std / 95% CI and grouping."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import aggregate_seeds as agg


def test_mean_std_ci_basic():
    s = agg.mean_std_ci([0.4, 0.5, 0.6])
    assert math.isclose(s["mean"], 0.5, abs_tol=1e-9)
    assert s["n"] == 3
    assert s["std"] > 0
    # CI half-width = t95(2) * std/sqrt(3); t95(2)=4.303
    expected = 4.303 * s["std"] / math.sqrt(3)
    assert math.isclose(s["ci95"], expected, rel_tol=1e-6)
    assert s["lo"] < s["mean"] < s["hi"]


def test_single_value_has_no_ci():
    s = agg.mean_std_ci([0.7])
    assert s["mean"] == 0.7
    assert s["std"] == 0.0
    assert math.isnan(s["ci95"])


def test_t95_falls_back_to_normal_for_large_n():
    assert agg.t95(500) == 1.96
    assert agg.t95(4) == 2.776


def _write_run(d: Path, name, pooling, split, balanced, traits):
    payload = {
        "run_name": name, "split_level": split, "pooling": pooling,
        "balanced_families": balanced,
        "per_head": {
            t: {"metric_kind": kind, "score": score}
            for t, (kind, score) in traits.items()
        },
    }
    (d / f"{name}.json").write_text(json.dumps(payload))


def test_collect_and_aggregate_groups_by_config(tmp_path):
    _write_run(tmp_path, "st_family_s0", "set_transformer", "family", False,
               {"aerobic": ("acc", 0.80)})
    _write_run(tmp_path, "st_family_s1", "set_transformer", "family", False,
               {"aerobic": ("acc", 0.84)})
    # different config (balanced) must not be merged with the above
    _write_run(tmp_path, "st_family_bal_s0", "set_transformer", "family", True,
               {"aerobic": ("acc", 0.88)})

    groups = agg.collect(tmp_path)
    key = ("set_transformer", "family", False, "aerobic", "acc")
    assert sorted(groups[key]) == [0.80, 0.84]
    assert groups[("set_transformer", "family", True, "aerobic", "acc")] == [0.88]

    rows = agg.aggregate(groups)
    row = next(r for r in rows if not r["balanced_families"])
    assert math.isclose(row["mean"], 0.82, abs_tol=1e-9)
    assert row["n"] == 2


def test_macro_excludes_rmse(tmp_path):
    _write_run(tmp_path, "st_family_s0", "set_transformer", "family", False,
               {"aerobic": ("acc", 0.80), "fame": ("rmse", 0.10)})
    rows = agg.aggregate(agg.collect(tmp_path))
    macro = agg.macro_rows(rows)
    assert len(macro) == 1
    # macro mean should equal the acc only (rmse excluded), = 0.80
    assert math.isclose(macro[0]["mean"], 0.80, abs_tol=1e-9)


def test_empty_dir_raises(tmp_path):
    try:
        agg.collect(tmp_path)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit on empty runs dir")
