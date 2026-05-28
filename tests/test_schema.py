"""Schema invariants: the 21-head structure must stay stable across edits."""
from __future__ import annotations

import json
from pathlib import Path

import schema as schema_mod

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_JSON = ROOT / "trait_schema.json"


def test_schema_module_has_21_traits():
    assert len(schema_mod.V1_TRAITS) == 21


def test_schema_has_all_seven_blocks():
    blocks = {t.block.value for t in schema_mod.V1_TRAITS}
    expected = {"morphology", "physiology", "growth", "cultivation", "safety", "ecology", "chemotaxonomy"}
    assert blocks == expected


def test_fatty_acid_head_exists():
    """The chemotaxonomy white-space differentiator must not be accidentally dropped."""
    names = {t.name for t in schema_mod.V1_TRAITS}
    assert "fatty_acid_profile" in names


def test_pathogenicity_heads_exist():
    """Both pathogenicity heads must exist for the safety block."""
    names = {t.name for t in schema_mod.V1_TRAITS}
    assert "pathogenicity_human" in names
    assert "pathogenicity_animal" in names


def test_unique_names():
    names = [t.name for t in schema_mod.V1_TRAITS]
    assert len(names) == len(set(names)), "Trait names must be unique"


def test_exported_json_matches_module():
    """The on-disk JSON must match the module — if not, run `python schema.py`."""
    if not SCHEMA_JSON.exists():
        return  # nothing to compare
    data = json.loads(SCHEMA_JSON.read_text())
    assert data["n_traits"] == len(schema_mod.V1_TRAITS), \
        "trait_schema.json is stale; rerun `python schema.py`"
    on_disk = {t["name"] for t in data["traits"]}
    in_module = {t.name for t in schema_mod.V1_TRAITS}
    assert on_disk == in_module


def test_head_types_are_known():
    valid = {"binary", "multiclass", "multilabel", "regression_vector"}
    for t in schema_mod.V1_TRAITS:
        assert t.head.value in valid


def test_multiclass_traits_have_classes():
    for t in schema_mod.V1_TRAITS:
        if t.head.value == "multiclass":
            # Either inline classes or a vocab-derived source
            assert t.classes or t.classes_source.value != "enumerated", \
                f"multiclass trait {t.name} lacks classes and classes_source"


def test_multilabel_traits_have_size():
    for t in schema_mod.V1_TRAITS:
        if t.head.value in ("multilabel", "regression_vector"):
            assert t.n_outputs is not None, \
                f"{t.name} ({t.head.value}) must declare n_outputs"
