"""Project paths and shared constants.

Adapted from microbe-model v0 to match microbe-foundation's flat layout
(no src/ directory). All paths resolve relative to the repo root regardless
of where Python is invoked from.

Set NCBI_API_KEY in your environment for 10x higher rate limits on the
NCBI Datasets API used by pipeline.py.
"""
from __future__ import annotations

import os
from pathlib import Path

# microbe_model/config.py -> microbe_model/ -> repo root
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

BACDIVE_DIR = DATA / "bacdive"
GENOME_DIR = DATA / "genomes"
FEATURE_DIR = DATA / "features"

for _d in (DATA, ARTIFACTS, BACDIVE_DIR, GENOME_DIR, FEATURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

NCBI_API_KEY = os.environ.get("NCBI_API_KEY")

# Kept for backward compatibility with microbe-model v0 scripts; microbe-foundation
# uses the 21-head schema in trait_schema.json instead.
PHENOTYPE_TARGETS = {
    "optimal_temperature_c": "regression",
    "optimal_ph": "regression",
    "oxygen_requirement": "classification",
    "salt_tolerance_pct": "regression",
}
