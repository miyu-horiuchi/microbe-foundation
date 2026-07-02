"""Label-quality audit pure helpers: entropy, audit metrics, diagnosis, cleaners."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import label_quality_audit as lq


def test_normalized_entropy_uniform_is_one():
    assert abs(lq.normalized_entropy([10, 10, 10, 10]) - 1.0) < 1e-9


def test_normalized_entropy_degenerate_is_zero():
    assert lq.normalized_entropy([42]) == 0.0
    assert lq.normalized_entropy([5, 0, 0]) == 0.0


def test_audit_metrics_categorical():
    s = pd.Series(["a", "a", "a", "a", "b", "c"])  # top1=4/6
    m = lq.audit_metrics(s)
    assert m["structured"] is False
    assert m["n_classes"] == 3
    assert abs(m["top1_share"] - 4 / 6) < 1e-9
    assert abs(m["singleton_frac"] - 2 / 3) < 1e-9  # b and c are singletons


def test_audit_metrics_structured():
    s = pd.Series([{"x": None}, {"y": "R"}])
    m = lq.audit_metrics(s)
    assert m["structured"] is True
    assert m["n_classes"] is None


def test_diagnose_paths():
    assert "structured" in lq.diagnose({"structured": True})
    assert "free-text" in lq.diagnose({"structured": False, "n_classes": 300,
                                       "n_labeled": 1000, "singleton_frac": 0.9, "top1_share": 0.1})
    assert "imbalance" in lq.diagnose({"structured": False, "n_classes": 3,
                                       "n_labeled": 1000, "singleton_frac": 0.0, "top1_share": 0.9})
    assert "geography" in lq.diagnose({"structured": False, "n_classes": 246,
                                       "n_labeled": 53065, "singleton_frac": 0.14, "top1_share": 0.20})


def test_canon_isolation_priority():
    assert lq.canon_isolation("Human blood") == "human/clinical"
    assert lq.canon_isolation("soil with plant residues") == "plant"  # plant rule precedes soil
    assert lq.canon_isolation("Soil") == "soil"
    assert lq.canon_isolation("marine sediment") == "sediment"  # sediment before water
    assert lq.canon_isolation("deep sea water") == "water/marine"
    assert lq.canon_isolation("Cheese") == "food/dairy"
    assert lq.canon_isolation("unknown xyz") == "other"


def test_clean_labels_oxygen_collapses_to_three():
    s = pd.Series(["obligate_aerobe", "facultative_anaerobe", "obligate_anaerobe",
                   "microaerophile", "aerotolerant", None])
    out = lq.clean_labels("oxygen_tolerance", s)
    assert set(out.dropna().unique()) == {"aerobe", "anaerobe", "facultative"}
    assert pd.isna(out.iloc[-1])
