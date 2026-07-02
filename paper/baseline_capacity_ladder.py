#!/usr/bin/env python3
"""Capacity-ladder plateau/escalation diagnostic, extra-data fusion, and
learned stacking on frozen genome embeddings.

Companion to paper/baseline_regressors.py (Table 32). Reuses its loaders,
targets, estimators, and metrics; adds Tables 34 (capacity ladder), 35
(fusion), and 36 (stacking). CPU-only, single seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_regressors as br  # noqa: E402

TAXONOMY_COLS = ["phylum", "class", "order"]
ISOLATION_COLS = ["isolation_source", "country"]


def build_onehot(train_series: pd.Series, top_k: int | None = None) -> list[str]:
    vals = train_series.dropna().astype(str)
    counts = vals.value_counts()
    cats = list(counts.index) if top_k is None else list(counts.index[:top_k])
    return sorted(cats) if top_k is None else list(cats)


def apply_onehot(series: pd.Series, categories: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((len(series), len(categories)), dtype=np.float32)
    for row, v in enumerate(series.tolist()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        j = index.get(str(v))
        if j is not None:
            mat[row, j] = 1.0
    return mat


LADDER: list[tuple[str, int | None]] = [
    ("logistic_l2", 6),
    ("logistic_l2", 10),
    ("logistic_l2", 25),
    ("logistic_l2", 50),
    ("logistic_l2", 100),
    ("logistic_l2", None),
    ("regularized_rf", None),
    ("hist_gd", None),
]


def detect_plateau(metrics: list[float], eps: float = 0.005) -> dict:
    lower_bound = metrics[0]
    running = metrics[0]
    flat = 0
    for i in range(1, len(metrics)):
        gain = metrics[i] - running
        running = max(running, metrics[i])
        if gain < eps:
            flat += 1
            if flat >= 2:
                return {
                    "lower_bound": lower_bound,
                    "plateau_index": i,
                    "plateau_value": metrics[i],
                    "plateaued": True,
                }
        else:
            flat = 0
    return {
        "lower_bound": lower_bound,
        "plateau_index": len(metrics) - 1,
        "plateau_value": metrics[-1],
        "plateaued": False,
    }


def escalation_verdict(plateau_value: float, ceiling: float, plateaued: bool,
                       margin: float = 0.02) -> str:
    if not plateaued:
        return "keep-scaling-simple"
    if ceiling is None or np.isnan(ceiling):
        return "unknown"
    return "NO — coverage-limited" if (ceiling - plateau_value) <= margin else "YES — capacity-limited"
