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
