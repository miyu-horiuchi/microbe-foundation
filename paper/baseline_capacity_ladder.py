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


def load_ceilings(path: str | Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(t): float(v) for t, v in zip(df["trait"], df["knn_auroc"])}


def _train_test_arrays(feats, df, target, seed):
    sub = df[target.mask].copy()
    y = target.y[target.mask]
    is_train = (sub["fsplit"] == "train").to_numpy()
    is_test = (sub["fsplit"] == "test").to_numpy()
    if is_train.sum() == 0 or is_test.sum() == 0:
        return None
    if len(np.unique(y[is_train])) < 2 or len(np.unique(y[is_test])) < 2:
        return None
    return (
        sub,
        feats[sub["row"].to_numpy()[is_train]], y[is_train].astype(int),
        feats[sub["row"].to_numpy()[is_test]], y[is_test].astype(int),
    )


def _score(x_train, y_train, x_test, y_test, model, rank, seed):
    est = br.build_estimator(model, rank, x_train, seed)
    prob = br.predict_probability(br.clone(est), x_train, y_train, x_test)
    row = br.metrics_row(y_test, prob)
    row["test_pos_rate"] = float(y_test.mean())
    metric = br.primary_metric(row)
    return metric, float(row[metric])


def run_capacity_ladder(feats, df, targets, ceilings, seed):
    rung_rows, summary_rows = [], []
    for target in targets:
        arr = _train_test_arrays(feats, df, target, seed)
        if arr is None:
            continue
        _, x_train, y_train, x_test, y_test = arr
        metrics, metric_name = [], None
        for step, (model, rank) in enumerate(LADDER):
            metric_name, value = _score(x_train, y_train, x_test, y_test, model, rank, seed)
            metrics.append(value)
            rung_rows.append({
                "target": target.name, "group": target.group, "step": step,
                "model": model, "rank": br.rank_label(rank),
                "primary_metric": metric_name, "score": value,
            })
        pl = detect_plateau(metrics)
        ceiling = ceilings.get(target.name, float("nan"))
        verdict = escalation_verdict(pl["plateau_value"], ceiling, pl["plateaued"])
        summary_rows.append({
            "target": target.name, "group": target.group,
            "primary_metric": metric_name,
            "lower_bound": pl["lower_bound"],
            "plateau_step": pl["plateau_index"],
            "plateau_value": pl["plateau_value"],
            "plateaued": pl["plateaued"],
            "ceiling_knn_auroc": ceiling,
            "verdict": verdict,
        })
    return rung_rows, summary_rows
