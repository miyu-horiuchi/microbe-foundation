#!/usr/bin/env python3
"""Cosine-similarity diagnostic for family-split collapse.

Question
--------
When a trait predictor fails on held-out bacterial families, is the test family
geometrically unsupported by the training families in frozen ESM-2 space?

This script adds a cosine-geometry view to the existing cross-clade diagnostic:

1. Normalize frozen genome embeddings in a train-fitted standardized space.
2. For each held-out test genome, compute:
   * nearest training-family centroid cosine similarity
   * top-k cosine-neighbor similarity to labeled training genomes
   * cosine kNN label-transfer probability
   * balanced logistic-probe probability
3. Ask whether low cosine coverage predicts larger per-genome error and whether
   low-similarity test families have worse label transfer.

Interpretation
--------------
If cosine kNN works and error rises as nearest-family cosine falls, collapse is
mostly coverage/geometry-limited: held-out families lack close labelled training
neighbours. If kNN fails even at high cosine similarity, the label is not local in
the frozen embedding or the labels are too noisy/sparse.

Usage:
    python paper/cosine_family_collapse.py
    python paper/cosine_family_collapse.py --traits sporulation motility catalase
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ood_error_analysis import DEFAULT_TRAITS, binary_trait_labels  # noqa: E402

K = 10


def safe_auroc(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, prob))
    except ValueError:
        return float("nan")


def macro_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    pred = (np.asarray(prob) >= 0.5).astype(int)
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def load_aligned(features_path: str, splits_path: str, traits_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(features_path, allow_pickle=False)
    feats = np.asarray(data["features"], dtype=np.float32)
    ids = np.array([str(i) for i in data["bacdive_ids"]])
    id_to_row = {b: i for i, b in enumerate(ids)}

    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    tr = tr[tr["bid"].isin(id_to_row)].copy()
    tr["row"] = tr["bid"].map(id_to_row).astype(int)

    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    split_by_id = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    tr["fsplit"] = tr["bid"].map(lambda b: split_by_id.get(b, "unknown"))
    if "family" not in tr.columns:
        raise SystemExit("traits parquet must include a `family` column for this diagnostic")
    return feats, tr


def standardize_then_normalize(feats: np.ndarray, tr: pd.DataFrame) -> np.ndarray:
    """Fit scale on all family-train genomes, then L2-normalize for cosine."""
    train_rows = tr.loc[tr["fsplit"] == "train", "row"].to_numpy()
    scaler = StandardScaler().fit(feats[train_rows])
    return normalize_rows(scaler.transform(feats))


def centroid_matrix(x_norm: np.ndarray, families: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fams = np.array(sorted(set(str(f) for f in families)))
    centroids = []
    fam_arr = np.asarray([str(f) for f in families])
    for fam in fams:
        centroids.append(x_norm[fam_arr == fam].mean(axis=0))
    return fams, normalize_rows(np.vstack(centroids))


def nearest_centroid_cosine(
    x_query_norm: np.ndarray,
    train_centroids_norm: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Max cosine similarity to any training-family centroid."""
    out = []
    for start in range(0, len(x_query_norm), chunk_size):
        sims = x_query_norm[start:start + chunk_size] @ train_centroids_norm.T
        out.append(sims.max(axis=1))
    return np.concatenate(out)


def cosine_knn_transfer(
    x_train_norm: np.ndarray,
    y_train: np.ndarray,
    x_test_norm: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (label probability, top-1 cosine, mean top-k cosine)."""
    kk = min(k, len(x_train_norm))
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine", algorithm="brute")
    nn.fit(x_train_norm)
    dist, idx = nn.kneighbors(x_test_norm)
    sim = 1.0 - dist
    prob = y_train[idx].mean(axis=1)
    return prob, sim[:, 0], sim.mean(axis=1)


def logistic_probe_prob(x_train_norm, y_train, x_test_norm) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train_norm, y_train)
    return clf.predict_proba(x_test_norm)[:, 1]


def bin_rows(y_test, knn_prob, probe_prob, centroid_cos, n_bins: int = 3) -> list[dict]:
    """Performance by cosine-support bin."""
    qs = np.quantile(centroid_cos, np.linspace(0, 1, n_bins + 1))
    rows = []
    names = ["low", "mid", "high"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    for i, name in enumerate(names):
        lo, hi = qs[i], qs[i + 1]
        if i == 0:
            mask = centroid_cos <= hi
        elif i == n_bins - 1:
            mask = centroid_cos >= lo
        else:
            mask = (centroid_cos > lo) & (centroid_cos < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": name,
            "n": int(mask.sum()),
            "mean_centroid_cos": float(np.mean(centroid_cos[mask])),
            "pos_rate": float(np.mean(y_test[mask])),
            "knn_f1": macro_f1(y_test[mask], knn_prob[mask]),
            "knn_auroc": safe_auroc(y_test[mask], knn_prob[mask]),
            "probe_f1": macro_f1(y_test[mask], probe_prob[mask]),
            "probe_auroc": safe_auroc(y_test[mask], probe_prob[mask]),
        })
    return rows


def family_rows(test_df, y_test, knn_prob, probe_prob, centroid_cos, top1_cos) -> list[dict]:
    rows = []
    tmp = test_df[["family"]].copy()
    tmp["y"] = y_test
    tmp["knn_prob"] = knn_prob
    tmp["probe_prob"] = probe_prob
    tmp["centroid_cos"] = centroid_cos
    tmp["top1_cos"] = top1_cos
    tmp["probe_abs_error"] = np.abs(y_test - probe_prob)
    tmp["knn_abs_error"] = np.abs(y_test - knn_prob)
    for fam, g in tmp.groupby("family"):
        rows.append({
            "family": str(fam),
            "n": int(len(g)),
            "pos_rate": float(g["y"].mean()),
            "mean_centroid_cos": float(g["centroid_cos"].mean()),
            "mean_top1_cos": float(g["top1_cos"].mean()),
            "probe_abs_error": float(g["probe_abs_error"].mean()),
            "knn_abs_error": float(g["knn_abs_error"].mean()),
        })
    return rows


def geometry_verdict(knn_auroc: float, rho_centroid_error: float) -> str:
    if knn_auroc >= 0.75 and rho_centroid_error < -0.05:
        return "coverage/geometry-limited"
    if knn_auroc < 0.60:
        return "label not local in cosine space"
    if rho_centroid_error < -0.05:
        return "mixed: cosine support matters"
    return "mixed/weak cosine-error link"


def evaluate_trait(
    trait: str,
    x_norm: np.ndarray,
    tr: pd.DataFrame,
    k: int = K,
) -> tuple[dict | None, list[dict], list[dict]]:
    if trait not in tr.columns:
        return None, [], []
    y_all, mask = binary_trait_labels(tr[trait])
    sub = tr[mask].copy()
    y = y_all[mask].astype(int)
    is_train = (sub["fsplit"] == "train").to_numpy()
    is_test = (sub["fsplit"] == "test").to_numpy()
    if len(np.unique(y[is_train])) < 2 or len(np.unique(y[is_test])) < 2:
        return None, [], []

    train_rows = sub.loc[is_train, "row"].to_numpy()
    test_rows = sub.loc[is_test, "row"].to_numpy()
    x_train = x_norm[train_rows]
    x_test = x_norm[test_rows]
    y_train = y[is_train]
    y_test = y[is_test]
    test_df = sub.iloc[np.where(is_test)[0]].copy()

    train_fams, train_centroids = centroid_matrix(
        x_norm[tr.loc[tr["fsplit"] == "train", "row"].to_numpy()],
        tr.loc[tr["fsplit"] == "train", "family"].to_numpy(),
    )
    centroid_cos = nearest_centroid_cosine(x_test, train_centroids)
    knn_prob, top1_cos, mean_topk_cos = cosine_knn_transfer(x_train, y_train, x_test, k=k)
    probe_prob = logistic_probe_prob(x_train, y_train, x_test)

    probe_error = np.abs(y_test - probe_prob)
    knn_error = np.abs(y_test - knn_prob)
    rho_centroid, p_centroid = spearmanr(centroid_cos, probe_error)
    rho_top1, p_top1 = spearmanr(top1_cos, probe_error)
    rho_knn, p_knn = spearmanr(mean_topk_cos, knn_error)

    summary = {
        "trait": trait,
        "n_train": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "test_families": int(test_df["family"].nunique()),
        "test_pos_rate": float(y_test.mean()),
        "mean_nearest_train_family_cos": float(np.mean(centroid_cos)),
        "p10_nearest_train_family_cos": float(np.quantile(centroid_cos, 0.10)),
        "mean_top1_train_genome_cos": float(np.mean(top1_cos)),
        "mean_top10_train_genome_cos": float(np.mean(mean_topk_cos)),
        "knn_f1": macro_f1(y_test, knn_prob),
        "knn_auroc": safe_auroc(y_test, knn_prob),
        "probe_f1": macro_f1(y_test, probe_prob),
        "probe_auroc": safe_auroc(y_test, probe_prob),
        "spearman_centroid_cos_probe_error": float(rho_centroid),
        "p_centroid_cos_probe_error": float(p_centroid),
        "spearman_top1_cos_probe_error": float(rho_top1),
        "p_top1_cos_probe_error": float(p_top1),
        "spearman_topk_cos_knn_error": float(rho_knn),
        "p_topk_cos_knn_error": float(p_knn),
        "verdict": geometry_verdict(safe_auroc(y_test, knn_prob), float(rho_centroid)),
    }
    bins = [{"trait": trait, **r} for r in bin_rows(y_test, knn_prob, probe_prob, centroid_cos)]
    fams = [{"trait": trait, **r} for r in family_rows(test_df, y_test, knn_prob, probe_prob, centroid_cos, top1_cos)]
    return summary, bins, fams


def to_markdown(summary_rows: list[dict], bin_rows_: list[dict]) -> str:
    lines = [
        "# Table 33 -- Cosine geometry diagnostic for family-collapse",
        "",
        "Frozen ESM-2 embeddings are standardized using family-train genomes, L2-normalized, "
        "and evaluated by cosine similarity. Negative Spearman correlations mean lower "
        "cosine support is associated with larger prediction error.",
        "",
        "## Trait-level geometry",
        "",
        "| Trait | Test n | Test families | Pos rate | nearest family cos | top-1 genome cos | kNN F1 | kNN AUROC | Probe F1 | Probe AUROC | rho(cos,error) | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| `{r['trait']}` | {r['n_test']} | {r['test_families']} | {r['test_pos_rate']:.3f} | "
            f"{r['mean_nearest_train_family_cos']:.3f} | {r['mean_top1_train_genome_cos']:.3f} | "
            f"{r['knn_f1']:.3f} | {r['knn_auroc']:.3f} | {r['probe_f1']:.3f} | {r['probe_auroc']:.3f} | "
            f"{r['spearman_centroid_cos_probe_error']:+.3f} | {r['verdict']} |"
        )

    lines += [
        "",
        "## Performance by nearest-training-family cosine bin",
        "",
        "| Trait | Cosine bin | n | mean centroid cos | Pos rate | kNN F1 | kNN AUROC | Probe F1 | Probe AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in bin_rows_:
        au_knn = "nan" if math.isnan(r["knn_auroc"]) else f"{r['knn_auroc']:.3f}"
        au_probe = "nan" if math.isnan(r["probe_auroc"]) else f"{r['probe_auroc']:.3f}"
        lines.append(
            f"| `{r['trait']}` | {r['bin']} | {r['n']} | {r['mean_centroid_cos']:.3f} | "
            f"{r['pos_rate']:.3f} | {r['knn_f1']:.3f} | {au_knn} | {r['probe_f1']:.3f} | {au_probe} |"
        )
    return "\n".join(lines) + "\n"


def run(features_path, splits_path, traits_path, traits, k: int):
    feats, tr = load_aligned(features_path, splits_path, traits_path)
    x_norm = standardize_then_normalize(feats, tr)
    summaries, bins, families = [], [], []
    for trait in traits:
        summary, b, f = evaluate_trait(trait, x_norm, tr, k=k)
        if summary is None:
            continue
        summaries.append(summary)
        bins.extend(b)
        families.extend(f)
    return summaries, bins, families


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    summaries, bins, families = run(args.features, args.splits, args.traits_path, args.traits, args.k)
    if not summaries:
        raise SystemExit("no traits evaluated")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_dir / "33_cosine_family_collapse_summary.csv", index=False)
    pd.DataFrame(bins).to_csv(out_dir / "33_cosine_family_collapse_bins.csv", index=False)
    pd.DataFrame(families).to_csv(out_dir / "33_cosine_family_collapse_families.csv", index=False)
    md = to_markdown(summaries, bins)
    (out_dir / "33_cosine_family_collapse.md").write_text(md)
    print(md)
    print(f"wrote {out_dir}/33_cosine_family_collapse.md and companion CSVs")


if __name__ == "__main__":
    main()
