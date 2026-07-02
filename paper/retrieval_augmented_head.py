#!/usr/bin/env python3
"""retrieval_augmented_head.py -- bake retrieval into the predictor (lever #1).

The cross-clade diagnostic (Table 15) and the static blend (Table 16) showed the
frozen ESM-2 geometry carries cross-family trait signal that a parametric probe
leaves partly on the table. This script turns that into an actual *prediction
rule* and asks how much it buys as the held-out clade gets more distant:

    p_final = alpha * p_probe + (1 - alpha) * p_retrieval

We compare three retrieval variants against the probe-alone baseline, for every
binary-head trait, across all three taxonomic splits (species/genus/family):

    probe          balanced logistic regression on standardized embeddings
    knn_uniform    fraction of the k nearest training neighbours that are positive
    knn_phylo      distance-weighted neighbours, w_i = exp(-d_i^2 / 2 sigma^2),
                   sigma = median neighbour distance  (phylogeny-aware: closer
                   genomes -- nearer in ESM-2 space, a proxy for evolutionary
                   distance -- count more)
    blend          alpha * probe + (1 - alpha) * knn_phylo, alpha tuned on val

Hypothesis: retrieval's marginal value rises with taxonomic distance, because the
parametric boundary degrades off-distribution while local neighbour evidence does
not. A flat / negative delta on family would falsify the lever.

Leakage control: scaler, probe, and the retrieval reference manifold are fit on
{level}-train only; alpha is tuned on {level}-val; {level}-test is scored once.
CPU-only, mean-pooled features.

Usage:
    python3 paper/retrieval_augmented_head.py
    python3 paper/retrieval_augmented_head.py --k 10 --out-dir paper/tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ood_error_analysis import binary_trait_labels  # noqa: E402

BINARY_TRAITS = [
    "motility", "sporulation", "catalase", "cytochrome_oxidase",
    "pigmentation", "pathogenicity_human", "pathogenicity_animal",
]
LEVELS = ["species", "genus", "family"]
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]


def macro_f1(y_true, prob) -> float:
    return float(f1_score(y_true, (np.asarray(prob) > 0.5).astype(int),
                          average="macro", zero_division=0))


def knn_predict(X_ref, y_ref, X_query, k: int, phylo: bool):
    """k-NN positive-probability. phylo=True -> Gaussian distance weighting."""
    X_ref = np.asarray(X_ref, float)
    y_ref = np.asarray(y_ref, float)
    X_query = np.asarray(X_query, float)
    kk = min(k, len(X_ref))
    nn = NearestNeighbors(n_neighbors=kk).fit(X_ref)
    dist, idx = nn.kneighbors(X_query)
    yk = y_ref[idx]                                   # [Nq, k]
    if not phylo:
        return yk.mean(axis=1)
    sigma = np.median(dist[dist > 0]) or 1.0          # global bandwidth
    w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))     # [Nq, k]
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    return (w * yk).sum(axis=1) / w_sum.ravel()


def tune_alpha(probe_val, retr_val, y_val) -> float:
    """Grid alpha maximizing val macro-F1; ties lean toward retrieval (small alpha)."""
    f1s = [macro_f1(y_val, a * probe_val + (1 - a) * retr_val) for a in ALPHA_GRID]
    return float(ALPHA_GRID[int(np.argmax(f1s))])


def eval_trait(feats, sub, k: int) -> dict | None:
    """sub has columns row, y, split in {train,val,test}. Returns metrics or None."""
    parts = {}
    for p in ("train", "val", "test"):
        sel = (sub["split"] == p).to_numpy()
        parts[p] = (feats[sub["row"].to_numpy()[sel]], sub["y"].to_numpy()[sel])
    if any(len(np.unique(y)) < 2 for _, y in parts.values()):
        return None
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, ytr)
    probe_va = clf.predict_proba(Xva_s)[:, 1]
    probe_te = clf.predict_proba(Xte_s)[:, 1]

    uni_te = knn_predict(Xtr_s, ytr, Xte_s, k, phylo=False)
    phylo_va = knn_predict(Xtr_s, ytr, Xva_s, k, phylo=True)
    phylo_te = knn_predict(Xtr_s, ytr, Xte_s, k, phylo=True)

    alpha = tune_alpha(probe_va, phylo_va, yva)
    blend_te = alpha * probe_te + (1 - alpha) * phylo_te

    return {
        "n_test": int(len(yte)), "pos_rate": float(yte.mean()), "alpha_star": alpha,
        "probe_f1": macro_f1(yte, probe_te), "probe_auroc": float(roc_auc_score(yte, probe_te)),
        "knn_uniform_f1": macro_f1(yte, uni_te), "knn_uniform_auroc": float(roc_auc_score(yte, uni_te)),
        "knn_phylo_f1": macro_f1(yte, phylo_te), "knn_phylo_auroc": float(roc_auc_score(yte, phylo_te)),
        "blend_f1": macro_f1(yte, blend_te), "blend_auroc": float(roc_auc_score(yte, blend_te)),
        "delta_f1": macro_f1(yte, blend_te) - macro_f1(yte, probe_te),
        "delta_auroc": float(roc_auc_score(yte, blend_te) - roc_auc_score(yte, probe_te)),
    }


def run(features_path, splits_path, traits_path, k, traits=BINARY_TRAITS):
    data = np.load(features_path)
    feats = data["features"]
    ids = np.array([str(i) for i in data["bacdive_ids"]])
    id_to_row = {b: i for i, b in enumerate(ids)}
    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    tr = tr[tr["bid"].isin(id_to_row)].copy()
    tr["row"] = tr["bid"].map(id_to_row)
    sp = pd.read_parquet(splits_path)
    sp["bid"] = sp["bacdive_id"].astype(str)

    results = []
    for level in LEVELS:
        col = f"{level}_split"
        split_map = dict(zip(sp["bid"], sp[col]))
        for trait in traits:
            if trait not in tr.columns:
                continue
            y, mask = binary_trait_labels(tr[trait])
            sub = tr[mask].copy()
            sub["y"] = y[mask]
            sub["split"] = sub["bid"].map(lambda b: split_map.get(b, "unknown"))
            r = eval_trait(feats, sub, k)
            if r is None:
                print(f"  skip {level}/{trait}: single-class split")
                continue
            results.append({"level": level, "trait": trait, **r})
    return pd.DataFrame(results)


def to_markdown(df: pd.DataFrame, k: int) -> str:
    lines = [
        "# Table 29 -- Retrieval-augmented prediction head (lever #1)",
        "",
        f"k={k}. Distance-weighted (phylogeny-aware) k-NN blended with the linear "
        "probe, `alpha*probe + (1-alpha)*knn_phylo`, alpha tuned on {level}-val and "
        "scored once on {level}-test. `delta` columns are blend minus probe-alone.",
        "",
        "## Per-trait, per-split",
        "",
        "| Level | Trait | Test n | Pos | a* | Probe F1 | kNN-uni F1 | kNN-phylo F1 | "
        "Blend F1 | dF1 | Probe AUC | Blend AUC | dAUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['level']} | `{r['trait']}` | {r['n_test']} | {r['pos_rate']:.2f} | "
            f"{r['alpha_star']:.1f} | {r['probe_f1']:.3f} | {r['knn_uniform_f1']:.3f} | "
            f"{r['knn_phylo_f1']:.3f} | {r['blend_f1']:.3f} | {r['delta_f1']:+.3f} | "
            f"{r['probe_auroc']:.3f} | {r['blend_auroc']:.3f} | {r['delta_auroc']:+.3f} |")
    lines += ["", "## Mean delta over traits, by split (does retrieval help more OOD?)", "",
              "| Level | mean dF1 | mean dAUROC | mean a* | n traits |",
              "|---|---:|---:|---:|---:|"]
    for level in LEVELS:
        d = df[df.level == level]
        if d.empty:
            continue
        lines.append(f"| {level} | {d['delta_f1'].mean():+.3f} | {d['delta_auroc'].mean():+.3f} | "
                     f"{d['alpha_star'].mean():.2f} | {len(d)} |")
    return "\n".join(lines) + "\n"


def save_figure(df: pd.DataFrame, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = df.groupby("level")["delta_f1"].mean().reindex(LEVELS)
    aucs = df.groupby("level")["delta_auroc"].mean().reindex(LEVELS)
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, means.values, 0.4, label="mean dMacro-F1", color="#c0392b")
    ax.bar(x + 0.2, aucs.values, 0.4, label="mean dAUROC", color="#2980b9")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(harder ->)" for l in LEVELS])
    ax.set_ylabel("blend minus probe-alone (test)")
    ax.set_title("Retrieval-augmented head: marginal gain vs taxonomic distance")
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits", default="data/traits.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig-dir", default="paper/figures")
    args = ap.parse_args()

    df = run(args.features, args.splits, args.traits, args.k)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "29_retrieval_augmented_head.csv", index=False)
    md = to_markdown(df, args.k)
    (out_dir / "29_retrieval_augmented_head.md").write_text(md)
    save_figure(df, Path(args.fig_dir) / "29_retrieval_augmented_head.png")
    print(md)


if __name__ == "__main__":
    main()
