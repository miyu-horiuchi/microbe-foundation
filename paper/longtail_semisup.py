#!/usr/bin/env python3
"""longtail_semisup.py -- coverage-efficient long-tail learning (lever #2).

The bottleneck for coverage-limited traits is labelled positives concentrated in a
few clades (Gini >= 0.85). But we hold ~19.6k genomes with embeddings and only a
fraction are labelled for any given trait -- the rest are an *unlabelled* pool.
This script tests whether exploiting that pool via semi-supervised self-training
recovers cross-clade accuracy without any new labels or encoder fine-tuning.

Method (per binary trait, per split level):
    supervised   balanced logistic regression on the labelled {level}-train genomes
    self_train   1. fit the supervised probe
                 2. pseudo-label the UNLABELLED genomes that fall in TRAINING
                    clades only (split == train; leak-safe -- never test clades)
                 3. keep high-confidence pseudo-labels (p > conf or p < 1-conf)
                 4. refit on labelled + pseudo-labelled (pseudo down-weighted)
    Evaluate both on {level}-test; report self_train minus supervised.

Hypothesis: densifying the training-distribution manifold with unlabelled genomes
sharpens the boundary and lifts transfer to neighbouring unseen clades. A null
result is informative -- it says unlabelled in-distribution data does not, on its
own, close an OOD gap, and the headroom is genuinely new labels / new clades.

Leakage control: pseudo-label pool is restricted to split == train; scaler/probe
fit on labelled train; test scored once. CPU-only, mean-pooled features.

Usage:
    python3 paper/longtail_semisup.py
    python3 paper/longtail_semisup.py --conf 0.85 --pseudo-weight 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ood_error_analysis import binary_trait_labels  # noqa: E402

# Coverage-limited / rare-positive binary traits where extra coverage could help.
TRAITS = ["motility", "sporulation", "pathogenicity_human", "pathogenicity_animal"]
LEVELS = ["species", "genus", "family"]


def macro_f1(y_true, prob) -> float:
    return float(f1_score(y_true, (np.asarray(prob) > 0.5).astype(int),
                          average="macro", zero_division=0))


def fit_probe(X, y, scaler, sample_weight=None):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X), y, sample_weight=sample_weight)
    return clf


def eval_trait(feats, lab, pool, conf, pseudo_weight) -> dict | None:
    """lab: dict split->(X,y) labelled; pool: X of unlabelled training-clade genomes."""
    if any(len(np.unique(lab[p][1])) < 2 for p in ("train", "test")):
        return None
    Xtr, ytr = lab["train"]
    Xte, yte = lab["test"]
    scaler = StandardScaler().fit(Xtr)

    sup = fit_probe(Xtr, ytr, scaler)
    sup_te = sup.predict_proba(scaler.transform(Xte))[:, 1]

    n_pseudo = 0
    if len(pool):
        p_pool = sup.predict_proba(scaler.transform(pool))[:, 1]
        keep = (p_pool > conf) | (p_pool < 1 - conf)
        n_pseudo = int(keep.sum())

    if n_pseudo >= 2:
        Xps = pool[keep]
        yps = (p_pool[keep] > 0.5).astype(float)
        X_aug = np.vstack([Xtr, Xps])
        y_aug = np.concatenate([ytr, yps])
        w = np.concatenate([np.ones(len(ytr)), np.full(len(yps), pseudo_weight)])
        if len(np.unique(y_aug)) >= 2:
            st = fit_probe(X_aug, y_aug, scaler, sample_weight=w)
            st_te = st.predict_proba(scaler.transform(Xte))[:, 1]
        else:
            st_te = sup_te
    else:
        st_te = sup_te

    return {
        "n_train": int(len(ytr)), "n_pool": int(len(pool)), "n_pseudo": n_pseudo,
        "n_test": int(len(yte)), "pos_rate": float(yte.mean()),
        "sup_f1": macro_f1(yte, sup_te), "sup_auroc": float(roc_auc_score(yte, sup_te)),
        "st_f1": macro_f1(yte, st_te), "st_auroc": float(roc_auc_score(yte, st_te)),
        "delta_f1": macro_f1(yte, st_te) - macro_f1(yte, sup_te),
        "delta_auroc": float(roc_auc_score(yte, st_te) - roc_auc_score(yte, sup_te)),
    }


def run(features_path, splits_path, traits_path, conf, pseudo_weight, traits=TRAITS):
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
        split_map = dict(zip(sp["bid"], sp[f"{level}_split"]))
        tr["split"] = tr["bid"].map(lambda b: split_map.get(b, "unknown"))
        for trait in traits:
            if trait not in tr.columns:
                continue
            y, mask = binary_trait_labels(tr[trait])
            lab = {}
            for part in ("train", "val", "test"):
                sel = mask & (tr["split"] == part).to_numpy()
                lab[part] = (feats[tr["row"].to_numpy()[sel]], y[sel])
            # Unlabelled pool: embedded, training-clade, but NOT labelled for this trait.
            pool_sel = (~mask) & (tr["split"] == "train").to_numpy()
            pool = feats[tr["row"].to_numpy()[pool_sel]]
            r = eval_trait(feats, lab, pool, conf, pseudo_weight)
            if r is None:
                print(f"  skip {level}/{trait}: single-class split")
                continue
            results.append({"level": level, "trait": trait, **r})
    return pd.DataFrame(results)


def to_markdown(df: pd.DataFrame, conf, pseudo_weight) -> str:
    lines = [
        "# Table 30 -- Coverage-efficient long-tail via semi-supervised self-training (lever #2)",
        "",
        f"conf={conf}  pseudo_weight={pseudo_weight}. Pseudo-labels drawn only from "
        "unlabelled genomes in TRAINING clades (leak-safe). `delta` = self-train minus "
        "supervised on {level}-test.",
        "",
        "## Per-trait, per-split",
        "",
        "| Level | Trait | Train n | Pool | Pseudo | Test n | Sup F1 | ST F1 | dF1 | "
        "Sup AUC | ST AUC | dAUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['level']} | `{r['trait']}` | {r['n_train']} | {r['n_pool']} | "
            f"{r['n_pseudo']} | {r['n_test']} | {r['sup_f1']:.3f} | {r['st_f1']:.3f} | "
            f"{r['delta_f1']:+.3f} | {r['sup_auroc']:.3f} | {r['st_auroc']:.3f} | "
            f"{r['delta_auroc']:+.3f} |")
    lines += ["", "## Mean delta over traits, by split", "",
              "| Level | mean dF1 | mean dAUROC | mean pseudo n |",
              "|---|---:|---:|---:|"]
    for level in LEVELS:
        d = df[df.level == level]
        if d.empty:
            continue
        lines.append(f"| {level} | {d['delta_f1'].mean():+.3f} | {d['delta_auroc'].mean():+.3f} | "
                     f"{d['n_pseudo'].mean():.0f} |")
    return "\n".join(lines) + "\n"


def save_figure(df: pd.DataFrame, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1 = df.groupby("level")["delta_f1"].mean().reindex(LEVELS)
    auc = df.groupby("level")["delta_auroc"].mean().reindex(LEVELS)
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, f1.values, 0.4, label="mean dMacro-F1", color="#27ae60")
    ax.bar(x + 0.2, auc.values, 0.4, label="mean dAUROC", color="#2980b9")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(harder ->)" for l in LEVELS])
    ax.set_ylabel("self-train minus supervised (test)")
    ax.set_title("Semi-supervised self-training: gain vs taxonomic distance")
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
    ap.add_argument("--conf", type=float, default=0.85)
    ap.add_argument("--pseudo-weight", type=float, default=0.3)
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig-dir", default="paper/figures")
    args = ap.parse_args()

    df = run(args.features, args.splits, args.traits, args.conf, args.pseudo_weight)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "30_longtail_semisup.csv", index=False)
    md = to_markdown(df, args.conf, args.pseudo_weight)
    (out_dir / "30_longtail_semisup.md").write_text(md)
    save_figure(df, Path(args.fig_dir) / "30_longtail_semisup.png")
    print(md)


if __name__ == "__main__":
    main()
