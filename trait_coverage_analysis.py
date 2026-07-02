"""
trait_coverage_analysis.py — confirm *why* a collapsing trait collapses off-clade.

Picks one trait (default pathogenicity_human) and shows, on the family-held-out
split, the mechanism behind its cross-clade collapse:

  1. Signal-vs-threshold gap: a linear probe's AUROC on novel families is well
     above chance even though its macro-F1 is on the floor — the embedding carries
     the signal; imbalance + a 0.5 threshold destroy F1.
  2. Positive concentration: a few training families hold most of the positive
     labels (high Gini), so most novel families have no nearby positive example.
  3. Neighbor-positive-rate stratification (the headline): for each novel-family
     test genome, the fraction of its k nearest *training* neighbours that are
     positive. Recall on true positives is ~0 when neighbours are all negative and
     rises steeply with neighbour positive rate — the model can only flag a novel
     pathogen when embedding-space neighbours were known pathogens. That *is* the
     coverage mechanism, made concrete.

Torch-free; uses the real ESM-2 embeddings and the same probe other tables use.

Usage:
    python trait_coverage_analysis.py --trait pathogenicity_human
    python trait_coverage_analysis.py --trait sporulation   # contrast: a solved trait
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cross_clade_diagnostic import _macro_f1
from ood_error_analysis import binary_trait_labels

DEFAULT_TRAIT = "pathogenicity_human"
DEFAULT_BINS = [0.0, 0.001, 0.1, 0.3, 1.01]  # neighbour positive-rate edges


def gini(counts) -> float:
    """Gini coefficient of a non-negative vector (0 = uniform, ->1 = concentrated)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def neighbor_positive_rate(X_train_s, y_train, X_test_s, k: int) -> np.ndarray:
    """For each test row, fraction of its k nearest TRAIN neighbours that are positive."""
    nn = NearestNeighbors(n_neighbors=min(k, len(X_train_s))).fit(X_train_s)
    idx = nn.kneighbors(X_test_s, return_distance=False)
    return y_train[idx].mean(axis=1)


def recall_by_bin(npr, y_test, proba, bins, thr=0.5) -> list[dict]:
    """Positive-class recall stratified by neighbour-positive-rate bin."""
    pred = (proba >= thr).astype(int)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (npr >= lo) & (npr < hi)
        pos = sel & (y_test == 1)
        n_pos = int(pos.sum())
        recall = float(pred[pos].mean()) if n_pos else float("nan")
        out.append({
            "bin": f"[{lo:.3g}, {hi:.3g})",
            "n_genomes": int(sel.sum()),
            "n_positives": n_pos,
            "recall_on_positives": recall,
        })
    return out


def analyze(trait, features_path, splits_path, traits_path, k=10, bins=DEFAULT_BINS) -> dict:
    data = np.load(features_path)
    feats = data["features"]
    ids = np.array([str(i) for i in data["bacdive_ids"]])
    id_to_row = {b: i for i, b in enumerate(ids)}

    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    tr = tr[tr["bid"].isin(id_to_row)].copy()
    sp = pd.read_parquet(splits_path)[["bacdive_id", "family"]].copy()
    sp["bid"] = sp["bacdive_id"].astype(str)
    fam_name = dict(zip(sp["bid"], sp["family"]))
    spl = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    fam_split = dict(zip(spl["bacdive_id"].astype(str), spl["family_split"]))
    tr["fsplit"] = tr["bid"].map(lambda b: fam_split.get(b, "unknown"))
    tr["famname"] = tr["bid"].map(lambda b: fam_name.get(b))
    tr["row"] = tr["bid"].map(id_to_row)

    if trait not in tr.columns:
        raise SystemExit(f"trait '{trait}' not in traits parquet")
    y, mask = binary_trait_labels(tr[trait])
    sub = tr[mask].copy()
    sub["y"] = y[mask]
    train = sub[sub["fsplit"] == "train"]
    test = sub[sub["fsplit"] == "test"]
    if train["y"].nunique() < 2 or test["y"].nunique() < 2:
        raise SystemExit(f"trait '{trait}' is single-class in train or test on family split")

    X_train = feats[train["row"].to_numpy()]
    y_train = train["y"].to_numpy().astype(int)
    X_test = feats[test["row"].to_numpy()]
    y_test = test["y"].to_numpy().astype(int)

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, y_train)
    proba = clf.predict_proba(Xte_s)[:, 1]

    auroc = float(roc_auc_score(y_test, proba))
    f1 = _macro_f1(y_test, proba)

    # positive concentration across TRAIN families
    pos_per_family = train[train["y"] == 1].groupby("famname").size()
    all_families = train["famname"].nunique()
    g = gini(pos_per_family.reindex(train["famname"].unique(), fill_value=0).to_numpy())
    top5_share = float(pos_per_family.sort_values(ascending=False).head(5).sum()
                       / max(pos_per_family.sum(), 1))

    npr = neighbor_positive_rate(Xtr_s, y_train, Xte_s, k=k)
    bin_rows = recall_by_bin(npr, y_test, proba, bins)

    return {
        "trait": trait,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "train_pos_rate": float(y_train.mean()), "test_pos_rate": float(y_test.mean()),
        "auroc": auroc, "macro_f1": f1, "auroc_minus_f1": auroc - f1,
        "n_train_families": int(all_families),
        "n_families_with_positives": int((pos_per_family > 0).sum()),
        "positive_gini_over_families": g,
        "top5_family_share_of_positives": top5_share,
        "k": k,
        "recall_by_neighbor_positive_rate": bin_rows,
    }


def to_markdown(r: dict) -> str:
    lines = [
        f"# Trait coverage analysis — `{r['trait']}` (family-held-out)",
        "",
        f"- Train / test genomes: {r['n_train']} / {r['n_test']}  "
        f"(pos rate {r['train_pos_rate']:.3f} / {r['test_pos_rate']:.3f})",
        f"- **Signal vs threshold:** probe AUROC = **{r['auroc']:.3f}** but macro-F1 = "
        f"**{r['macro_f1']:.3f}** (gap {r['auroc_minus_f1']:+.3f}). The embedding ranks "
        f"novel-family positives well above chance; imbalance + a 0.5 threshold flatten F1.",
        f"- **Positive concentration:** {r['n_families_with_positives']} of "
        f"{r['n_train_families']} training families contain any positive; Gini = "
        f"{r['positive_gini_over_families']:.3f}; top-5 families hold "
        f"{r['top5_family_share_of_positives']:.1%} of all positive labels.",
        "",
        f"## Recall on true positives by neighbour-positive-rate (k={r['k']})",
        "",
        "Fraction of each novel-family test genome's k nearest *training* neighbours that "
        "are positive. Recall climbs with neighbour positive rate: the model flags a novel "
        "pathogen only when embedding-space neighbours were known pathogens — the coverage "
        "mechanism.",
        "",
        "| Neighbour pos-rate | Genomes | Positives | Recall on positives |",
        "|---|---:|---:|---:|",
    ]
    for b in r["recall_by_neighbor_positive_rate"]:
        rec = "—" if b["recall_on_positives"] != b["recall_on_positives"] else f"{b['recall_on_positives']:.3f}"
        lines.append(f"| {b['bin']} | {b['n_genomes']} | {b['n_positives']} | {rec} |")
    return "\n".join(lines) + "\n"


def save_figure(r: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [b for b in r["recall_by_neighbor_positive_rate"]
            if b["recall_on_positives"] == b["recall_on_positives"]]
    labels = [b["bin"] for b in rows]
    recalls = [b["recall_on_positives"] for b in rows]
    npos = [b["n_positives"] for b in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(rows)), recalls, color="#d62728", edgecolor="black")
    for i, (bar, n) in enumerate(zip(bars, npos)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Fraction of k={r['k']} nearest training neighbours that are positive")
    ax.set_ylabel("Recall on true positives")
    ax.set_title(f"{r['trait']}: novel-family recall depends on neighbour coverage")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", default=DEFAULT_TRAIT)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits", default="data/traits.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig", default=None, help="Figure path (default paper/figures/coverage_<trait>.png)")
    args = ap.parse_args()

    r = analyze(args.trait, args.features, args.splits, args.traits, k=args.k)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(r)
    (out_dir / f"coverage_{args.trait}.md").write_text(md)

    fig = args.fig or f"paper/figures/coverage_{args.trait}.png"
    Path(fig).parent.mkdir(parents=True, exist_ok=True)
    save_figure(r, fig)
    print(md)
    print(f"wrote {out_dir / f'coverage_{args.trait}.md'} and {fig}")


if __name__ == "__main__":
    main()
