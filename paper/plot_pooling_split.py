"""
plot_pooling_split.py — grouped bar chart of macro trait-prediction quality for
each pooling architecture across the three cross-clade splits (species, genus,
family) plus the family-balanced variant.

Headline figure for the tier-1 pooling comparison. For each run we compute the
macro score (mean over all non-regression heads), then report the mean and a 95%
Student-t confidence interval ACROSS SEEDS (n=3). This seed-level CI is the
honest error bar for the figure -- it reflects run-to-run noise, not the
(much larger) dispersion across heterogeneous traits.

Usage:
    python paper/plot_pooling_split.py --runs-dir runs/tier1_pull \
        --out paper/figures/18_pooling_split_macro.png
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def t95(dof: int) -> float:
    return _T95.get(dof, 1.96) if dof > 0 else float("nan")


def macro_of_run(d: dict) -> float:
    """Mean score over all non-regression heads (higher-is-better)."""
    scores = [
        h["score"]
        for h in d.get("per_head", {}).values()
        if h.get("metric_kind") and h.get("metric_kind") != "rmse" and h.get("score") is not None
    ]
    return sum(scores) / len(scores) if scores else float("nan")


def mean_ci(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return mean, t95(n - 1) * sem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/tier1_pull"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/18_pooling_split_macro.png"))
    args = ap.parse_args()

    # (pooling, column-label) -> list of per-seed macro scores
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for fp in sorted(args.runs_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        pooling = d.get("pooling", "?")
        split = d.get("split_level", "?")
        balanced = bool(d.get("balanced_families", False))
        col = "family\n(balanced)" if (split == "family" and balanced) else split
        groups[(pooling, col)].append(macro_of_run(d))

    poolings = ["mean", "attention", "set_transformer"]
    columns = ["species", "genus", "family", "family\n(balanced)"]
    pretty = {"mean": "Mean pool", "attention": "Attention", "set_transformer": "Set Transformer"}
    colors = {"mean": "#9ecae1", "attention": "#2171b5", "set_transformer": "#fd8d3c"}

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    n_p = len(poolings)
    width = 0.8 / n_p
    x = list(range(len(columns)))

    for i, pool in enumerate(poolings):
        means, errs = [], []
        for col in columns:
            vals = groups.get((pool, col), [])
            if vals:
                m, ci = mean_ci(vals)
            else:
                m, ci = float("nan"), 0.0
            means.append(m)
            errs.append(ci)
        offs = [xi + (i - (n_p - 1) / 2) * width for xi in x]
        bars = ax.bar(offs, means, width, yerr=errs, capsize=3,
                      label=pretty[pool], color=colors[pool],
                      edgecolor="black", linewidth=0.5)
        for b, m in zip(bars, means):
            if not math.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.3f}",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Species", "Genus", "Family", "Family\n(balanced)"])
    ax.set_ylabel("Macro trait-prediction score\n(mean over 20 traits, ±95% CI over 3 seeds)")
    ax.set_title("Cross-clade generalization by pooling architecture")
    ax.set_ylim(0.45, 0.70)
    ax.axvspan(-0.5, 1.5, alpha=0.05, color="green")
    ax.axvspan(1.5, 3.5, alpha=0.06, color="red")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.02)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    # also dump the seed-level macro table that the figure visualizes
    print("pooling           split             mean    ±95%CI   n")
    for pool in poolings:
        for col in columns:
            vals = groups.get((pool, col), [])
            if vals:
                m, ci = mean_ci(vals)
                lbl = col.replace("\n", " ")
                print(f"{pool:17s} {lbl:17s} {m:.4f}  ±{ci:.4f}  {len(vals)}")
    print(f"\nwrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
