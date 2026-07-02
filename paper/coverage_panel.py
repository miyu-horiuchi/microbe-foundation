"""
coverage_panel.py — combined coverage-mechanism panel across coverage-limited traits.

Runs the neighbour-positive-rate analysis (trait_coverage_analysis.analyze) for each
binary coverage-limited trait and assembles:

  - one multi-panel figure (recall on novel-family positives vs neighbour coverage,
    one subplot per trait), and
  - one summary table (Table 21): AUROC, macro-F1, the signal-vs-threshold gap,
    positive concentration, and recall in the no-coverage vs high-coverage bins.

The shared shape across traits is the argument: cross-clade recall is governed by
whether embedding-space neighbours carried the label, not by model capacity.

Only binary traits are supported (the neighbour-positive-rate is a binary notion);
multiclass coverage-limited traits like temperature_class are excluded here.

Usage:
    python paper/coverage_panel.py
    python paper/coverage_panel.py --traits pathogenicity_human motility
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trait_coverage_analysis import analyze  # noqa: E402

DEFAULT_TRAITS = ["pathogenicity_human", "pathogenicity_animal", "motility"]


def summary_row(r: dict) -> dict:
    """Flatten one analyze() result into a Table-21 row (no-coverage vs covered recall)."""
    bins = r["recall_by_neighbor_positive_rate"]
    no_cov = bins[0]                       # [0, 0.001): no positive neighbours
    covered = bins[-1]                     # highest neighbour-positive-rate bin
    def rec(b):
        v = b["recall_on_positives"]
        return None if v != v else v       # NaN -> None
    return {
        "trait": r["trait"],
        "test_pos_rate": r["test_pos_rate"],
        "auroc": r["auroc"],
        "macro_f1": r["macro_f1"],
        "auroc_minus_f1": r["auroc_minus_f1"],
        "positive_gini": r["positive_gini_over_families"],
        "top5_family_share": r["top5_family_share_of_positives"],
        "recall_no_coverage": rec(no_cov),
        "n_pos_no_coverage": no_cov["n_positives"],
        "recall_high_coverage": rec(covered),
        "n_pos_high_coverage": covered["n_positives"],
    }


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "# Table 21 — Coverage mechanism across coverage-limited traits (family-held-out)",
        "",
        "For each trait: a linear probe's ranking quality (AUROC) far exceeds its macro-F1, "
        "positives are concentrated in a few training families (Gini, top-5 share), and "
        "recall on novel-family positives jumps from the no-neighbour-coverage bin to the "
        "high-coverage bin. Same mechanism in every case: cross-clade recall tracks "
        "neighbour coverage, not model capacity.",
        "",
        "| Trait | Test pos rate | AUROC | Macro-F1 | AUROC−F1 | Pos Gini | Top-5 share | "
        "Recall (no cov) | Recall (high cov) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def f(x, p=3):
        return "—" if x is None else f"{x:.{p}f}"
    for r in rows:
        lines.append(
            f"| `{r['trait']}` | {f(r['test_pos_rate'])} | {f(r['auroc'])} | {f(r['macro_f1'])} | "
            f"{r['auroc_minus_f1']:+.3f} | {f(r['positive_gini'])} | {r['top5_family_share']:.1%} | "
            f"{f(r['recall_no_coverage'])} (n={r['n_pos_no_coverage']}) | "
            f"{f(r['recall_high_coverage'])} (n={r['n_pos_high_coverage']}) |"
        )
    return "\n".join(lines) + "\n"


def save_panel(results: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        rows = [b for b in r["recall_by_neighbor_positive_rate"]
                if b["recall_on_positives"] == b["recall_on_positives"]]
        labels = [b["bin"] for b in rows]
        recalls = [b["recall_on_positives"] for b in rows]
        npos = [b["n_positives"] for b in rows]
        bars = ax.bar(range(len(rows)), recalls, color="#d62728", edgecolor="black")
        for bar, nP in zip(bars, npos):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={nP}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{r['trait']}\nAUROC {r['auroc']:.2f} / F1 {r['macro_f1']:.2f}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set_ylabel("Recall on true positives")
    fig.supxlabel("Fraction of k nearest training neighbours that are positive", fontsize=9)
    fig.suptitle("Cross-clade recall tracks neighbour coverage, not capacity", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig", default="paper/figures/coverage_panel.png")
    args = ap.parse_args()

    results, rows = [], []
    for trait in args.traits:
        try:
            r = analyze(trait, args.features, args.splits, args.traits_path, k=args.k)
        except SystemExit as e:
            print(f"  skip {trait}: {e}")
            continue
        results.append(r)
        rows.append(summary_row(r))
    if not results:
        raise SystemExit("no traits analyzed")

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "21_coverage_panel.md"), "w").write(to_markdown(rows))
    import csv
    with open(os.path.join(args.out_dir, "21_coverage_panel.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    save_panel(results, args.fig)
    print(to_markdown(rows))
    print(f"wrote {args.out_dir}/21_coverage_panel.md/.csv and {args.fig}")


if __name__ == "__main__":
    main()
