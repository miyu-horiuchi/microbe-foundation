#!/usr/bin/env python3
"""Generate figures for the frozen-embedding baseline-regressor writeup.

Reads paper/tables/32_baseline_regressors.csv and produces:
  baseline_regressor_schematic.{png,pdf}   experiment pipeline schematic
  baseline_regressor_rank_sweep.{png,pdf}   mean AUROC vs PCA rank per model
  baseline_regressor_per_trait.{png,pdf}    best-model AUROC / macro-F1 per trait

Palette matches make_encoder_figures.py (Wong 2011 colourblind-safe on white).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "tables", "32_baseline_regressors.csv")
COS_BINS_CSV = os.path.join(HERE, "..", "tables", "33_cosine_family_collapse_bins.csv")
CAPACITY_LADDER_RUNGS_CSV = os.path.join(HERE, "..", "tables", "34_capacity_ladder_rungs.csv")
CAPACITY_LADDER_CSV = os.path.join(HERE, "..", "tables", "34_capacity_ladder.csv")
FUSION_CSV = os.path.join(HERE, "..", "tables", "35_fusion.csv")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "figure.dpi": 200,
    }
)

# Wong 2011 colourblind-safe palette
INK = "#222222"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7f7f7f"
ENC_TINT = "#eaf0f7"
PCA_TINT = "#fdf1dd"
HEAD_TINT = "#eef6f1"
EVAL_TINT = "#f2f2f2"

MODEL_LABEL = {
    "logistic_l2": "L2 logistic",
    "sgd_logistic": "SGD logistic",
    "regularized_rf": "Regularized RF",
    "hist_gd": "Hist grad. boosting",
    "poly2_logistic": "Polynomial logistic",
    "soft_vote_ensemble": "Soft-vote ensemble",
}
MODEL_STYLE = {
    "logistic_l2": dict(color=BLUE, marker="o"),
    "regularized_rf": dict(color=GREEN, marker="s"),
    "hist_gd": dict(color=VERM, marker="^"),
    "poly2_logistic": dict(color=PURPLE, marker="D"),
    "sgd_logistic": dict(color=GREY, marker="v"),
}


def save(fig, base):
    p = os.path.join(HERE, base)
    fig.savefig(p + ".png", bbox_inches="tight", facecolor="white")
    fig.savefig(p + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load():
    df = pd.read_csv(CSV)
    return df[df["status"] == "ok"].copy()


# --------------------------------------------------------------------------- #
# Figure 1: schematic
# --------------------------------------------------------------------------- #
def fig_schematic():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, tint, edge, title, lines, title_size=12.5):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.3,rounding_size=1.2",
                linewidth=1.6,
                edgecolor=edge,
                facecolor=tint,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 3.2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=INK,
        )
        for i, ln in enumerate(lines):
            ax.text(
                x + w / 2,
                y + h - 7.0 - i * 3.1,
                ln,
                ha="center",
                va="center",
                fontsize=9.5,
                color="#333333",
            )

    def arrow(x0, x1, y):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y),
                (x1, y),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.8,
                color="#444444",
            )
        )

    y = 20
    h = 13
    box(1, y, 16, h, "#f7f7f7", "#555555", "Genome", ["BacDive strain", "+ trait labels"])
    arrow(17.5, 21, y + h / 2)
    box(21, y, 17, h, ENC_TINT, BLUE, "Frozen ESM-2", ["640-d genome", "embedding"])
    arrow(38.5, 42, y + h / 2)
    box(42, y, 16, h, PCA_TINT, ORANGE, "PCA rank", ["6 / 10 / 25", "capacity sweep"])
    arrow(58.5, 62, y + h / 2)
    box(
        62,
        16.5,
        37,
        20,
        HEAD_TINT,
        GREEN,
        "Simple readouts",
        [
            "L2 logistic  ·  SGD logistic",
            "Polynomial logistic",
            "Regularized random forest",
            "Histogram gradient boosting",
            "Soft-vote ensemble",
        ],
    )
    # down arrow to eval
    ax.add_patch(
        FancyArrowPatch(
            (80.5, 16.5),
            (80.5, 12.5),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#444444",
        )
    )
    box(
        62,
        4,
        37,
        8,
        EVAL_TINT,
        "#555555",
        "Family-held-out AUROC / macro-F1",
        [],
        title_size=11.5,
    )

    ax.text(
        1,
        11.5,
        "Interpretation",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        1,
        7.6,
        "Strong simple readouts  \u2192  frozen embeddings already carry trait signal.",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#333333",
    )
    ax.text(
        1,
        4.4,
        "Rank 25 > rank 6  \u2192  not a six-dimensional perturbation-style task.",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#333333",
    )
    ax.set_title(
        "Baseline readout experiment: is encoder fine-tuning necessary?",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    save(fig, "baseline_regressor_schematic")


# --------------------------------------------------------------------------- #
# Figure 2: rank sweep
# --------------------------------------------------------------------------- #
def fig_rank_sweep(df):
    ranks = [6, 10, 25]
    sweep = df[df["rank"].isin([str(r) for r in ranks] + ranks)].copy()
    sweep["rank_i"] = sweep["rank"].astype(int)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for model in ["logistic_l2", "regularized_rf", "hist_gd", "poly2_logistic", "sgd_logistic"]:
        sub = sweep[sweep["model"] == model]
        means = sub.groupby("rank_i")["auroc"].mean().reindex(ranks)
        st = MODEL_STYLE[model]
        ax.plot(
            ranks,
            means.values,
            color=st["color"],
            marker=st["marker"],
            markersize=7,
            linewidth=2.2,
            label=MODEL_LABEL[model],
        )

    ens = df[df["model"] == "soft_vote_ensemble"]["auroc"].mean()
    ax.axhline(ens, color=INK, linestyle="--", linewidth=1.8, alpha=0.8)
    ax.text(
        25,
        ens + 0.004,
        f"soft-vote ensemble  {ens:.3f}",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=INK,
    )

    ax.set_xticks(ranks)
    ax.set_xlabel("PCA rank before downstream model")
    ax.set_ylabel("Mean AUROC (catalase, motility, sporulation)")
    ax.set_title("Rank 25 beats rank 6 on frozen ESM-2 embeddings")
    ax.set_ylim(0.68, 0.84)
    ax.margins(x=0.08)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.text(
        0.5,
        -0.02,
        "Family-held-out split. Source: paper/tables/32_baseline_regressors.csv",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )
    save(fig, "baseline_regressor_rank_sweep")


# --------------------------------------------------------------------------- #
# Figure 3: best model per trait (AUROC + macro-F1)
# --------------------------------------------------------------------------- #
def fig_per_trait(df):
    traits = ["catalase", "motility", "sporulation"]
    best = {}
    for t in traits:
        sub = df[(df["target"] == t) & (df["model"] != "chance")]
        row = sub.loc[sub["auroc"].idxmax()]
        best[t] = row

    x = np.arange(len(traits))
    w = 0.36
    auroc = [best[t]["auroc"] for t in traits]
    f1 = [best[t]["macro_f1"] for t in traits]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    b1 = ax.bar(x - w / 2, auroc, w, color=BLUE, label="AUROC")
    b2 = ax.bar(x + w / 2, f1, w, color=ORANGE, label="Macro-F1")
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1.4)
    ax.text(len(traits) - 0.5, 0.51, "chance AUROC", ha="right", fontsize=8.5, color=GREY)

    for bars in (b1, b2):
        for rect in bars:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.012,
                f"{rect.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )

    labels = [f"{t}\n({MODEL_LABEL[best[t]['model']]}, r={best[t]['rank']})" for t in traits]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Score on held-out families")
    ax.set_ylim(0, 1.0)
    ax.set_title("Best simple readout per trait")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    save(fig, "baseline_regressor_per_trait")


# --------------------------------------------------------------------------- #
# Figure: cosine-support geometry diagnostic (for the family-collapse paper)
# --------------------------------------------------------------------------- #
def fig_cosine_geometry():
    b = pd.read_csv(COS_BINS_CSV)
    traits = ["catalase", "motility", "sporulation"]
    bins = ["low", "mid", "high"]
    bin_label = {"low": "low\n(far)", "mid": "mid", "high": "high\n(near)"}
    trait_color = {"catalase": BLUE, "motility": ORANGE, "sporulation": GREEN}
    trait_marker = {"catalase": "o", "motility": "s", "sporulation": "^"}

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(bins))
    for t in traits:
        sub = b[b["trait"] == t].set_index("bin").reindex(bins)
        ax.plot(
            x,
            sub["knn_f1"].values,
            color=trait_color[t],
            marker=trait_marker[t],
            markersize=8,
            linewidth=2.4,
            label=t,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([bin_label[bn] for bn in bins])
    ax.set_xlabel("Cosine support of held-out family (nearest training-family centroid)")
    ax.set_ylabel("Cosine kNN macro-F1")
    ax.set_title("Family transfer improves with geometric coverage")
    ax.set_ylim(0.50, 0.92)
    ax.margins(x=0.12)
    ax.legend(frameon=False, loc="lower right", fontsize=10, title="Trait")
    fig.text(
        0.5,
        -0.02,
        "Held-out genomes binned by cosine similarity to the nearest training family. "
        "Source: paper/tables/33_cosine_family_collapse_bins.csv",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )
    save(fig, "enc_fig_cosine_geometry")


RUNG_LABELS = ["lr6", "lr10", "lr25", "lr50", "lr100", "lrfull", "rf", "histgb"]


# --------------------------------------------------------------------------- #
# Figure: capacity ladder (simple readout capacity vs. held-out performance)
# --------------------------------------------------------------------------- #
def fig_capacity_ladder():
    rungs = pd.read_csv(CAPACITY_LADDER_RUNGS_CSV)
    summ = pd.read_csv(CAPACITY_LADDER_CSV).set_index("target")
    trait_color = {"catalase": BLUE, "motility": ORANGE, "sporulation": GREEN}
    trait_marker = {"catalase": "o", "motility": "s", "sporulation": "^"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target, g in rungs.groupby("target"):
        g = g.sort_values("step")
        ax.plot(
            g["step"],
            g["score"],
            color=trait_color.get(target, GREY),
            marker=trait_marker.get(target, "o"),
            markersize=6,
            linewidth=2.0,
            label=target,
        )
        ceil = summ.loc[target, "ceiling_knn_auroc"] if target in summ.index else float("nan")
        if pd.notna(ceil):
            ax.axhline(ceil, color=trait_color.get(target, GREY), ls="--", lw=1.2, alpha=0.6)

    ax.set_xticks(range(8))
    ax.set_xticklabels(RUNG_LABELS, rotation=30)
    ax.set_xlabel("capacity rung (simple → complex)")
    ax.set_ylabel("primary metric on held-out families")
    ax.set_title("Capacity ladder: where simple readouts plateau")
    ax.legend(frameon=False, fontsize=9, title="Trait")
    fig.text(
        0.5,
        -0.02,
        "Dashed lines: cosine kNN ceiling per trait. "
        "Source: paper/tables/34_capacity_ladder_rungs.csv, 34_capacity_ladder.csv",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )
    save(fig, "capacity_ladder")


# --------------------------------------------------------------------------- #
# Figure: fusion lift (does cheap side-data help under family shift?)
# --------------------------------------------------------------------------- #
def fig_fusion_lift():
    fusion = pd.read_csv(FUSION_CSV)
    sub = fusion[(fusion["arm"] == "embed+extra") & fusion["lift"].notna()]
    means = sub.groupby("source")["lift"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [GREEN if v >= 0 else VERM for v in means.values]
    ax.barh(means.index, means.values, color=colors)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("mean lift over embedding-only")
    ax.set_title("Does cheap side-data help under family shift?")
    fig.text(
        0.5,
        -0.02,
        "Source: paper/tables/35_fusion.csv (arm = embed+extra)",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )
    save(fig, "fusion_lift")


def main():
    df = load()
    fig_schematic()
    fig_rank_sweep(df)
    fig_per_trait(df)
    fig_cosine_geometry()
    fig_capacity_ladder()
    fig_fusion_lift()
    print(
        "wrote baseline_regressor_schematic / _rank_sweep / _per_trait, "
        "enc_fig_cosine_geometry, capacity_ladder, and fusion_lift (.png + .pdf)"
    )


if __name__ == "__main__":
    main()
