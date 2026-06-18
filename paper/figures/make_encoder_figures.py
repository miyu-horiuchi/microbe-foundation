#!/usr/bin/env python3
"""Generate figures for the encoder-adaptation family-shift null-result paper.

Reads the two run records (frozen vs LoRA, family split, seed 0) and produces:
  enc_fig1_schematic.png      - controlled-comparison study schematic
  enc_fig2_accuracy_illusion.png - accuracy up / F1 to zero on pathogenicity heads
  enc_fig3_signflip.png       - aggregate gain flips sign once 2 heads removed
  enc_fig4_f1_delta.png       - per-head F1 change (LoRA - frozen), diverging bars
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "runs", "lora")

# ---- aesthetics -------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "figure.dpi": 200,
})
# Colorblind-safe (Wong 2011) palette
FROZEN = "#0072B2"   # blue
LORA = "#E69F00"     # orange
GAIN = "#009E73"     # bluish green
LOSS = "#D55E00"     # vermillion
INK = "#222222"


def load(name):
    with open(os.path.join(RUNS, name)) as f:
        return json.load(f)


def f1_of(head):
    m = head["metrics"]
    if "f1" in m:
        return m["f1"]
    return None


frozen = load("frozen_family_s0.json")
lora = load("lora_family_s0.json")
heads = list(frozen["per_head"].keys())


# ---- Figure 1: study schematic ---------------------------------------------
def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec=INK, fs=10, tc=INK, lw=1.3):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                           linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, wrap=True)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=14, linewidth=1.4, color="#555555"))

    ymid = 24
    box(1, ymid, 17, 11, "Genome\n(set of ~$10^3$\nproteins)", "#f2f2f2", fs=9.5)
    arrow(18, 23, ymid + 5.5)
    box(23, ymid, 18, 11, "ESM-2\nprotein encoder", "#eaf0f7", ec=FROZEN, fs=10)
    arrow(41, 46, ymid + 5.5)
    box(46, ymid, 18, 11, "Set Transformer\npooling", "#f2f2f2", fs=9.5)
    arrow(64, 69, ymid + 5.5)
    box(69, ymid, 14, 11, "21 task\nheads", "#f2f2f2", fs=9.5)
    arrow(83, 88, ymid + 5.5)
    box(88, ymid, 11, 11, "Family-\nheld-out\neval", "#f2f2f2", fs=9)

    # the single variable: two encoder treatments
    ax.annotate("only this block changes", xy=(32, ymid), xytext=(32, 14),
                ha="center", fontsize=9, color=LORA, style="italic",
                arrowprops=dict(arrowstyle="-|>", color=LORA, lw=1.2))
    box(2, 1, 28, 9, "Frozen\nencoder fixed\n(5.54M trainable)", "#eaf0f7",
        ec=FROZEN, fs=9, tc=FROZEN)
    ax.text(31.5, 5.5, "vs", ha="center", va="center", fontsize=11, color=INK)
    box(33, 1, 28, 9, "LoRA  (r=16)\nadapters trained\n(7.39M trainable)", "#fdf1dd",
        ec=LORA, fs=9, tc="#b07900")

    ax.text(50, 40, "Everything held fixed except the encoder treatment",
            ha="center", fontsize=11, color=INK, weight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "enc_fig1_schematic")
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 2: the accuracy illusion ---------------------------------------
def fig_accuracy_illusion():
    labels = ["pathogenicity\nhuman", "pathogenicity\nanimal"]
    keys = ["pathogenicity_human", "pathogenicity_animal"]
    acc_f = [frozen["per_head"][k]["metrics"]["acc"] for k in keys]
    acc_l = [lora["per_head"][k]["metrics"]["acc"] for k in keys]
    f1_f = [frozen["per_head"][k]["metrics"]["f1"] for k in keys]
    f1_l = [lora["per_head"][k]["metrics"]["f1"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    x = np.arange(2)
    w = 0.36
    for ax, (vf, vl, title) in zip(
        axes, [(acc_f, acc_l, "Accuracy  (primary metric)"),
               (f1_f, f1_l, "F1  (positive-class)")]):
        b1 = ax.bar(x - w / 2, vf, w, label="Frozen", color=FROZEN)
        b2 = ax.bar(x + w / 2, vl, w, label="LoRA", color=LORA)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.02,
                    f"{h:.2f}" if h > 0 else "0.00",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    axes[0].set_ylabel("score")
    # up / down annotations
    axes[0].annotate("", xy=(0.18, 0.97), xytext=(-0.18, 0.74),
                     arrowprops=dict(arrowstyle="-|>", color=GAIN, lw=2))
    axes[0].text(0.5, 0.55, "accuracy\nUP", color=GAIN, ha="center",
                 fontsize=9, weight="bold")
    axes[1].text(0.5, 0.5, "F1 collapses\nto 0\n(majority-class)", color=LOSS,
                 ha="center", fontsize=9, weight="bold")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("The aggregate \u201cgain\u201d is an accuracy illusion",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(HERE, "enc_fig2_accuracy_illusion")
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 3: sign flip ----------------------------------------------------
def fig_signflip():
    drop = {"pathogenicity_human", "pathogenicity_animal"}

    def avg_primary(run, exclude=()):
        vals = [v["score"] for k, v in run["per_head"].items() if k not in exclude]
        # fatty acid is RMSE (lower better) but it is a small constant offset in both;
        # avg_primary in the records mixes metric kinds exactly as the paper reports.
        return float(np.mean(vals))

    all_delta = lora["avg_primary"] - frozen["avg_primary"]
    ex_l = avg_primary(lora, drop)
    ex_f = avg_primary(frozen, drop)
    ex_delta = ex_l - ex_f

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    cats = ["All 21 heads", "Excluding 2\npathogenicity heads"]
    vals = [all_delta, ex_delta]
    colors = [GAIN if v > 0 else LOSS for v in vals]
    bars = ax.bar(cats, vals, color=colors, width=0.5)
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylabel("LoRA \u2212 Frozen  (avg primary)")
    ax.set_title("Removing two heads flips the conclusion", fontsize=12, weight="bold")
    lim = max(abs(v) for v in vals) * 1.6
    ax.set_ylim(-lim, lim)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.0015 if v > 0 else -0.0015),
                f"{v:+.4f}", ha="center",
                va="bottom" if v > 0 else "top",
                fontsize=11, weight="bold",
                color=GAIN if v > 0 else LOSS)
    ax.text(0, lim * 0.85, "looks like a win", ha="center", fontsize=9,
            color=GAIN, style="italic")
    ax.text(1, -lim * 0.85, "actually a loss", ha="center", fontsize=9,
            color=LOSS, style="italic")
    fig.tight_layout()
    out = os.path.join(HERE, "enc_fig3_signflip")
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return all_delta, ex_delta


# ---- Figure 4: per-head F1 delta -------------------------------------------
def fig_f1_delta():
    rows = []
    for k in heads:
        ff = f1_of(frozen["per_head"][k])
        fl = f1_of(lora["per_head"][k])
        if ff is None or fl is None:
            continue
        rows.append((k, fl - ff))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    colors = [GAIN if d > 1e-4 else (LOSS if d < -1e-4 else "#bcbcbc") for d in deltas]

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    y = np.arange(len(names))
    ax.barh(y, deltas, color=colors)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax.set_xlabel("F1 change  (LoRA \u2212 Frozen)")
    ax.set_title("Encoder adaptation is neutral-to-harmful on F1",
                 fontsize=12, weight="bold")
    ax.grid(axis="y", visible=False)
    for yi, d in zip(y, deltas):
        if abs(d) < 1e-4:
            continue
        ax.text(d + (0.004 if d > 0 else -0.004), yi, f"{d:+.3f}",
                va="center", ha="left" if d > 0 else "right",
                fontsize=8, color=INK)
    # highlight the two artifact heads
    for i, n in enumerate(names):
        if n in ("pathogenicity_human", "pathogenicity_animal"):
            ax.get_yticklabels()[i].set_color(LORA)
            ax.get_yticklabels()[i].set_weight("bold")
    pad = max(abs(min(deltas)), abs(max(deltas))) * 1.35
    ax.set_xlim(-pad, pad)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LOSS, label="F1 loss"),
                       Patch(color=GAIN, label="F1 gain"),
                       Patch(color="#bcbcbc", label="no change")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "enc_fig4_f1_delta")
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 5: accuracy-vs-F1 scatter (artifact generalizes) ---------------
def fig_scatter():
    rows = []
    for k in heads:
        mf, ml = frozen["per_head"][k]["metrics"], lora["per_head"][k]["metrics"]
        if "acc" in mf and "acc" in ml and "f1" in mf and "f1" in ml:
            rows.append((k, ml["acc"] - mf["acc"], ml["f1"] - mf["f1"]))
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    # shade the "accuracy illusion" quadrant: acc up, F1 down
    ax.axhspan(-1, 0, xmin=0.5, xmax=1, color="#fbe3d6", zorder=0)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(0, color=INK, lw=1)
    for k, dacc, df1 in rows:
        artifact = dacc > 0 and df1 < -1e-4
        ax.scatter(dacc, df1, s=70,
                   color=LOSS if artifact else FROZEN,
                   edgecolor="white", linewidth=0.8, zorder=3)
        if abs(dacc) > 0.03 or abs(df1) > 0.03:
            ax.annotate(k.replace("_", " "), (dacc, df1),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=INK)
    ax.set_xlabel("accuracy change  (LoRA \u2212 Frozen)")
    ax.set_ylabel("F1 change  (LoRA \u2212 Frozen)")
    ax.set_title("Where accuracy rises, F1 falls", fontsize=12, weight="bold")
    ax.text(0.74, 0.30, "accuracy up, F1 down\n(the artifact)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color=LOSS, style="italic")
    xl = max(abs(x[1]) for x in rows) * 1.25
    yl = max(abs(x[2]) for x in rows) * 1.25
    ax.set_xlim(-xl, xl)
    ax.set_ylim(-yl, yl)
    fig.tight_layout()
    out = os.path.join(HERE, "enc_fig5_acc_vs_f1")
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_schematic()
    fig_accuracy_illusion()
    ad, ed = fig_signflip()
    fig_f1_delta()
    fig_scatter()
    print(f"avg_primary delta all heads:        {ad:+.4f}")
    print(f"avg_primary delta excl. 2 heads:    {ed:+.4f}")
    print("figures written to", HERE)
