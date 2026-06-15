#!/usr/bin/env python3
"""lora_finetune_compare.py -- Table 31: frozen vs LoRA vs full encoder adaptation.

Reads the per-run metric JSONs written by finetune_lora.py (--save-metrics) and
builds the decisive comparison: does adapting ESM-2 close the family-split gap a
frozen encoder cannot? Aggregates avg_primary and per-trait scores by mode
(frozen / lora / full), mean +/- std over seeds, and reports the LoRA-minus-frozen
delta -- the headline number of the encoder-adaptation lever.

Run after the GPU jobs land their JSONs (e.g. runs/lora/*.json):
    python3 paper/lora_finetune_compare.py --runs-dir runs/lora
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODE_ORDER = ["frozen", "lora", "full"]


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for fp in sorted(runs_dir.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception as e:
            print(f"[skip] {fp.name}: {e}")
            continue
        if "mode" in d and "per_head" in d:
            runs.append(d)
    return runs


def agg(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


def build(runs: list[dict]):
    by_mode_overall = defaultdict(list)
    by_mode_trait = defaultdict(lambda: defaultdict(list))
    split = runs[0].get("split_level", "?") if runs else "?"
    for r in runs:
        by_mode_overall[r["mode"]].append(r.get("avg_primary"))
        for trait, h in r.get("per_head", {}).items():
            by_mode_trait[r["mode"]][trait].append(h.get("score"))
    return split, by_mode_overall, by_mode_trait


def to_markdown(split, overall, by_trait) -> str:
    present = [m for m in MODE_ORDER if m in overall]
    lines = [
        "# Table 31 -- Encoder adaptation: frozen vs LoRA vs full (the decisive lever)",
        "",
        f"split={split}. Per-run metric JSONs from `finetune_lora.py`, aggregated as "
        "mean +/- std over seeds. `avg_primary` is the across-head headline metric "
        "(per-head primary: acc for binary/multiclass, F1 for multilabel, RMSE for "
        "regression). The LoRA-minus-frozen delta is the encoder-adaptation lever.",
        "",
        "## Overall (avg_primary across heads)",
        "",
        "| Mode | avg_primary | seeds |",
        "|---|---|---:|",
    ]
    means = {}
    for m in present:
        mu, sd, n = agg(overall[m])
        means[m] = mu
        lines.append(f"| {m} | {mu:.4f} +/- {sd:.4f} | {n} |" if mu is not None else f"| {m} | n/a | 0 |")
    if means.get("frozen") is not None and means.get("lora") is not None:
        lines += ["", f"**LoRA - frozen = {means['lora'] - means['frozen']:+.4f} avg_primary.**"]
        if means.get("full") is not None:
            lines.append(f"**full - frozen = {means['full'] - means['frozen']:+.4f}; "
                         f"LoRA recovers {100*(means['lora']-means['frozen'])/(means['full']-means['frozen']):.0f}% "
                         "of full fine-tuning's gain.**" if means["full"] != means["frozen"] else "")

    traits = sorted({t for m in present for t in by_trait[m]})
    lines += ["", "## Per-trait (primary metric, mean over seeds)", "",
              "| Trait | " + " | ".join(present) + " | LoRA-frozen |",
              "|---|" + "|".join(["---"] * present.__len__()) + "|---:|"]
    for t in traits:
        cells = []
        mu_by = {}
        for m in present:
            mu, sd, n = agg(by_trait[m].get(t, []))
            mu_by[m] = mu
            cells.append(f"{mu:.3f}" if mu is not None else "n/a")
        delta = (f"{mu_by['lora'] - mu_by['frozen']:+.3f}"
                 if mu_by.get("lora") is not None and mu_by.get("frozen") is not None else "n/a")
        lines.append(f"| `{t}` | " + " | ".join(cells) + f" | {delta} |")
    return "\n".join(lines) + "\n"


def save_figure(overall, by_trait, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [m for m in MODE_ORDER if m in overall]
    traits = sorted({t for m in present for t in by_trait[m]})
    x = np.arange(len(traits))
    width = 0.8 / max(len(present), 1)
    colors = {"frozen": "#7f8c8d", "lora": "#c0392b", "full": "#2980b9"}
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(traits)), 4.5))
    for i, m in enumerate(present):
        vals = [agg(by_trait[m].get(t, []))[0] or 0.0 for t in traits]
        ax.bar(x + (i - (len(present) - 1) / 2) * width, vals, width,
               label=m, color=colors.get(m, None))
    ax.set_xticks(x)
    ax.set_xticklabels(traits, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("family-test primary metric")
    ax.set_title("Encoder adaptation per trait: frozen vs LoRA vs full")
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs/lora")
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig-dir", default="paper/figures")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs_dir))
    if not runs:
        raise SystemExit(f"no run JSONs in {args.runs_dir} -- run finetune_lora.py first.")
    split, overall, by_trait = build(runs)
    md = to_markdown(split, overall, by_trait)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "31_lora_finetune.md").write_text(md)
    save_figure(overall, by_trait, Path(args.fig_dir) / "31_lora_finetune.png")
    print(md)
    print(f"wrote {out_dir / '31_lora_finetune.md'}")


if __name__ == "__main__":
    main()
