"""
failure_mode_analysis.py — classify every trait's failure mode from the real runs.

The central message of the project is that "the model can't predict functions" is
not one problem but three, and only one is a model problem. This script makes that
quantitative: it reads the per-seed metric JSONs (model.py --save-metrics) for the
species/genus/family splits and, per trait, computes

  - in-distribution ceiling   = species-split F1
  - cross-clade generalization = family-split F1
  - decay                      = species F1 - family F1

then assigns a failure mode:

  solved           family F1 >= SOLVED_FAMILY              (sequence-intrinsic; done)
  label-ceiling    species F1 <  FLOOR_SPECIES             (bad even in-distribution;
                                                            label/task problem, not the model)
  coverage-limited species F1 ok but decay >= DECAY_GAP    (learnable, collapses off-clade;
                                                            needs training-clade coverage)
  moderate-flat    partial signal, generalizes but capped  (no collapse, no ceiling)

It emits Table 20 (md + csv) and a scatter figure (ceiling vs decay, colored by mode)
that visually separates the regimes.

Usage:
    python paper/failure_mode_analysis.py --runs-dir runs --prefix attnpool
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

SOLVED_FAMILY = 0.70
FLOOR_SPECIES = 0.30
DECAY_GAP = 0.10

MODE_COLORS = {
    "solved": "#2ca02c",            # green
    "coverage-limited": "#d62728",  # red
    "label-ceiling": "#7f7f7f",     # grey
    "moderate-flat": "#1f77b4",     # blue
    "incomplete": "#cccccc",
}


def classify_failure_mode(species_f1, family_f1,
                          solved=SOLVED_FAMILY, floor=FLOOR_SPECIES, gap=DECAY_GAP) -> str:
    """Assign a trait to one of the four failure modes (see module docstring)."""
    if species_f1 is None or family_f1 is None:
        return "incomplete"
    if family_f1 >= solved:
        return "solved"
    if species_f1 < floor:
        return "label-ceiling"
    if (species_f1 - family_f1) >= gap:
        return "coverage-limited"
    return "moderate-flat"


def _quality(head: dict):
    """One higher-is-better quality score per head: F1 if present, else accuracy.

    Regression (rmse) heads return None — they are not comparable to F1.
    """
    m = head.get("metrics", {})
    if "f1" in m:
        return float(m["f1"])
    if "acc" in m:
        return float(m["acc"])
    return None


def load_split_means(runs_dir: str, prefix: str) -> dict:
    """{split: {trait: mean_quality_over_seeds}} for the chosen run prefix."""
    out: dict[str, dict[str, list]] = {s: defaultdict(list) for s in ("species", "genus", "family")}
    for split in out:
        for fp in sorted(glob.glob(os.path.join(runs_dir, f"{prefix}-{split}-s*.json"))):
            d = json.loads(open(fp).read())
            for trait, head in d.get("per_head", {}).items():
                q = _quality(head)
                if q is not None:
                    out[split][trait].append(q)
    return {split: {t: sum(v) / len(v) for t, v in traits.items() if v}
            for split, traits in out.items()}


def build_rows(means: dict) -> list[dict]:
    traits = sorted(set().union(*[set(means[s]) for s in means]))
    rows = []
    for trait in traits:
        sp = means["species"].get(trait)
        ge = means["genus"].get(trait)
        fa = means["family"].get(trait)
        decay = (sp - fa) if (sp is not None and fa is not None) else None
        rows.append({
            "trait": trait,
            "species_f1": sp, "genus_f1": ge, "family_f1": fa,
            "decay": decay,
            "mode": classify_failure_mode(sp, fa),
        })
    # order: coverage-limited (the story) first, then by decay desc
    order = {"coverage-limited": 0, "moderate-flat": 1, "label-ceiling": 2,
             "solved": 3, "incomplete": 4}
    rows.sort(key=lambda r: (order[r["mode"]], -(r["decay"] or -1)))
    return rows


def to_markdown(rows: list[dict]) -> str:
    counts = defaultdict(int)
    for r in rows:
        counts[r["mode"]] += 1
    lines = [
        "# Table 20 — Per-trait failure-mode classification",
        "",
        "Each trait's in-distribution ceiling (species-split F1) vs cross-clade "
        "generalization (family-split F1). `decay = species - family`. Modes: "
        "**solved** (family F1 ≥ 0.70), **label-ceiling** (species F1 < 0.30 — poor even "
        "in-distribution; a label/task problem, not the model), **coverage-limited** "
        "(learnable but collapses off-clade, decay ≥ 0.10 — needs training-clade coverage), "
        "**moderate-flat** (partial signal, no collapse, no ceiling).",
        "",
        "Summary: " + ", ".join(f"{counts[m]} {m}" for m in
                                ["solved", "coverage-limited", "moderate-flat", "label-ceiling"]
                                if counts[m]),
        "",
        "| Trait | Species F1 | Genus F1 | Family F1 | Decay | Failure mode |",
        "|---|---:|---:|---:|---:|:--|",
    ]
    def f(x):
        return "—" if x is None else f"{x:.3f}"
    for r in rows:
        lines.append(
            f"| `{r['trait']}` | {f(r['species_f1'])} | {f(r['genus_f1'])} | "
            f"{f(r['family_f1'])} | {f(r['decay'])} | {r['mode']} |"
        )
    return "\n".join(lines) + "\n"


def save_figure(rows: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for r in rows:
        if r["species_f1"] is None or r["decay"] is None:
            continue
        mode = r["mode"]
        ax.scatter(r["species_f1"], r["decay"], s=90, c=MODE_COLORS[mode],
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=mode if mode not in seen else None)
        seen.add(mode)
        ax.annotate(r["trait"], (r["species_f1"], r["decay"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(DECAY_GAP, ls="--", c="grey", lw=0.8)
    ax.axvline(FLOOR_SPECIES, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("In-distribution ceiling (species-split F1)")
    ax.set_ylabel("Cross-clade decay (species F1 − family F1)")
    ax.set_title("Trait failure modes: ceiling vs cross-clade decay")
    ax.legend(title="failure mode", loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--prefix", default="attnpool",
                    help="Run-file prefix; loads <prefix>-{species,genus,family}-s*.json")
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig", default="paper/figures/failure_modes.png")
    args = ap.parse_args()

    means = load_split_means(args.runs_dir, args.prefix)
    rows = build_rows(means)
    if not any(r["species_f1"] is not None for r in rows):
        raise SystemExit(f"no usable runs under {args.runs_dir} with prefix '{args.prefix}'")

    os.makedirs(args.out_dir, exist_ok=True)
    md = to_markdown(rows)
    open(os.path.join(args.out_dir, "20_failure_modes.md"), "w").write(md)
    import csv
    with open(os.path.join(args.out_dir, "20_failure_modes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trait", "species_f1", "genus_f1",
                                           "family_f1", "decay", "mode"])
        w.writeheader()
        w.writerows(rows)

    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    save_figure(rows, args.fig)
    print(md)
    print(f"wrote {args.out_dir}/20_failure_modes.md/.csv and {args.fig}")


if __name__ == "__main__":
    main()
