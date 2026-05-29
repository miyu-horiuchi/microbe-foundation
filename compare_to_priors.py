"""
compare_to_priors.py — produce a side-by-side table of microbe-foundation
results against published prior-work numbers.

Reads:
    <our_metrics_json>   from model.py --save-metrics
    prior_numbers.json   curated reference table of prior-work scores

Writes:
    paper/tables/06_vs_prior.md  side-by-side comparison table

The script is honest about comparability: each prior entry is tagged
with `directly_comparable`. The output table groups entries into
"comparable" (same trait + same metric type + similar evaluation
protocol) and "context" (cited for completeness but not apples-to-apples).

Usage:
    # After: python model.py --features data/esm2_features.npz --save-metrics runs/esm2.json
    python compare_to_priors.py --our runs/esm2.json

    # Compare multiple runs:
    python compare_to_priors.py --our runs/esm2.json --our runs/bacformer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRIORS_PATH = ROOT / "prior_numbers.json"
DEFAULT_OUT = ROOT / "paper" / "tables" / "06_vs_prior.md"


def fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "rmse":
        return f"{value:.3f}"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--our", type=Path, action="append", default=[],
                        help="JSON of our test metrics (model.py --save-metrics). Repeat for multiple runs.")
    parser.add_argument("--priors", type=Path, default=PRIORS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    priors = json.loads(args.priors.read_text())
    our_runs = []
    for p in args.our:
        if not p.exists():
            sys.exit(f"Our metrics file not found: {p}")
        our_runs.append(json.loads(p.read_text()))

    # All traits known either to us or to priors
    all_traits = set(priors.get("comparators", {}).keys())
    for run in our_runs:
        all_traits.update(run.get("per_head", {}).keys())

    out = ["# Table 6 — microbe-foundation vs. Prior Work", ""]
    out.append("_Comparators curated in `prior_numbers.json`; comparability flagged per row._")
    out.append("")

    # === Section A: directly comparable rows ===
    out.append("## A. Directly comparable")
    out.append("")
    header_cols = ["Trait", "Metric"] + [run.get("run_name", f"run{i}") for i, run in enumerate(our_runs)] + [
        "Prior method", "Prior score", "Cite"
    ]
    out.append("| " + " | ".join(header_cols) + " |")
    out.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    comparable_count = 0
    for trait in sorted(all_traits):
        for prior in priors.get("comparators", {}).get(trait, []):
            if not prior.get("directly_comparable"):
                continue
            row = [f"`{trait}`", prior["metric_kind"]]
            for run in our_runs:
                head = run.get("per_head", {}).get(trait)
                if not head:
                    row.append("—")
                    continue
                # Prefer new `metrics` dict (multi-metric output) if present;
                # fall back to legacy single-metric `metric_kind`/`score`.
                metrics = head.get("metrics") or {head["metric_kind"]: head["score"]}
                if prior["metric_kind"] in metrics:
                    row.append(fmt(metrics[prior["metric_kind"]], prior["metric_kind"]))
                else:
                    row.append("—")
            row.append(prior["method"])
            row.append(fmt(prior.get("score"), prior["metric_kind"]))
            cite = prior.get("cite", "")
            row.append(f"[@{cite}]" if cite else "—")
            out.append("| " + " | ".join(row) + " |")
            comparable_count += 1
    if comparable_count == 0:
        out.append("| _(no directly comparable rows — fill prior_numbers.json scores)_ | | | | | |")

    # === Section B: context-only ===
    out.append("")
    out.append("## B. Context (different metric / evaluation — not apples-to-apples)")
    out.append("")
    out.append("| Trait | Prior method | Metric | Score | Cite | Notes |")
    out.append("|---|---|---|---|---|---|")
    for trait in sorted(all_traits):
        for prior in priors.get("comparators", {}).get(trait, []):
            if prior.get("directly_comparable"):
                continue
            cite = prior.get("cite", "")
            cite_s = f"[@{cite}]" if cite else "—"
            notes = (prior.get("notes") or "").replace("|", "\\|")
            out.append(
                f"| `{trait}` | {prior['method']} | {prior['metric_kind']} | "
                f"{fmt(prior.get('score'), prior['metric_kind'])} | {cite_s} | {notes} |"
            )

    # === Section C: white-space (traits with no prior comparator) ===
    out.append("")
    out.append("## C. Literature white-space (no prior comparator)")
    out.append("")
    out.append("| Trait | Status |")
    out.append("|---|---|")
    for trait in sorted(all_traits):
        comparators = priors.get("comparators", {}).get(trait, [])
        if not comparators:
            out.append(f"| `{trait}` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |")
        else:
            real = [c for c in comparators if c.get("cite") and c["cite"]]
            if not real:
                marker = comparators[0].get("notes", "")
                out.append(f"| `{trait}` | {marker} |")

    out.append("")
    out.append(f"_{len(our_runs)} of our runs compared against {len(priors.get('comparators', {}))} prior-trait entries._")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out.relative_to(ROOT)}  ({len(out)} lines)")


if __name__ == "__main__":
    main()
