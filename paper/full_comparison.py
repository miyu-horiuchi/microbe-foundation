"""
Build paper/tables/09_full_comparison.md.

Covers every trait (incl. fatty_acid_profile regression head), our BEST score
across all 12 model variants in runs/ for each metric the trait emits, AND
every directly-comparable prior — not just the best one — from prior_numbers.json.

Verdict logic:
  ⬆  : our best > prior_best on the prior's reported metric
  ⬇  : our best < prior_best on the prior's reported metric
  ~  : prior has no reported score (TBD)
  🆕 : no directly-comparable prior published
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"
PRIORS = ROOT / "prior_numbers.json"


def all_runs() -> list[dict]:
    out = []
    for f in sorted(RUNS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if "per_head" in d:
                d["_path"] = f.name
                out.append(d)
        except Exception:
            pass
    return out


def metric_of(head: dict, metric: str) -> float | None:
    """Return the score for `metric` from this head, considering both the new
    `metrics` dict and the legacy `metric_kind`/`score` fallback."""
    metrics = head.get("metrics") or {head.get("metric_kind"): head.get("score")}
    return metrics.get(metric)


def best_across_runs(runs: list[dict], trait: str, metric: str, lower_is_better: bool):
    best = None
    best_run = None
    for r in runs:
        h = r.get("per_head", {}).get(trait)
        if not h:
            continue
        v = metric_of(h, metric)
        if v is None:
            continue
        if best is None or (lower_is_better and v < best) or (not lower_is_better and v > best):
            best, best_run = v, r.get("run_name", r["_path"])
    return best, best_run


def fmt(x):
    if x is None:
        return "—"
    return f"{x:.3f}"


def main():
    runs = all_runs()
    priors = json.loads(PRIORS.read_text())
    # Collect all traits we know about, from runs + priors
    traits = set()
    for r in runs:
        traits.update(r.get("per_head", {}).keys())
    traits.update(priors.get("comparators", {}).keys())
    traits = sorted(traits)

    lines = []
    lines.append("# Table 9 — Full per-trait comparison")
    lines.append("")
    lines.append(f"Best score across all {len(runs)} model variants in `runs/` "
                 "for each metric the head emits, vs every directly-comparable "
                 "prior published number. RMSE is lower-is-better; everything "
                 "else is higher-is-better.")
    lines.append("")
    lines.append("Legend: **⬆** = we beat the listed prior · "
                 "**⬇** = we lose to the listed prior · "
                 "**~** = prior method exists but no score published · "
                 "**🆕** = no directly-comparable prior exists")
    lines.append("")
    lines.append("| Trait | Metric | Best ours | (from run) | Prior method | Prior score | Verdict |")
    lines.append("|---|---|---:|---|---|---:|:---:|")

    summary = {"beat": 0, "lose": 0, "tbd_prior": 0, "no_prior": 0, "n_rows": 0}
    seen_traits_with_prior = set()
    seen_traits_with_beat = set()

    for trait in traits:
        comparators = priors.get("comparators", {}).get(trait, [])
        direct = [c for c in comparators if c.get("directly_comparable")]

        if not direct:
            # No prior — emit one row per metric the trait emits, with 🆕 verdict
            for metric in ("acc", "f1", "f1_macro", "rmse"):
                lower = metric == "rmse"
                v, run_name = best_across_runs(runs, trait, metric, lower)
                if v is None:
                    continue
                lines.append(
                    f"| `{trait}` | {metric} | **{fmt(v)}** | `{run_name}` | — | — | 🆕 |"
                )
                summary["n_rows"] += 1
                summary["no_prior"] += 1
            continue

        seen_traits_with_prior.add(trait)
        for c in direct:
            metric = c.get("metric_kind", "")
            prior_score = c.get("score")
            method = c.get("method", "")
            lower = metric == "rmse"
            v, run_name = best_across_runs(runs, trait, metric, lower)
            if v is None:
                # We don't track this metric — note as gap
                lines.append(
                    f"| `{trait}` | {metric} | — | _no run_ | {method} | "
                    f"{fmt(prior_score)} | (metric not tracked) |"
                )
                continue
            if prior_score is None:
                verdict = "~"
                summary["tbd_prior"] += 1
            elif lower:
                verdict = "⬆" if v < prior_score else "⬇"
                if v < prior_score:
                    summary["beat"] += 1
                    seen_traits_with_beat.add(trait)
                else:
                    summary["lose"] += 1
            else:
                verdict = "⬆" if v > prior_score else "⬇"
                if v > prior_score:
                    summary["beat"] += 1
                    seen_traits_with_beat.add(trait)
                else:
                    summary["lose"] += 1
            lines.append(
                f"| `{trait}` | {metric} | **{fmt(v)}** | `{run_name}` | "
                f"{method} | {fmt(prior_score)} | {verdict} |"
            )
            summary["n_rows"] += 1

    lines.append("")
    lines.append("## Summary counters")
    lines.append("")
    lines.append(f"- Total comparison rows: **{summary['n_rows']}**")
    lines.append(f"- Rows where we beat the prior (⬆): **{summary['beat']}** "
                 f"({len(seen_traits_with_beat)} distinct traits)")
    lines.append(f"- Rows where we lose to the prior (⬇): **{summary['lose']}**")
    lines.append(f"- Rows where prior exists but no published score (~): "
                 f"**{summary['tbd_prior']}**")
    lines.append(f"- Rows where no directly-comparable prior exists (🆕): "
                 f"**{summary['no_prior']}**")
    lines.append("")
    n_traits = len(traits)
    n_priored = len(seen_traits_with_prior)
    n_whitespace = n_traits - n_priored
    lines.append(f"- Distinct traits in benchmark: **{n_traits}**")
    lines.append(f"  - with at least one directly-comparable prior: **{n_priored}** "
                 f"({100 * n_priored / n_traits:.0f}%)")
    lines.append(f"  - **white-space** (no direct prior published): **{n_whitespace}** "
                 f"({100 * n_whitespace / n_traits:.0f}%)")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- 'Best ours' picks the strongest score across all configurations in "
                 "`runs/` for that trait+metric. Different traits' best runs may come "
                 "from different model variants (selective class weights, h1024 head, "
                 "different split level, etc.) — see the run name column.")
    lines.append("- Multilabel heads can report both `f1` (sample-averaged) and "
                 "`f1_macro` (label-averaged). Both rows are emitted when a prior "
                 "reports the corresponding metric.")
    lines.append("- Numbers marked '~' indicate the prior method exists in the "
                 "literature but the published paper does not report the metric we "
                 "need for a head-to-head comparison.")
    lines.append("")

    out_path = ROOT / "paper" / "tables" / "09_full_comparison.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print()
    print(f"  rows               : {summary['n_rows']}")
    print(f"  beats              : {summary['beat']}  (across {len(seen_traits_with_beat)} traits)")
    print(f"  losses             : {summary['lose']}")
    print(f"  prior-w/o-score    : {summary['tbd_prior']}")
    print(f"  no-prior (white-space rows): {summary['no_prior']}")
    print(f"  total distinct traits: {n_traits}  (white-space {n_whitespace})")


if __name__ == "__main__":
    main()
