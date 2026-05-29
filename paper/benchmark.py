"""
Build paper/tables/08_benchmark.md — a single comparison table that:

1. Quantifies improvement: chance / random-features / our model
2. Quantifies novelty: how many traits have published prior comparators,
   where we beat / tie / lose, where we set the first baseline.

Run after at least the following runs exist in runs/:
    chance-family.json
    random640-family.json
    esm2-150M-family-sel.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"
PRIORS = ROOT / "prior_numbers.json"


def load(name: str) -> dict:
    return json.loads((RUNS / f"{name}.json").read_text())


def get_metric(run: dict, head: str, metric: str) -> float | None:
    h = run.get("per_head", {}).get(head)
    if not h:
        return None
    m = h.get("metrics") or {h.get("metric_kind"): h.get("score")}
    return m.get(metric)


def best_prior(priors: dict, head: str, metric: str) -> tuple[float, str] | None:
    for p in priors.get("comparators", {}).get(head, []):
        if not p.get("directly_comparable"):
            continue
        if p.get("metric_kind") == metric and p.get("score") is not None:
            return float(p["score"]), p.get("method", "")
    return None


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def main() -> None:
    chance = load("chance-family")
    random640 = load("random640-family")
    ours = load("esm2-150M-family-sel")
    priors = json.loads(PRIORS.read_text())

    heads = sorted(set(chance["per_head"].keys()) | set(ours["per_head"].keys()))

    out: list[str] = []
    out.append("# Table 8 — Benchmark: chance / random / ESM-2 / prior")
    out.append("")
    out.append("Family-held-out split. F1 is the honest metric on imbalanced heads; "
               "accuracy can be misleading when one class dominates (see `pathogenicity_*`). "
               "All comparators reported here are directly comparable in metric and "
               "evaluation protocol (see `paper/tables/06_vs_prior.md` for the full list).")
    out.append("")
    out.append("Legend:  **bold** = best non-chance row;  ⬆ = our model beats prior;  "
               "⬇ = our model loses to prior;  🆕 = no published prior comparator")
    out.append("")

    out.append("| Trait | Metric | Chance | Random-640 | **ESM-2 150M (ours)** | Prior best | Verdict |")
    out.append("|---|---|---:|---:|---:|---:|:---:|")

    # decide per trait which metric to surface
    metric_pref = {
        # heavily imbalanced binaries: report F1 (acc is misleading)
        "pathogenicity_human": "f1",
        "pathogenicity_animal": "f1",
        "sporulation": "f1",
        "motility": "f1",
        # multiclass with class imbalance: F1
        "gram_stain": "f1",
        "temperature_class": "f1",
        "oxygen_tolerance": "f1",
        "cell_shape": "f1",
        # multilabel: f1 sample by default
        "cultivation_medium": "f1",
        "carbon_utilization": "f1",
        "metabolite_production": "f1_macro",
        "amr_phenotype": "f1_macro",
    }
    # default: pick whichever is in our metrics dict
    summary = {"beat_prior": 0, "lose_prior": 0, "no_prior": 0, "n_total": 0}

    def any_prior_metric(head: str) -> str | None:
        """Return the metric_kind of the first directly-comparable prior, or None."""
        for p in priors.get("comparators", {}).get(head, []):
            if p.get("directly_comparable") and p.get("score") is not None:
                return p.get("metric_kind")
        return None

    for h in heads:
        head_type = chance["per_head"].get(h, {}).get("head_type") or \
                    ours["per_head"].get(h, {}).get("head_type")
        if head_type == "regression_vector":
            continue
        # If a prior exists for this trait, use its metric for fair comparison;
        # otherwise fall back to the metric we'd report anyway.
        prior_metric = any_prior_metric(h)
        metric = prior_metric or metric_pref.get(h, "f1" if head_type == "multilabel" else "acc")
        c = get_metric(chance, h, metric)
        r = get_metric(random640, h, metric)
        o = get_metric(ours, h, metric)
        p = best_prior(priors, h, metric)
        verdict = ""
        if p is None:
            verdict = "🆕"
            summary["no_prior"] += 1
        else:
            if o is not None and o > p[0]:
                verdict = f"⬆ ({p[1]})"
                summary["beat_prior"] += 1
            else:
                verdict = f"⬇ ({p[1]})"
                summary["lose_prior"] += 1
        summary["n_total"] += 1
        out.append(f"| `{h}` | {metric} | {fmt(c)} | {fmt(r)} | **{fmt(o)}** | "
                   f"{fmt(p[0] if p else None)} | {verdict} |")

    out.append("")
    out.append("## Headline counters")
    out.append("")
    out.append(f"- Traits evaluated: **{summary['n_total']}**")
    out.append(f"- Where a directly-comparable prior exists: **{summary['beat_prior'] + summary['lose_prior']}**")
    out.append(f"  - We **beat** prior: **{summary['beat_prior']}**")
    out.append(f"  - We **lose** to prior: **{summary['lose_prior']}**")
    out.append(f"- Traits where we set the **first published baseline** ('white-space'): "
               f"**{summary['no_prior']}** "
               f"({100 * summary['no_prior'] / summary['n_total']:.0f}% of evaluated traits)")
    out.append("")
    out.append("## Aggregate improvement (mean across heads, where the same metric exists across rows)")

    # aggregate
    def mean_metric(run, metric):
        vals = []
        for h in heads:
            v = get_metric(run, h, metric)
            if v is not None:
                vals.append(v)
        return sum(vals) / max(len(vals), 1)

    out.append("")
    out.append("| Metric | Chance | Random-640 | **ESM-2 150M (ours)** | Δ vs chance | Δ vs random |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for mk in ("acc", "f1", "f1_macro"):
        c, r, o = mean_metric(chance, mk), mean_metric(random640, mk), mean_metric(ours, mk)
        if o == 0:
            continue
        out.append(
            f"| {mk} (mean across heads) | {c:.3f} | {r:.3f} | **{o:.3f}** | "
            f"{o - c:+.3f} | {o - r:+.3f} |"
        )

    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append("- **Chance** = always predict the train-set majority class on test. Sets the "
               "no-signal floor.")
    out.append("- **Random-640** = our same downstream head trained on 640-dim Gaussian noise "
               "instead of ESM-2 embeddings. Sets the 'head capacity alone' floor.")
    out.append("- **ESM-2 150M (ours)** = `compute_esm2_features_mp.py` features + multi-task "
               "head with selective class weighting (`--class-weights --imbalance-threshold 5`).")
    out.append("- F1 — not accuracy — is the metric to read on imbalanced heads "
               "(`pathogenicity_*`, `sporulation`, `temperature_class`). Random-640 can match "
               "ESM-2 on accuracy there *just by predicting the majority class*. F1 separates "
               "real signal from class-imbalance gaming.")
    out.append("")

    out_path = ROOT / "paper" / "tables" / "08_benchmark.md"
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path}")
    print()
    print(f"  traits evaluated  : {summary['n_total']}")
    print(f"  beat prior        : {summary['beat_prior']}")
    print(f"  lose to prior     : {summary['lose_prior']}")
    print(f"  white-space (new) : {summary['no_prior']}")


if __name__ == "__main__":
    main()
