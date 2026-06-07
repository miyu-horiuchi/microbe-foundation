from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
OUT = ROOT / "spaces" / "research_showcase" / "assets" / "predictability_gradient.json"

COMPOSITIONAL = [
    "gram_stain",
    "cell_shape",
    "motility",
    "sporulation",
    "oxygen_tolerance",
    "catalase",
    "cytochrome_oxidase",
    "temperature_class",
    "ph_class",
    "halophily",
    "pigmentation",
]

MACHINERY = [
    "pathogenicity_human",
    "pathogenicity_animal",
    "cultivation_medium",
    "carbon_utilization",
    "metabolite_production",
    "amr_phenotype",
    "biosafety_level",
    "fatty_acid_profile",
]

HEADLINE_GRADIENT = [
    {"split": "species", "class": "compositional", "delta_f1": 0.021, "std": 0.002},
    {"split": "species", "class": "machinery", "delta_f1": 0.083, "std": 0.012},
    {"split": "genus", "class": "compositional", "delta_f1": 0.016, "std": 0.004},
    {"split": "genus", "class": "machinery", "delta_f1": 0.067, "std": 0.010},
    {"split": "family", "class": "compositional", "delta_f1": 0.009, "std": 0.002},
    {"split": "family", "class": "machinery", "delta_f1": 0.010, "std": 0.003},
]


def load_run(name: str) -> dict | None:
    path = RUNS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def head_f1(run: dict, head: str) -> float | None:
    entry = run.get("per_head", {}).get(head)
    if not entry:
        return None
    metrics = entry.get("metrics", {})
    if "f1" in metrics:
        return float(metrics["f1"])
    if entry.get("metric_kind") == "f1":
        return float(entry["score"])
    return None


def seeded_head_deltas() -> list[dict]:
    rows: list[dict] = []
    class_of = {head: "compositional" for head in COMPOSITIONAL}
    class_of.update({head: "machinery" for head in MACHINERY})
    for split in ["species", "genus", "family"]:
        for head, trait_class in class_of.items():
            vals = []
            for seed in [1, 2, 3]:
                mean_run = load_run(f"meanpool-{split}-s{seed}")
                attn_run = load_run(f"attnpool-{split}-s{seed}")
                if not mean_run or not attn_run:
                    continue
                m = head_f1(mean_run, head)
                a = head_f1(attn_run, head)
                if m is None or a is None:
                    continue
                vals.append(a - m)
            if vals:
                rows.append(
                    {
                        "split": split,
                        "head": head,
                        "class": trait_class,
                        "delta_f1_mean": mean(vals),
                        "delta_f1_std": pstdev(vals) if len(vals) > 1 else 0.0,
                        "n_seeds": len(vals),
                    }
                )
    return sorted(rows, key=lambda r: r["delta_f1_mean"], reverse=True)


def best_rows(limit: int = 12) -> list[dict]:
    path = ROOT / "paper" / "tables" / "09_full_comparison.md"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        trait, metric, best, run, prior, prior_score, verdict = parts[:7]
        rows.append(
            {
                "trait": trait.strip("`"),
                "metric": metric,
                "best_ours": best.replace("**", ""),
                "run": run.strip("`"),
                "prior": prior,
                "prior_score": prior_score,
                "verdict": verdict,
            }
        )
    priority = {"⬆": 0, "🆕": 1, "~": 2, "⬇": 3}
    rows.sort(key=lambda r: (priority.get(r["verdict"][:1], 9), r["trait"]))
    return rows[:limit]


def main() -> None:
    asset = {
        "paper": {
            "title": "When Does Attention Help? A Predictability Gradient for Genomic Trait Prediction",
            "date": "2026-06-05",
            "n_genomes": 19592,
            "n_proteins": 82000000,
            "n_traits": 21,
            "embedding": "ESM-2 t30 150M, 640-d per protein",
        },
        "trait_classes": {
            "compositional": COMPOSITIONAL,
            "machinery": MACHINERY,
            "excluded": ["isolation_source", "country"],
        },
        "headline_gradient": HEADLINE_GRADIENT,
        "head_deltas": seeded_head_deltas(),
        "attention": {
            "animal": {
                "head": "pathogenicity_animal",
                "auroc": 0.88,
                "median_entropy": 0.26,
                "top5_attention_mass": 0.81,
                "within_top": 0.281,
                "within_random": 0.059,
                "within_p": "2.5e-7",
                "between_top_pathogenic": 0.281,
                "between_top_non_pathogenic": 0.109,
                "between_or": 3.2,
                "between_p": "6.8e-14",
                "ablation_flip": 0.227,
                "ablation_p": "1.2e-4",
            },
            "human": {
                "head": "pathogenicity_human",
                "auroc": 0.85,
                "between_or": 3.1,
                "between_p": "3.8e-5",
                "ablation_flip": 0.125,
                "ablation_p": "9e-3",
            },
            "genes": ["papC", "mrkC", "fhaB", "ail", "pilQ", "fliR"],
        },
        "comparison_rows": best_rows(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(asset, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

