"""
aggregate_seeds.py — collapse per-seed model.py metric JSONs into mean +/- 95% CI
tables for the manuscript.

Tier-1 venues expect headline numbers reported over multiple seeds with
confidence intervals, not single runs. This reads every `*.json` written by
`model.py --save-metrics`, groups runs that share (pooling, split_level,
balanced_families, trait, metric), and reports mean, std, n, and a 95% CI whose
half-width uses the Student-t critical value for n-1 degrees of freedom (falls
back to the normal 1.96 for large n).

Usage:
    python paper/aggregate_seeds.py --runs-dir runs/tier1 --out paper/tables
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# Two-sided 95% Student-t critical values, indexed by degrees of freedom (n-1).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000,
}


def t95(dof: int) -> float:
    """95% two-sided t critical value for `dof` degrees of freedom."""
    if dof <= 0:
        return float("nan")
    if dof in _T95:
        return _T95[dof]
    keys = sorted(_T95)
    if dof > keys[-1]:
        return 1.96
    # nearest tabulated dof at or above (conservative)
    for k in keys:
        if k >= dof:
            return _T95[k]
    return 1.96


def mean_std_ci(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "n": 1, "ci95": float("nan"),
                "lo": mean, "hi": mean}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    ci = t95(n - 1) * sem
    return {"mean": mean, "std": std, "n": n, "ci95": ci,
            "lo": mean - ci, "hi": mean + ci}


def collect(runs_dir: Path) -> dict[tuple, list[float]]:
    """Map (pooling, split, balanced, trait, metric_kind) -> list of scores."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json metric files in {runs_dir}")
    for fp in files:
        d = json.loads(fp.read_text())
        pooling = d.get("pooling", "?")
        split = d.get("split_level", "?")
        balanced = bool(d.get("balanced_families", False))
        for trait, h in d.get("per_head", {}).items():
            kind = h.get("metric_kind")
            score = h.get("score")
            if kind is None or score is None:
                continue
            groups[(pooling, split, balanced, trait, kind)].append(float(score))
    return groups


def aggregate(groups: dict[tuple, list[float]]) -> list[dict]:
    rows = []
    for (pooling, split, balanced, trait, kind), vals in sorted(groups.items()):
        stats = mean_std_ci(vals)
        rows.append({
            "pooling": pooling, "split": split, "balanced_families": balanced,
            "trait": trait, "metric": kind, **stats,
        })
    return rows


def macro_rows(rows: list[dict]) -> list[dict]:
    """Per-(pooling, split, balanced) macro-average of each trait's mean score.

    Regression heads (rmse, lower-is-better) are excluded so the macro number is a
    comparable higher-is-better quality score.
    """
    by_cfg: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["metric"] == "rmse":
            continue
        by_cfg[(r["pooling"], r["split"], r["balanced_families"])].append(r["mean"])
    out = []
    for (pooling, split, balanced), means in sorted(by_cfg.items()):
        stats = mean_std_ci(means)
        out.append({
            "pooling": pooling, "split": split, "balanced_families": balanced,
            "trait": "MACRO (mean over traits)", "metric": "score",
            **stats,
        })
    return out


def to_markdown(rows: list[dict], macro: list[dict]) -> str:
    lines = [
        "## Per-seed aggregated metrics (mean ± 95% CI)",
        "",
        "| pooling | split | balanced | trait | metric | mean | 95% CI | std | n |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in macro + rows:
        ci = "" if math.isnan(r["ci95"]) else f"±{r['ci95']:.4f}"
        lines.append(
            f"| {r['pooling']} | {r['split']} | {'yes' if r['balanced_families'] else 'no'} "
            f"| {r['trait']} | {r['metric']} | {r['mean']:.4f} | {ci} | {r['std']:.4f} | {r['n']} |"
        )
    return "\n".join(lines) + "\n"


def to_csv(rows: list[dict]) -> str:
    cols = ["pooling", "split", "balanced_families", "trait", "metric",
            "mean", "std", "ci95", "lo", "hi", "n"]
    out = [",".join(cols)]
    for r in rows:
        out.append(",".join(str(r[c]) for c in cols))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/tier1"))
    ap.add_argument("--out", type=Path, default=Path("paper/tables"))
    args = ap.parse_args()

    groups = collect(args.runs_dir)
    rows = aggregate(groups)
    macro = macro_rows(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, macro)
    (args.out / "18_seed_aggregated.md").write_text(md)
    (args.out / "18_seed_aggregated.csv").write_text(to_csv(macro + rows))
    print(md)
    print(f"wrote {args.out / '18_seed_aggregated.md'} and .csv")


if __name__ == "__main__":
    main()
