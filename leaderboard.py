"""
leaderboard.py — aggregate runs/*.json into a per-head leaderboard table.

Reads:  runs/*.json   (each from `model.py --save-metrics`)
Writes: paper/tables/07_leaderboard.md

Sorting: each head ranks higher = better, except RMSE (lower = better).
Overall rank is computed as the mean per-head rank across all heads each
method has a score for.

Usage:
    python leaderboard.py                          # uses runs/*.json
    python leaderboard.py --runs-dir external_runs # alternate dir
    python leaderboard.py --out paper/tables/07_leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = ROOT / "runs"
DEFAULT_OUT = ROOT / "paper" / "tables" / "07_leaderboard.md"


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            runs.append({"_path": p, **json.loads(p.read_text())})
        except (json.JSONDecodeError, OSError):
            print(f"  [skip] could not parse {p}", file=sys.stderr)
    return runs


def fmt_score(score: float | None, kind: str) -> str:
    if score is None:
        return "—"
    if kind == "rmse":
        return f"{score:.3f}"
    return f"{score:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.runs_dir.exists():
        sys.exit(f"runs dir not found: {args.runs_dir}")
    runs = load_runs(args.runs_dir)
    if not runs:
        sys.exit(f"no run JSONs in {args.runs_dir}")

    print(f"loaded {len(runs)} runs from {args.runs_dir}")

    # All heads across all runs
    all_heads: set[str] = set()
    for r in runs:
        all_heads.update(r.get("per_head", {}).keys())
    heads = sorted(all_heads)

    # Compute per-head ranks (1 = best). Higher is better for acc/f1; lower for rmse.
    ranks: dict[str, dict[str, int]] = {}  # run_name -> head -> rank
    for head in heads:
        # Gather (run_name, score, kind) tuples
        scored = []
        for r in runs:
            entry = r.get("per_head", {}).get(head)
            if entry is None:
                continue
            scored.append((r.get("run_name", r["_path"].stem), entry["score"], entry["metric_kind"]))
        if not scored:
            continue
        # Sort: rmse ascending, everything else descending
        kind = scored[0][2]
        reverse = kind != "rmse"
        scored.sort(key=lambda x: x[1], reverse=reverse)
        for rank, (name, _, _) in enumerate(scored, start=1):
            ranks.setdefault(name, {})[head] = rank

    # Overall mean-rank per run
    overall: dict[str, float] = {}
    for name, head_ranks in ranks.items():
        overall[name] = sum(head_ranks.values()) / max(len(head_ranks), 1)
    ranked_runs = sorted(runs, key=lambda r: overall.get(r.get("run_name", r["_path"].stem), 999))

    # Build the table
    out = ["# Table 7 — microbe-foundation Leaderboard", ""]
    out.append(f"_{len(runs)} submitted runs, ranked by mean per-head rank (lower = better)._")
    out.append("")
    out.append(f"Family-held-out splits unless otherwise noted in the run's `split_level` field.")
    out.append("")

    # Summary table
    out.append("## Overall ranking")
    out.append("")
    out.append("| Rank | Run | Heads scored | Mean rank | Split | Feature dim | Params |")
    out.append("|---:|---|---:|---:|---|---:|---:|")
    for i, r in enumerate(ranked_runs, start=1):
        name = r.get("run_name", r["_path"].stem)
        n_heads = len(r.get("per_head", {}))
        mean_rank = overall.get(name, float("nan"))
        split = r.get("split_level", "?")
        feat_dim = r.get("feature_dim", "?")
        n_params = r.get("n_params", "?")
        out.append(
            f"| {i} | `{name}` | {n_heads} | {mean_rank:.2f} | {split} | "
            f"{feat_dim} | {n_params:,} |" if isinstance(n_params, int) else
            f"| {i} | `{name}` | {n_heads} | {mean_rank:.2f} | {split} | {feat_dim} | {n_params} |"
        )

    # Per-head detail
    out.append("")
    out.append("## Per-head scores")
    out.append("")
    header = "| Head | Metric | " + " | ".join(f"`{r.get('run_name', r['_path'].stem)}`" for r in ranked_runs) + " |"
    sep = "|---|---|" + "---:|" * len(ranked_runs)
    out.append(header)
    out.append(sep)

    for head in heads:
        # Find metric kind from any run that has it
        kind = None
        for r in runs:
            entry = r.get("per_head", {}).get(head)
            if entry:
                kind = entry["metric_kind"]
                break
        if kind is None:
            continue
        cells = [f"`{head}`", kind]
        # Find the best score for highlighting
        best_score = None
        for r in ranked_runs:
            entry = r.get("per_head", {}).get(head)
            if entry is None:
                continue
            s = entry["score"]
            if best_score is None:
                best_score = s
            elif kind == "rmse":
                best_score = min(best_score, s)
            else:
                best_score = max(best_score, s)
        for r in ranked_runs:
            entry = r.get("per_head", {}).get(head)
            if entry is None:
                cells.append("—")
            else:
                s = entry["score"]
                s_fmt = fmt_score(s, kind)
                # Bold the best score per row
                if best_score is not None and abs(s - best_score) < 1e-9:
                    cells.append(f"**{s_fmt}**")
                else:
                    cells.append(s_fmt)
        out.append("| " + " | ".join(cells) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out.relative_to(ROOT)}")
    for i, r in enumerate(ranked_runs[:5], start=1):
        name = r.get("run_name", r["_path"].stem)
        print(f"  {i}. {name}  (mean rank {overall.get(name, 0):.2f})")


if __name__ == "__main__":
    main()
