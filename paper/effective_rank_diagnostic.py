#!/usr/bin/env python3
"""Effective-rank diagnostic of the learned pooling/encoder weights.

Motivation
----------
On the family (hardest cross-clade) split we observed seed-level instability for
the high-capacity ``set_transformer`` pooler. The spectral-structure literature
(9D Labs: "backward bottleneck", "spectral PAC-Bayes") argues that a layer's
*effective rank* -- the entropy of its singular-value spectrum -- is a cheap,
data-free proxy for how much of its nominal capacity it actually uses, and that
weight-decay + backprop drive over-parameterised layers toward low effective
rank. If ``set_transformer`` is collapsing on the family split, its pooler
weights should show a lower / more erratic effective rank there than on the
easier species/genus splits, while ``mean`` (no params) and the simpler
``attention`` pooler stay flat.

This script computes, directly from the saved checkpoints (no embeddings, no
GPU, no forward pass), for every 2-D weight matrix W:

    sigma            = singular values of W
    p_k              = sigma_k^2 / sum_j sigma_j^2          (spectral energy)
    H                = -sum_k p_k log p_k                   (Shannon entropy, nats)
    effective_rank   = exp(H)                               (Roy & Vetterli, 2007)
    rank_fraction    = effective_rank / min(rows, cols)     (0..1, size-normalised)
    stable_rank      = sum_j sigma_j^2 / sigma_max^2        (||W||_F^2 / ||W||_2^2)

Layers are bucketed into ``pool`` / ``encoder`` / ``head`` by key prefix. The
encoder layers have identical shapes across all poolers, so their effective rank
is directly comparable pooler-to-pooler; the pooler block carries the capacity
that differs between architectures.

Outputs
-------
    paper/tables/19_effective_rank_perlayer.csv   one row per (checkpoint, weight)
    paper/tables/19_effective_rank_summary.csv    aggregated by pooling x split
    paper/tables/19_effective_rank_summary.md     same, human-readable
    paper/figures/19_effective_rank.{png,pdf}     pooler + encoder rank vs split

Usage
-----
    python3 paper/effective_rank_diagnostic.py --runs-dir runs/tier1
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

SPLIT_ORDER = {"species": 0, "genus": 1, "family": 2}


def module_of(key: str) -> str:
    """Coarse bucket for a state-dict key."""
    head = key.split(".", 1)[0]
    if head == "pool":
        return "pool"
    if head == "encoder":
        return "encoder"
    if head == "heads":
        return "head"
    return head


def spectral_stats(w: torch.Tensor) -> tuple[float, float, float, float]:
    """Return (effective_rank, rank_fraction, stable_rank, sigma_max)."""
    # float64 on CPU for a numerically stable spectrum.
    sv = torch.linalg.svdvals(w.detach().to(torch.float64))
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    energy = sv.pow(2)
    total = energy.sum()
    p = energy / total
    # entropy in nats; guard p>0 already ensured.
    entropy = -(p * p.log()).sum()
    eff_rank = float(torch.exp(entropy))
    min_dim = min(w.shape[0], w.shape[1])
    stable_rank = float(total / energy.max())
    return eff_rank, eff_rank / min_dim, stable_rank, float(sv.max())


def label_balanced(path: Path, meta: dict) -> bool:
    return "balancedfamilies" in path.stem or bool(meta.get("balanced_families"))


def cell_key(pooling: str, split: str, balanced: bool) -> str:
    suffix = " (bal)" if balanced else ""
    return f"{pooling}/{split}{suffix}"


def mean_range(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return math.nan, math.nan, math.nan
    m = sum(xs) / len(xs)
    return m, min(xs), max(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/tier1"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper"))
    ap.add_argument("--glob", default="*.pt")
    ap.add_argument("--min-epochs", type=int, default=40,
                    help="Skip checkpoints whose sibling .json reports fewer "
                         "epochs (excludes smoke-test runs whose near-random "
                         "weights have artificially high effective rank).")
    args = ap.parse_args()

    all_ckpts = sorted(args.runs_dir.glob(args.glob))
    if not all_ckpts:
        raise SystemExit(f"no checkpoints matched {args.runs_dir}/{args.glob}")

    import json as _json
    ckpts = []
    for fp in all_ckpts:
        sib = fp.with_suffix(".json")
        ep = None
        if sib.exists():
            try:
                ep = _json.loads(sib.read_text()).get("epochs")
            except Exception:
                ep = None
        if ep is not None and ep < args.min_epochs:
            print(f"[skip] {fp.name}: epochs={ep} < {args.min_epochs} "
                  f"(undertrained -- effective rank not comparable)")
            continue
        ckpts.append(fp)
    if not ckpts:
        raise SystemExit("all checkpoints filtered out by --min-epochs")

    tables = args.out_dir / "tables"
    figs = args.out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    perlayer_rows: list[dict] = []
    # cell -> module -> list of (seed, eff_rank, rank_fraction, stable_rank) per matrix
    # We aggregate pool and encoder separately.
    # pool: mean over all pool.* matrices within a checkpoint (per-seed scalar).
    # encoder: mean over encoder.{0,3}.weight within a checkpoint (per-seed scalar).
    per_ckpt: dict[str, dict] = {}

    for fp in ckpts:
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        pooling = str(ck.get("pooling", "?"))
        split = str(ck.get("split_level", "?"))
        seed = ck.get("seed", -1)
        balanced = label_balanced(fp, ck)

        pool_eff, pool_frac, enc_frac, enc_eff = [], [], [], []
        for key, w in sd.items():
            if not (hasattr(w, "ndim") and w.ndim == 2):
                continue
            mod = module_of(key)
            eff, frac, stab, smax = spectral_stats(w)
            perlayer_rows.append({
                "file": fp.name, "pooling": pooling, "split": split,
                "seed": seed, "balanced": int(balanced), "module": mod, "key": key,
                "rows": w.shape[0], "cols": w.shape[1],
                "min_dim": min(w.shape), "effective_rank": round(eff, 4),
                "rank_fraction": round(frac, 4), "stable_rank": round(stab, 4),
                "sigma_max": round(smax, 4),
            })
            if mod == "pool":
                pool_eff.append(eff)
                pool_frac.append(frac)
            elif mod == "encoder":
                enc_frac.append(frac)
                enc_eff.append(eff)

        ckey = cell_key(pooling, split, balanced)
        rec = per_ckpt.setdefault(ckey, {
            "pooling": pooling, "split": split, "balanced": balanced,
            "pool_eff": [], "pool_frac": [], "enc_frac": [], "enc_eff": [],
            "seeds": [], "pool_eff_byseed": [], "enc_frac_byseed": [],
        })
        rec["seeds"].append(seed)
        if pool_eff:
            rec["pool_eff"].append(sum(pool_eff) / len(pool_eff))
            rec["pool_frac"].append(sum(pool_frac) / len(pool_frac))
            rec["pool_eff_byseed"].append((seed, sum(pool_eff) / len(pool_eff)))
        if enc_frac:
            rec["enc_frac"].append(sum(enc_frac) / len(enc_frac))
            rec["enc_eff"].append(sum(enc_eff) / len(enc_eff))
            rec["enc_frac_byseed"].append((seed, sum(enc_frac) / len(enc_frac)))

    # ---- write per-layer csv ----
    perlayer_csv = tables / "19_effective_rank_perlayer.csv"
    with perlayer_csv.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(perlayer_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(perlayer_rows)

    # ---- summary csv + md ----
    def sortkey(item):
        rec = item[1]
        return (rec["pooling"], SPLIT_ORDER.get(rec["split"], 9), rec["balanced"])

    summary_rows = []
    for ckey, rec in sorted(per_ckpt.items(), key=sortkey):
        pe_m, pe_lo, pe_hi = mean_range(rec["pool_eff"])
        pf_m, *_ = mean_range(rec["pool_frac"])
        ef_m, ef_lo, ef_hi = mean_range(rec["enc_frac"])
        n = len(rec["seeds"])
        summary_rows.append({
            "cell": ckey, "pooling": rec["pooling"], "split": rec["split"],
            "balanced": int(rec["balanced"]), "n_seeds": n,
            "pool_eff_mean": pe_m, "pool_eff_min": pe_lo, "pool_eff_max": pe_hi,
            "pool_eff_spread": (pe_hi - pe_lo) if not math.isnan(pe_m) else math.nan,
            "pool_rank_frac_mean": pf_m,
            "enc_rank_frac_mean": ef_m, "enc_rank_frac_min": ef_lo,
            "enc_rank_frac_max": ef_hi,
        })

    summary_csv = tables / "19_effective_rank_summary.csv"
    with summary_csv.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wtr.writeheader()
        for r in summary_rows:
            wtr.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r.items()})

    def fmt(x):
        return "--" if (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}" if isinstance(x, float) and x > 5 else (f"{x:.3f}" if isinstance(x, float) else str(x))

    md = ["# Effective-rank diagnostic (pooling / encoder weights)",
          "",
          "Effective rank = exp(entropy of the squared singular-value spectrum).",
          "`pool_eff` is the mean effective rank over the pooler's weight matrices",
          "(640x640 each; `set_transformer` has 12, `attention` a small MLP, `mean` none).",
          "`enc_rank_frac` is effective_rank / min_dim averaged over the two encoder",
          "layers (shapes identical across poolers, so directly comparable).",
          "Spread = max-min of pool_eff across seeds (a proxy for solution",
          "consistency). NOTE: undertrained (sub-40-epoch) checkpoints are excluded",
          "via --min-epochs, since near-random weights have artificially high",
          "effective rank and would masquerade as a collapse.",
          "",
          "| cell | n | pool_eff (mean) | pool_eff spread | enc rank-frac (mean) | enc rank-frac range |",
          "|------|---|-----------------|-----------------|----------------------|---------------------|"]
    for r in summary_rows:
        pe = fmt(r["pool_eff_mean"])
        spread = fmt(r["pool_eff_spread"])
        ef = fmt(r["enc_rank_frac_mean"])
        erange = (f"{r['enc_rank_frac_min']:.3f}-{r['enc_rank_frac_max']:.3f}"
                  if not math.isnan(r["enc_rank_frac_mean"]) else "--")
        md.append(f"| {r['cell']} | {r['n_seeds']} | {pe} | {spread} | {ef} | {erange} |")
    md.append("")
    md.append("## Per-seed pooler effective rank (set_transformer)")
    md.append("")
    md.append("| cell | seed | pool_eff |")
    md.append("|------|------|----------|")
    for ckey, rec in sorted(per_ckpt.items(), key=sortkey):
        if rec["pooling"] != "set_transformer":
            continue
        for seed, val in sorted(rec["pool_eff_byseed"]):
            md.append(f"| {ckey} | {seed} | {val:.2f} |")
    (tables / "19_effective_rank_summary.md").write_text("\n".join(md) + "\n")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable ({e}); skipping figure")
        plt = None

    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
        # Left: set_transformer pooler effective rank by split, per-seed dots + mean.
        st_cells = [(k, v) for k, v in per_ckpt.items()
                    if v["pooling"] == "set_transformer" and v["pool_eff_byseed"]]
        st_cells.sort(key=lambda kv: (SPLIT_ORDER.get(kv[1]["split"], 9),
                                      kv[1]["balanced"]))
        xs = list(range(len(st_cells)))
        labels = []
        for x, (ckey, rec) in zip(xs, st_cells):
            vals = [v for _, v in rec["pool_eff_byseed"]]
            ax1.scatter([x] * len(vals), vals, color="#c0392b", zorder=3, s=42)
            ax1.scatter([x], [sum(vals) / len(vals)], color="black", marker="_",
                        s=900, linewidths=2, zorder=4)
            labels.append(rec["split"] + ("\n(bal)" if rec["balanced"] else ""))
        ax1.set_xticks(xs)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("pooler effective rank (mean over 12 matrices)")
        ax1.set_title("set_transformer pooler capacity use\n(dots = seeds, bar = mean)")
        ax1.grid(axis="y", alpha=0.3)

        # Right: encoder rank fraction across poolers x split (mean +/- range).
        poolers = ["mean", "attention", "set_transformer"]
        colors = {"mean": "#7f8c8d", "attention": "#2980b9",
                  "set_transformer": "#c0392b"}
        split_cells = ["species", "genus", "family"]
        width = 0.25
        for i, pooler in enumerate(poolers):
            ys, los, his = [], [], []
            for split in split_cells:
                rec = per_ckpt.get(cell_key(pooler, split, False))
                if rec and rec["enc_frac"]:
                    m, lo, hi = mean_range(rec["enc_frac"])
                else:
                    m = lo = hi = math.nan
                ys.append(m)
                los.append(m - lo if not math.isnan(m) else 0)
                his.append(hi - m if not math.isnan(m) else 0)
            xpos = [j + (i - 1) * width for j in range(len(split_cells))]
            ax2.bar(xpos, ys, width, yerr=[los, his], capsize=3,
                    color=colors[pooler], label=pooler, alpha=0.9)
        ax2.set_xticks(range(len(split_cells)))
        ax2.set_xticklabels(split_cells)
        ax2.set_ylabel("encoder effective-rank fraction")
        ax2.set_title("encoder capacity use across poolers x split")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Spectral effective-rank diagnostic of learned weights", y=1.02)
        fig.tight_layout()
        out_png = figs / "19_effective_rank.png"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
        print(f"wrote {out_png} (+ .pdf)")

    print(f"wrote {perlayer_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {tables / '19_effective_rank_summary.md'}")
    print(f"\nscanned {len(ckpts)} checkpoints, {len(perlayer_rows)} weight matrices")


if __name__ == "__main__":
    main()
