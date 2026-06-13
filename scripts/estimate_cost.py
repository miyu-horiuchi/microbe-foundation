#!/usr/bin/env python3
"""Dry-run GPU cost estimate for the tier-1 experiment matrix.

Reads the per-protein manifest (genome + protein counts) and the run
configuration, then estimates GPU-hours and $ cost *before* you spin up a box.

All throughput numbers are coarse, clearly-labeled assumptions you can override
from the command line. The point is an order-of-magnitude budget, not a promise.

Example:
    python3 scripts/estimate_cost.py \
        --manifest data/esm2_perprotein_manifest.parquet \
        --esm-tag 650M --max-proteins 2048 \
        --poolings mean attention set_transformer \
        --splits species genus family --seeds 5 --epochs 40 \
        --gpu A100 --extract
"""
from __future__ import annotations

import argparse
import sys

# --- coarse throughput priors (override via flags) --------------------------
# ESM-2 forward-pass throughput on a single A100 (proteins/sec), batched.
# Scales down roughly with parameter count; treat as ballpark midpoints.
ESM_PROT_PER_SEC = {
    "150M": 1500.0,
    "650M": 400.0,
    "3B": 80.0,
}
# Per-GPU on-demand $/hr priors (Lambda Cloud-ish; override with --rate).
GPU_RATE = {
    "A100": 1.10,   # 40GB A100
    "A100-80": 1.79,
    "H100": 2.49,
    "A10": 0.75,
    "A6000": 0.80,
}
# Training cost prior: GPU-seconds spent per (train genome x epoch). The pooler
# + MLP head over precomputed per-protein embeddings is light and largely IO-
# bound, so this is small. Override with --sec-per-genome-epoch.
DEFAULT_SEC_PER_GENOME_EPOCH = 0.004


def human_h(h: float) -> str:
    return f"{h:,.1f} GPU-hr"


def estimate(
    n_genomes: int,
    capped_proteins: int,
    *,
    n_poolings: int,
    n_splits: int,
    seeds: int,
    epochs: int,
    train_frac: float = 0.7,
    sec_per_genome_epoch: float = DEFAULT_SEC_PER_GENOME_EPOCH,
    balanced: bool = True,
    extract: bool = False,
    throughput: float = 400.0,
    rate: float = 1.10,
) -> dict:
    """Pure cost model -> GPU-hours and $ (no IO). See main() for CLI wiring."""
    extract_h = (capped_proteins / throughput / 3600.0) if extract else 0.0
    n_runs_a = n_poolings * n_splits * seeds
    n_runs_b = (n_poolings * seeds) if balanced else 0
    n_runs = n_runs_a + n_runs_b
    train_genomes = n_genomes * train_frac
    per_run_h = epochs * train_genomes * sec_per_genome_epoch / 3600.0
    train_h = n_runs * per_run_h
    total_h = extract_h + train_h
    return {
        "extract_h": extract_h,
        "n_runs": n_runs,
        "n_runs_a": n_runs_a,
        "n_runs_b": n_runs_b,
        "train_genomes": train_genomes,
        "per_run_h": per_run_h,
        "train_h": train_h,
        "total_h": total_h,
        "total_cost": total_h * rate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/esm2_perprotein_manifest.parquet")
    ap.add_argument("--esm-tag", default="650M",
                    help="encoder size key for throughput prior (150M/650M/3B)")
    ap.add_argument("--esm-throughput", type=float, default=None,
                    help="override proteins/sec for extraction")
    ap.add_argument("--max-proteins", type=int, default=2048,
                    help="cap proteins/genome (0 = no cap)")
    ap.add_argument("--extract", action="store_true",
                    help="include Section 0 (re-embed all proteins with the larger ESM)")
    ap.add_argument("--poolings", nargs="+",
                    default=["mean", "attention", "set_transformer"])
    ap.add_argument("--splits", nargs="+", default=["species", "genus", "family"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="fraction of genomes used for training each run")
    ap.add_argument("--sec-per-genome-epoch", type=float,
                    default=DEFAULT_SEC_PER_GENOME_EPOCH)
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="include Section B family-balanced runs (poolings x seeds)")
    ap.add_argument("--no-balanced", dest="balanced", action="store_false")
    ap.add_argument("--gpu", default="A100", help="GPU type for $ prior")
    ap.add_argument("--rate", type=float, default=None, help="override $/GPU-hr")
    ap.add_argument("--uncertainty", type=float, default=0.5,
                    help="+/- fraction shown as a range around the point estimate")
    args = ap.parse_args()

    # --- read manifest ------------------------------------------------------
    try:
        import pandas as pd
        m = pd.read_parquet(args.manifest)
        n_genomes = int(len(m))
        prot = m["n_proteins"].to_numpy()
        total_proteins = int(prot.sum())
        if args.max_proteins and args.max_proteins > 0:
            import numpy as np
            capped_proteins = int(np.minimum(prot, args.max_proteins).sum())
        else:
            capped_proteins = total_proteins
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not read manifest '{args.manifest}': {e}", file=sys.stderr)
        return 1

    rate = args.rate if args.rate is not None else GPU_RATE.get(args.gpu, 1.10)
    thr = (args.esm_throughput if args.esm_throughput is not None
           else ESM_PROT_PER_SEC.get(args.esm_tag, 400.0))

    extract_proteins = capped_proteins if args.max_proteins else total_proteins
    est = estimate(
        n_genomes, extract_proteins,
        n_poolings=len(args.poolings), n_splits=len(args.splits),
        seeds=args.seeds, epochs=args.epochs, train_frac=args.train_frac,
        sec_per_genome_epoch=args.sec_per_genome_epoch, balanced=args.balanced,
        extract=args.extract, throughput=thr, rate=rate,
    )
    extract_h = est["extract_h"]
    n_runs, n_runs_a, n_runs_b = est["n_runs"], est["n_runs_a"], est["n_runs_b"]
    train_genomes, per_run_h, train_h = est["train_genomes"], est["per_run_h"], est["train_h"]
    total_h = est["total_h"]
    lo = total_h * (1 - args.uncertainty)
    hi = total_h * (1 + args.uncertainty)

    # --- report -------------------------------------------------------------
    print("=" * 64)
    print("  TIER-1 GPU COST ESTIMATE (dry run)")
    print("=" * 64)
    print(f"  manifest          : {args.manifest}")
    print(f"  genomes           : {n_genomes:,}")
    print(f"  proteins (total)  : {total_proteins:,}")
    if args.max_proteins:
        print(f"  proteins (capped) : {capped_proteins:,}  (<= {args.max_proteins}/genome)")
    print("-" * 64)
    print("  ASSUMPTIONS (override via flags):")
    print(f"    encoder          : esm2_{args.esm_tag}  @ {thr:,.0f} prot/s on {args.gpu}")
    print(f"    train throughput : {args.sec_per_genome_epoch} GPU-s per (genome x epoch)")
    print(f"    GPU rate         : ${rate:.2f}/hr ({args.gpu})")
    print("-" * 64)
    print("  WORK:")
    if args.extract:
        print(f"    Section 0 extract: {extract_proteins:,} prot -> {human_h(extract_h)}")
    else:
        print("    Section 0 extract: skipped (reuse existing embeddings)")
    print(f"    Sections A+B     : {n_runs} runs "
          f"({n_runs_a} pooling/split/seed + {n_runs_b} family-balanced)")
    print(f"                       {args.epochs} epochs x ~{train_genomes:,.0f} train genomes")
    print(f"                       ~{per_run_h:.2f} GPU-hr/run -> {human_h(train_h)}")
    print("-" * 64)
    print(f"  TOTAL (point)     : {human_h(total_h)}   ~= ${total_h * rate:,.0f}")
    print(f"  TOTAL (+/-{int(args.uncertainty*100)}%)   : "
          f"{lo:,.0f}-{hi:,.0f} GPU-hr   ~= ${lo*rate:,.0f}-${hi*rate:,.0f}")
    print("=" * 64)
    print("  NOTE: throughput priors are coarse. Re-run with --esm-throughput /")
    print("        --sec-per-genome-epoch from a short real timing to tighten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
