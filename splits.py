"""
splits.py — generate train/val/test splits for the microbe-foundation benchmark.

Reads:  data/traits.parquet  (output of parse_bacdive.py)
Writes: data/splits.parquet  with columns:
    bacdive_id, family, genus, species,
    family_split, genus_split, species_split  (each ∈ {train, val, test, unknown})

Splits are held out at three taxonomic levels:
    - family:  microbe-foundation's primary benchmark protocol
               (cross-family generalization — strictly harder than BacBench)
    - genus:   matches BacBench protocol (cross-genus generalization)
    - species: near-random baseline (within-genus generalization)

For each level, taxonomic groups are randomly assigned to {train, val, test}
roughly 80/10/10 by group count, then strains inherit their group's split.
Different RNG seed per level so the three splits are independent.

Strains with missing taxonomy at a given level are tagged "unknown" and
should be excluded from that level's evaluation (but can still be used
in training if they have other usable labels).

Usage:
    python splits.py
    python splits.py --seed 42 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas pyarrow")


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_IN = DATA_DIR / "traits.parquet"
DEFAULT_OUT = DATA_DIR / "splits.parquet"


def assign_groups(
    group_sizes: dict[str, int], val_frac: float, test_frac: float, seed: int
) -> dict[str, str]:
    """
    Assign taxonomic groups to train/val/test, stratified by strain count.

    Algorithm: largest-group-first greedy bin-packing.
    1. Shuffle groups (for randomness within same-size groups).
    2. Sort by descending size — large groups placed first so they land where
       they fit best instead of overshooting whichever bucket they hit by luck.
    3. For each group, assign to the bucket with the largest remaining capacity
       (target - filled). Whole groups never split, so cross-group leakage
       is impossible by construction.
    """
    rng = random.Random(seed)
    items = list(group_sizes.items())
    rng.shuffle(items)
    items.sort(key=lambda x: -x[1])  # stable sort preserves shuffle within ties
    total = sum(group_sizes.values())
    targets = {
        "train": total * (1.0 - val_frac - test_frac),
        "val": total * val_frac,
        "test": total * test_frac,
    }
    filled = {"train": 0, "val": 0, "test": 0}
    out: dict[str, str] = {}
    for name, size in items:
        best = max(filled, key=lambda b: targets[b] - filled[b])
        out[name] = best
        filled[best] += size
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_IN, help="Input traits parquet")
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUT, help="Output splits parquet")
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction. Default: 0.1")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction. Default: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed. Default: 42")
    args = parser.parse_args()

    if not args.input_path.exists():
        sys.exit(f"Input file not found: {args.input_path}. Run parse_bacdive.py first.")

    df = pd.read_parquet(args.input_path, columns=["bacdive_id", "family", "genus", "species"])
    print(f"Loaded {len(df):,} strains from {args.input_path}")

    out = df.copy()

    # Independent seed per level so splits aren't correlated by accident.
    level_seed_offset = {"family": 0, "genus": 1, "species": 2}

    for level in ("family", "genus", "species"):
        col = f"{level}_split"
        group_sizes = df[level].dropna().value_counts().to_dict()
        assignment = assign_groups(group_sizes, args.val, args.test, args.seed + level_seed_offset[level])
        out[col] = df[level].map(assignment).fillna("unknown")

        print(f"\n[{level}-held-out splits]")
        print(f"  unique {level}s: {len(group_sizes):,}")
        n_unknown = (df[level].isna()).sum()
        if n_unknown:
            print(f"  strains missing {level}: {n_unknown:,}")
        print("  strain split counts:")
        for split, count in out[col].value_counts().items():
            pct = 100 * count / len(out)
            print(f"    {split:<8} {count:>6,} ({pct:5.1f}%)")

    # Family/genus consistency sanity check: any family that appears in more than one
    # train/val/test bucket would mean a genus-level leak. Should never happen with this code.
    fam_to_splits = out.groupby("family")["family_split"].nunique()
    leaks = fam_to_splits[fam_to_splits > 1]
    if len(leaks):
        print(f"\nWARNING: {len(leaks)} families span multiple family_split buckets — this is a bug!")
    else:
        print("\nFamily-split consistency check passed (no family spans multiple buckets).")

    args.output_path.parent.mkdir(exist_ok=True)
    out.to_parquet(args.output_path, index=False)
    print(f"\nWrote {len(out):,} rows × {len(out.columns)} cols to {args.output_path}")


if __name__ == "__main__":
    main()
