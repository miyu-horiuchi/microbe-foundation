"""
combine_features.py — concatenate two .npz feature files into one.

Both inputs use the standard schema (bacdive_ids, features, accessions).
Output keeps the intersection of bacdive_ids (so all rows are aligned)
and concatenates the feature columns.

Usage:
    python combine_features.py \
        --a data/esm2_features_200p.npz \
        --b data/eggnog_features.npz \
        --out data/hybrid_features.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True, help="first .npz")
    p.add_argument("--b", type=Path, required=True, help="second .npz")
    p.add_argument("--out", type=Path, required=True, help="output .npz")
    args = p.parse_args()

    za = np.load(args.a)
    zb = np.load(args.b)
    ids_a = za["bacdive_ids"].astype(np.int64)
    ids_b = zb["bacdive_ids"].astype(np.int64)
    print(f"  a: {ids_a.shape[0]:,} rows × {za['features'].shape[1]:,} dims  ({args.a.name})")
    print(f"  b: {ids_b.shape[0]:,} rows × {zb['features'].shape[1]:,} dims  ({args.b.name})")

    # Intersection of bacdive_ids, in the order of `a`
    pos_b = {int(b): i for i, b in enumerate(ids_b)}
    keep_a = np.array([int(x) in pos_b for x in ids_a])
    ids_out = ids_a[keep_a]
    accs_out = za["accessions"][keep_a]
    feats_a = za["features"][keep_a]
    feats_b = zb["features"][np.array([pos_b[int(x)] for x in ids_out])]
    feats_out = np.concatenate([feats_a, feats_b], axis=1).astype(np.float32)
    print(f"  intersection: {ids_out.shape[0]:,} rows × {feats_out.shape[1]:,} dims")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        bacdive_ids=ids_out,
        features=feats_out,
        accessions=accs_out,
    )
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
