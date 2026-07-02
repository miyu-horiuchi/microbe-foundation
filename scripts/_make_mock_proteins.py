#!/usr/bin/env python3
"""Generate a tiny synthetic proteins dir for CPU DDP smoke tests.

Writes data/_mock_proteins/<bacdive_id>.txt.gz (a few random AA sequences each)
for a handful of genomes per family split so finetune_lora.py --model-name mock
has train/val/test populated. Throwaway; the dir is gitignored under data/.
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "_mock_proteins"
AA = "ACDEFGHIKLMNPQRSTVWY"
PER = {"train": 60, "val": 30, "test": 30}


def main() -> None:
    rng = np.random.default_rng(0)
    traits = pd.read_parquet(ROOT / "data" / "traits.parquet")
    splits = pd.read_parquet(ROOT / "data" / "splits.parquet")
    df = traits.merge(splits[["bacdive_id", "family_split"]], on="bacdive_id", how="left")
    df = df[df.family_split.notna() & df.family.notna()]
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for split, n in PER.items():
        sub = df[df.family_split == split]
        if len(sub) == 0:
            continue
        pick = sub.sample(n=min(n, len(sub)), random_state=0)
        for bid in pick.bacdive_id:
            nseq = int(rng.integers(3, 9))
            seqs = ["".join(rng.choice(list(AA), size=int(rng.integers(20, 60))))
                    for _ in range(nseq)]
            with gzip.open(OUT / f"{bid}.txt.gz", "wt") as fh:
                fh.write("\n".join(seqs) + "\n")
            total += 1
    print(f"wrote {total} mock genomes to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
