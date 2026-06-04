"""
embed_from_cache.py — embed pre-fetched proteins into per-genome ESM-2 matrices.

Consumes the protein cache produced by the Modal pipeline
(modal_esm2_perprotein.py): a directory of `proteins/<bid>.txt.gz` files, each
gzipped with one amino-acid sequence per line. For each genome it runs ESM-2
(residue-mean-pooled per protein, NOT pooled across proteins) and writes
`<bid>.npy` ([n_proteins, embed_dim] fp16) — the exact format model.py
--per-protein expects.

No NCBI fetch: the proteins are already cached, so this is pure GPU work and
shards cleanly across a multi-GPU box (one process per GPU). Resumable: skips any
genome whose <bid>.npy already exists, so you can drop in .npy already computed
elsewhere (e.g. downloaded from a Modal volume) and only the rest get embedded.

Usage on an 8-GPU box (one process per GPU):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i nohup python embed_from_cache.py \
        --proteins-dir data/esm2_perprotein/proteins \
        --out-dir      data/esm2_perprotein \
        --model facebook/esm2_t30_150M_UR50D --batch-size 64 \
        --shard $i/8 > logs/shard_$i.log 2>&1 &
    done; wait
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import torch


def read_proteins(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln for ln in fh.read().split("\n") if ln]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proteins-dir", type=Path, required=True, help="dir of <bid>.txt.gz")
    p.add_argument("--out-dir", type=Path, required=True, help="where <bid>.npy are written")
    p.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--shard", default="0/1", help="i/N: this process handles files where idx %% N == i")
    p.add_argument("--checkpoint-every", type=int, default=50)
    args = p.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if not (0 <= shard_i < shard_n):
        raise SystemExit(f"--shard {args.shard}: need 0 <= i < N")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from microbe_model.features.embeddings import embed_proteins, load_esm2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"shard {shard_i}/{shard_n}  device={device}  loading {args.model} ...", flush=True)
    tokenizer, model, mdev = load_esm2(args.model, device=device)
    print(f"  embed_dim={model.config.hidden_size}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.proteins_dir.glob("*.txt.gz"))
    files = files[shard_i::shard_n]            # deterministic disjoint slice
    todo = [f for f in files if not (args.out_dir / f"{f.name[:-7]}.npy").exists()]
    print(f"  {len(files):,} in shard, {len(todo):,} to embed (rest already done)", flush=True)

    t0 = time.time()
    n = n_prot = 0
    for f in todo:
        bid = f.name[:-7]                      # strip ".txt.gz"
        proteins = read_proteins(f)
        if not proteins:
            continue
        matrix = embed_proteins(proteins, tokenizer, model, mdev, batch_size=args.batch_size)
        np.save(args.out_dir / f"{bid}.npy", matrix.astype(np.float16))
        n += 1
        n_prot += int(matrix.shape[0])
        if n % args.checkpoint_every == 0:
            rate = n / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n) / max(rate, 1e-6) / 60
            print(f"  [{n:,}/{len(todo):,}] {n_prot:,} prot  rate={rate:.2f}/s  eta={eta:.1f}min", flush=True)

    print(f"shard {shard_i} done: {n:,} embedded, {n_prot:,} proteins, {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
