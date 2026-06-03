"""
compute_esm2_perprotein_mp.py — un-pooled, full-proteome ESM-2 embeddings.

Unlike compute_esm2_features_mp.py (which mean-pools every genome down to one
[640] vector), this keeps EACH protein as its own row: one [n_proteins, 640]
matrix per genome, written to its own .npy file. An attention-pooling head in
model.py then learns which proteins matter per trait, instead of averaging them.

Why per-genome files instead of one big .npz:
  - ragged: proteomes vary from hundreds to ~10k proteins
  - ~100 GB total at fp16 — can't live in RAM as one array
  - resumable: a finished genome = a file on disk

Output layout (--out-dir, default data/esm2_perprotein/):
    <bacdive_id>.npy          float16 [n_proteins, embed_dim]
    manifest.parquet          bacdive_id, accession, n_proteins, status, path

Workers (ProcessPoolExecutor) do NCBI fetch + pyrodigal predict_genes; the main
process runs ESM-2 on the GPU and writes each genome's matrix as it completes.

Usage (on a Lambda GPU box):
    export NCBI_API_KEY=<key>
    python compute_esm2_perprotein_mp.py \
        --model facebook/esm2_t30_150M_UR50D \
        --batch-size 32 --workers 16

Smoke test first (cheap, proves the format):
    python compute_esm2_perprotein_mp.py --model facebook/esm2_t6_8M_UR50D \
        --batch-size 4 --workers 4 --limit 5 --out-dir data/esm2_perprotein_smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_ACCESSIONS = DATA_DIR / "genome_accessions.tsv"
DEFAULT_OUT_DIR = DATA_DIR / "esm2_perprotein"


def _fetch_predict(bid_acc):
    """Worker: fetch FASTA + pyrodigal predict, return (bid, acc, proteins | sentinel)."""
    bid, acc = bid_acc
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes
    try:
        contigs = _fetch_fasta_bytes(acc)
    except Exception:
        return (bid, acc, "FETCH_FAIL")
    if not contigs:
        return (bid, acc, "FETCH_FAIL")
    try:
        proteins, _cds, _total_nt = predict_genes(contigs)
    except Exception as e:
        return (bid, acc, f"PRED_FAIL:{type(e).__name__}")
    if not proteins:
        return (bid, acc, "PRED_FAIL:empty")
    return (bid, acc, proteins)


def genome_path(out_dir: Path, bid: int) -> Path:
    return out_dir / f"{bid}.npy"


def scan_done(out_dir: Path) -> set[int]:
    """Resume support: bacdive_ids whose .npy already exists and is non-empty."""
    done = set()
    for p in out_dir.glob("*.npy"):
        try:
            if p.stat().st_size > 0:
                done.add(int(p.stem))
        except (ValueError, OSError):
            continue
    return done


def write_manifest(out_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_parquet(out_dir / "manifest.parquet", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--accessions", type=Path, default=DEFAULT_ACCESSIONS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-proteins", type=int, default=0,
                   help="If >0, cap proteins per genome (uniform sample) to bound storage. 0 = keep all.")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", default="0/1",
                   help="i/N: process only genomes where row_index %% N == i. For a "
                        "multi-GPU box run N processes, one per GPU: "
                        "CUDA_VISIBLE_DEVICES=i python ... --shard i/N (i = 0..N-1). "
                        "Shards write disjoint <bid>.npy to a shared --out-dir.")
    args = p.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if not (0 <= shard_i < shard_n):
        raise SystemExit(f"--shard {args.shard}: need 0 <= i < N")

    from microbe_model.features.embeddings import embed_proteins, load_esm2, pick_device
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  workers: {args.workers}  out_dir: {args.out_dir}", flush=True)

    done = scan_done(args.out_dir)
    print(f"resumed: {len(done):,} genomes already on disk", flush=True)

    acc_df = pd.read_csv(args.accessions, sep="\t")
    if shard_n > 1:
        # Deterministic round-robin slice: each shard owns a fixed, disjoint
        # subset of rows regardless of resume state.
        acc_df = acc_df.iloc[shard_i::shard_n].reset_index(drop=True)
        print(f"shard {shard_i}/{shard_n}: {len(acc_df):,} of corpus this process", flush=True)
    todo = acc_df[~acc_df.bacdive_id.isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    pairs = [(int(r.bacdive_id), str(r.accession)) for r in todo.itertuples()]
    print(f"to compute: {len(pairs):,} genomes", flush=True)
    if not pairs:
        return

    print(f"loading {args.model} ...", flush=True)
    tokenizer, model, mdev = load_esm2(args.model, device=device)
    print(f"  embed_dim = {model.config.hidden_size}", flush=True)

    rng = np.random.default_rng(args.seed)
    manifest: list[dict] = []
    n_ok = n_fetch_fail = n_pred_fail = n_embed_fail = 0
    n_proteins_total = 0
    start = time.time()
    i = 0

    max_inflight = args.workers * 4
    ex = ProcessPoolExecutor(max_workers=args.workers)
    try:
        in_flight = set()
        idx = 0
        while idx < len(pairs) and len(in_flight) < max_inflight:
            in_flight.add(ex.submit(_fetch_predict, pairs[idx]))
            idx += 1
        while in_flight:
            done_fut = next(as_completed(in_flight))
            in_flight.remove(done_fut)
            if idx < len(pairs):
                in_flight.add(ex.submit(_fetch_predict, pairs[idx]))
                idx += 1
            bid, acc, payload = done_fut.result()
            i += 1
            status = "ok"
            n_prot = 0
            if payload == "FETCH_FAIL":
                n_fetch_fail += 1
                status = "FETCH_FAIL"
            elif isinstance(payload, str) and payload.startswith("PRED_FAIL"):
                n_pred_fail += 1
                status = payload
            else:
                proteins = payload
                if args.max_proteins and len(proteins) > args.max_proteins:
                    sel = rng.choice(len(proteins), size=args.max_proteins, replace=False)
                    proteins = [proteins[j] for j in sel]
                try:
                    # [n_proteins, embed_dim] — each protein residue-mean-pooled, NOT pooled across proteins.
                    matrix = embed_proteins(proteins, tokenizer, model, mdev, batch_size=args.batch_size)
                    np.save(genome_path(args.out_dir, bid), matrix.astype(np.float16))
                    n_prot = int(matrix.shape[0])
                    n_proteins_total += n_prot
                    n_ok += 1
                except Exception as e:
                    print(f"  [warn] {bid} embed: {type(e).__name__}: {e}", flush=True)
                    n_embed_fail += 1
                    status = f"EMBED_FAIL:{type(e).__name__}"

            manifest.append({
                "bacdive_id": bid, "accession": acc,
                "n_proteins": n_prot, "status": status,
                "path": str(genome_path(args.out_dir, bid).name) if status == "ok" else "",
            })

            if i % args.checkpoint_every == 0:
                write_manifest(args.out_dir, manifest)
                elapsed = time.time() - start
                rate = i / max(elapsed, 1e-6)
                eta_min = (len(pairs) - i) / max(rate, 1e-6) / 60
                avg_prot = n_proteins_total / max(n_ok, 1)
                print(
                    f"  [{i:>6,}/{len(pairs):,}]  ok={n_ok:,} ff={n_fetch_fail:,} "
                    f"pf={n_pred_fail:,} ef={n_embed_fail:,} avg_prot={avg_prot:.0f} "
                    f"rate={rate:.2f}/s eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    write_manifest(args.out_dir, manifest)
    elapsed = time.time() - start
    print(
        f"\ndone in {elapsed/60:.1f}min. ok={n_ok:,} ff={n_fetch_fail:,} "
        f"pf={n_pred_fail:,} ef={n_embed_fail:,}  total_proteins={n_proteins_total:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
