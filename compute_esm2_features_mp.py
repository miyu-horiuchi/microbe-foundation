"""
compute_esm2_features_mp.py — multiprocessing fetch+predict version.

Same output format as compute_esm2_features.py (data/esm2_features.npz).
Workers (ProcessPoolExecutor) do NCBI fetch + pyrodigal predict_genes; the
main process runs ESM-2 on the GPU. Pyrodigal holds the GIL so threads
don't help — multiprocessing does. On a 1x H100 + 16 CPU workers + NCBI
API key, this hits ~3 genomes/sec (~9x serial), and finishes the 19,637
BacDive corpus in ~100 minutes.

Usage:
    export NCBI_API_KEY=<key>
    python compute_esm2_features_mp.py \
        --model facebook/esm2_t30_150M_UR50D \
        --sample-n 50 --batch-size 32 --workers 16
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
DEFAULT_OUT = DATA_DIR / "esm2_features.npz"
ACC_DTYPE = "<U24"


def _worker_init():
    pass


def _fetch_predict(bid_acc):
    """Worker: fetch FASTA + pyrodigal predict, return (bid, acc, proteins or fail sentinel)."""
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


def load_existing(path):
    if not path.exists():
        return [], [], []
    z = np.load(path)
    return z["bacdive_ids"].tolist(), list(z["features"]), [str(a) for a in z["accessions"]]


def save_npz(path, ids, feats, accs):
    np.savez(
        path,
        bacdive_ids=np.array(ids, dtype=np.int64),
        features=np.array(feats, dtype=np.float32),
        accessions=np.array(accs, dtype=ACC_DTYPE),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--accessions", type=Path, default=DEFAULT_ACCESSIONS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--sample-n", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from microbe_model.features.embeddings import embed_genome, load_esm2, pick_device
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"device: {device}  workers: {args.workers}", flush=True)

    ids_out, feats_out, accs_out = load_existing(args.out)
    done = set(ids_out)
    print(f"resumed: {len(done):,} embeddings already in {args.out}", flush=True)

    acc_df = pd.read_csv(args.accessions, sep="\t")
    todo = acc_df[~acc_df.bacdive_id.isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    pairs = [(int(r.bacdive_id), str(r.accession)) for r in todo.itertuples()]
    print(f"to compute: {len(pairs):,} genomes", flush=True)
    if not pairs:
        return

    print(f"loading {args.model} ...", flush=True)
    tokenizer, model, modelf_device = load_esm2(args.model, device=device)
    print(f"  embed_dim = {model.config.hidden_size}", flush=True)

    rng = np.random.default_rng(args.seed)
    n_ok = n_fetch_fail = n_pred_fail = n_embed_fail = 0
    start = time.time()
    i = 0

    max_inflight = args.workers * 4
    ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init)
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
            if payload == "FETCH_FAIL":
                n_fetch_fail += 1
            elif isinstance(payload, str) and payload.startswith("PRED_FAIL"):
                n_pred_fail += 1
            else:
                proteins = payload
                try:
                    emb = embed_genome(
                        proteins, tokenizer, model, modelf_device,
                        sample_n=args.sample_n, batch_size=args.batch_size, rng=rng,
                    )
                    ids_out.append(bid)
                    feats_out.append(emb)
                    accs_out.append(acc)
                    n_ok += 1
                except Exception as e:
                    print(f"  [warn] {bid} embed: {type(e).__name__}: {e}", flush=True)
                    n_embed_fail += 1

            if i % args.checkpoint_every == 0:
                save_npz(args.out, ids_out, feats_out, accs_out)
                elapsed = time.time() - start
                rate = i / max(elapsed, 1e-6)
                eta_min = (len(pairs) - i) / max(rate, 1e-6) / 60
                print(
                    f"  [{i:>6,}/{len(pairs):,}]  ok={n_ok:,} ff={n_fetch_fail:,} "
                    f"pf={n_pred_fail:,} ef={n_embed_fail:,} "
                    f"rate={rate:.2f}/s eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    save_npz(args.out, ids_out, feats_out, accs_out)
    elapsed = time.time() - start
    print(
        f"\ndone in {elapsed/60:.1f}min. {len(ids_out):,} total. "
        f"this run ok={n_ok:,} ff={n_fetch_fail:,} pf={n_pred_fail:,} ef={n_embed_fail:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
