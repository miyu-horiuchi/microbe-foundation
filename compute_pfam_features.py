"""
compute_pfam_features.py — Pfam domain presence/absence features per genome.

Pipeline (mirrors compute_esm2_features_mp.py):

  fetch FASTA  →  pyrodigal predict proteins  →  pyhmmer hmmscan Pfam-A
              →  binary vector of domain hits  →  npz

ProcessPoolExecutor with N workers parallelizes the slow per-genome scan;
each worker is an independent CPU pipeline. No GPU needed.

Vocabulary handling:
  Pass 1 (with --build-vocab): scan a sample of genomes, write data/pfam_vocab.json
         with families seen in >= min-freq fraction of sampled genomes.
  Pass 2 (default): use the saved vocab to project per-genome hit-sets into
         a fixed-dim binary vector.

Usage:
    # Pass 1: build vocab from a sample of 2000 genomes
    python compute_pfam_features.py --hmm pfam_db/Pfam-A.hmm \
        --build-vocab --sample-genomes 2000 --min-freq 0.01

    # Pass 2: compute features for all genomes using the vocab
    python compute_pfam_features.py --hmm pfam_db/Pfam-A.hmm \
        --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_ACCESSIONS = DATA_DIR / "genome_accessions.tsv"
DEFAULT_OUT = DATA_DIR / "pfam_features.npz"
DEFAULT_VOCAB = DATA_DIR / "pfam_vocab.json"
ACC_DTYPE = "<U24"


def _scan_one(args_tuple):
    """Worker: fetch FASTA, predict proteins, pyhmmer.hmmscan, return Pfam hit accessions.

    Returns: (bid, acc, hits_set_or_fail_str)
    """
    bid, acc, hmm_path, eval_cutoff = args_tuple
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes
    import pyhmmer
    from pyhmmer.easel import Alphabet, TextSequence

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

    # Build Easel digital sequences from protein strings
    abc = Alphabet.amino()
    seqs = []
    for i, p in enumerate(proteins):
        if not p:
            continue
        try:
            ts = TextSequence(name=f"p{i}".encode(), sequence=p)
            seqs.append(ts.digitize(abc))
        except Exception:
            continue
    if not seqs:
        return (bid, acc, "PRED_FAIL:no_valid")

    hits_set: set[str] = set()
    try:
        with pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
            for top in pyhmmer.hmmer.hmmscan(seqs, hmm_file, E=eval_cutoff, cpus=1):
                for hit in top.included:
                    name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
                    hits_set.add(name)
    except Exception as e:
        return (bid, acc, f"SCAN_FAIL:{type(e).__name__}:{e}")

    return (bid, acc, sorted(hits_set))


def load_existing(path: Path, vocab: list[str]) -> tuple[list[int], list[np.ndarray], list[str]]:
    if not path.exists():
        return [], [], []
    z = np.load(path)
    return z["bacdive_ids"].tolist(), list(z["features"]), [str(a) for a in z["accessions"]]


def save_npz(path: Path, ids: list[int], feats: list[np.ndarray], accs: list[str]) -> None:
    np.savez(
        path,
        bacdive_ids=np.array(ids, dtype=np.int64),
        features=np.array(feats, dtype=np.float32),  # 0/1 binary, stored as float for downstream compat
        accessions=np.array(accs, dtype=ACC_DTYPE),
    )


def build_vocab(hits_per_genome: list[list[str]], min_freq: float) -> list[str]:
    """Keep Pfam families seen in at least `min_freq` fraction of input genomes."""
    n = len(hits_per_genome)
    counts: dict[str, int] = {}
    for hits in hits_per_genome:
        for h in hits:
            counts[h] = counts.get(h, 0) + 1
    threshold = max(1, int(n * min_freq))
    kept = sorted([k for k, c in counts.items() if c >= threshold])
    return kept


def hits_to_vec(hits: list[str], idx: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(idx), dtype=np.float32)
    for h in hits:
        i = idx.get(h)
        if i is not None:
            v[i] = 1.0
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accessions", type=Path, default=DEFAULT_ACCESSIONS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--hmm", type=Path, required=True, help="Path to Pfam-A.hmm")
    p.add_argument("--eval-cutoff", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Process at most N new genomes")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--build-vocab", action="store_true",
                   help="First pass: build the vocab from --sample-genomes, write to --vocab, exit.")
    p.add_argument("--sample-genomes", type=int, default=2000,
                   help="Number of genomes to sample for vocab building.")
    p.add_argument("--min-freq", type=float, default=0.01,
                   help="Pfam family must appear in at least this fraction of vocab-sample genomes.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.hmm.exists():
        sys.exit(f"HMM file not found: {args.hmm}")
    if not args.accessions.exists():
        sys.exit(f"Accessions file not found: {args.accessions}")

    acc_df = pd.read_csv(args.accessions, sep="\t")
    rng = np.random.default_rng(args.seed)

    if args.build_vocab:
        # Sample N genomes, scan, collect hit sets, write vocab.
        sample = acc_df.sample(n=min(args.sample_genomes, len(acc_df)), random_state=args.seed)
        pairs = [(int(r.bacdive_id), str(r.accession), str(args.hmm), args.eval_cutoff) for r in sample.itertuples()]
        print(f"build-vocab: scanning {len(pairs):,} genomes with {args.workers} workers ...", flush=True)

        hits_per_genome: list[list[str]] = []
        ok = ff = pf = sf = 0
        start = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_scan_one, t) for t in pairs]
            for i, fut in enumerate(as_completed(futures), start=1):
                bid, acc, payload = fut.result()
                if isinstance(payload, list):
                    hits_per_genome.append(payload)
                    ok += 1
                elif payload == "FETCH_FAIL":
                    ff += 1
                elif isinstance(payload, str) and payload.startswith("PRED_FAIL"):
                    pf += 1
                elif isinstance(payload, str) and payload.startswith("SCAN_FAIL"):
                    sf += 1
                if i % 25 == 0:
                    elapsed = time.time() - start
                    print(f"  [{i:>5}/{len(pairs)}] ok={ok} ff={ff} pf={pf} sf={sf} rate={i/elapsed:.2f}/s", flush=True)

        kept = build_vocab(hits_per_genome, args.min_freq)
        out = {
            "n_sample_genomes": len(hits_per_genome),
            "min_freq": args.min_freq,
            "eval_cutoff": args.eval_cutoff,
            "vocab": kept,
        }
        args.vocab.write_text(json.dumps(out, indent=2))
        print(f"\nwrote vocab: {len(kept):,} Pfam families to {args.vocab} "
              f"(min_freq={args.min_freq} of {len(hits_per_genome)} sampled genomes)", flush=True)
        return

    # Feature-extraction pass: needs vocab.
    if not args.vocab.exists():
        sys.exit(f"Vocab not found: {args.vocab}. Run with --build-vocab first.")
    vocab_doc = json.loads(args.vocab.read_text())
    vocab = vocab_doc["vocab"]
    idx = {name: i for i, name in enumerate(vocab)}
    print(f"loaded vocab: {len(vocab):,} Pfam families", flush=True)

    ids_out, feats_out, accs_out = load_existing(args.out, vocab)
    done = set(ids_out)
    print(f"resumed: {len(done):,} embeddings already in {args.out}", flush=True)

    todo = acc_df[~acc_df.bacdive_id.isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    pairs = [(int(r.bacdive_id), str(r.accession), str(args.hmm), args.eval_cutoff) for r in todo.itertuples()]
    print(f"to compute: {len(pairs):,} genomes", flush=True)
    if not pairs:
        return

    n_ok = n_fetch_fail = n_pred_fail = n_scan_fail = 0
    start = time.time()
    i = 0
    max_inflight = args.workers * 4
    ex = ProcessPoolExecutor(max_workers=args.workers)
    try:
        in_flight = set()
        nxt = 0
        while nxt < len(pairs) and len(in_flight) < max_inflight:
            in_flight.add(ex.submit(_scan_one, pairs[nxt])); nxt += 1
        while in_flight:
            done_fut = next(as_completed(in_flight))
            in_flight.remove(done_fut)
            if nxt < len(pairs):
                in_flight.add(ex.submit(_scan_one, pairs[nxt])); nxt += 1
            bid, acc, payload = done_fut.result()
            i += 1
            if isinstance(payload, list):
                v = hits_to_vec(payload, idx)
                ids_out.append(bid); feats_out.append(v); accs_out.append(acc); n_ok += 1
            elif payload == "FETCH_FAIL":
                n_fetch_fail += 1
            elif isinstance(payload, str) and payload.startswith("PRED_FAIL"):
                n_pred_fail += 1
            elif isinstance(payload, str) and payload.startswith("SCAN_FAIL"):
                n_scan_fail += 1
            if i % args.checkpoint_every == 0:
                save_npz(args.out, ids_out, feats_out, accs_out)
                elapsed = time.time() - start
                rate = i / max(elapsed, 1e-6)
                eta_min = (len(pairs) - i) / max(rate, 1e-6) / 60
                print(
                    f"  [{i:>6,}/{len(pairs):,}] ok={n_ok:,} ff={n_fetch_fail:,} "
                    f"pf={n_pred_fail:,} sf={n_scan_fail:,} rate={rate:.2f}/s eta={eta_min:.1f}min",
                    flush=True,
                )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    save_npz(args.out, ids_out, feats_out, accs_out)
    elapsed = time.time() - start
    print(
        f"\ndone in {elapsed/60:.1f}min. {len(ids_out):,} total. "
        f"this run ok={n_ok:,} ff={n_fetch_fail:,} pf={n_pred_fail:,} sf={n_scan_fail:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
