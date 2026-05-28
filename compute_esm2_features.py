"""
compute_esm2_features.py — fetch genomes from NCBI in memory, predict proteins
with pyrodigal, embed with ESM-2, save per-genome mean-pooled vectors.

Reuses the microbe-model v0 modules under microbe_model/:
    pipeline._fetch_fasta_bytes()                NCBI Datasets fetch, in-memory
    features.genome.predict_genes()              pyrodigal protein prediction
    features.embeddings.load_esm2/embed_genome() ESM-2 encoder + mean-pool

Reads:
    data/genome_accessions.tsv      (output of extract_genome_accessions.py)

Writes:
    data/esm2_features.npz with:
        bacdive_ids: int64 [N]
        features:    float32 [N, embed_dim]
        accessions:  unicode <U24 [N]

Resumable: skips genomes already in the output npz. Checkpoints every 25
genomes so a crash loses at most that many.

ESM-2 model choices (set via --model):
    facebook/esm2_t6_8M_UR50D    -> 320-dim, ~1s/protein on CPU (laptop OK)
    facebook/esm2_t12_35M_UR50D  -> 480-dim
    facebook/esm2_t30_150M_UR50D -> 640-dim, recommended GPU default
    facebook/esm2_t33_650M_UR50D -> 1280-dim, needs ~8GB VRAM

Set --sample-n 50 to embed only 50 randomly-sampled proteins per genome
(big speedup on large genomes; matches microbe-model v0 default).

Usage:
    # Laptop smoke test (smallest model, sample 30 proteins/genome, 10 genomes)
    python compute_esm2_features.py --model facebook/esm2_t6_8M_UR50D \\
        --sample-n 30 --limit 10

    # Full corpus on GPU
    python compute_esm2_features.py --model facebook/esm2_t30_150M_UR50D \\
        --sample-n 50 --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError:
    sys.exit("requires: pip install pandas pyarrow torch numpy")

# Make the vendored microbe_model package importable
sys.path.insert(0, str(Path(__file__).parent))

try:
    from microbe_model.features.embeddings import embed_genome, load_esm2, pick_device
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes
except ImportError as e:
    sys.exit(
        f"microbe_model import failed: {e}\n"
        "Install runtime deps: pip install pyrodigal biopython requests transformers"
    )


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_ACCESSIONS = DATA_DIR / "genome_accessions.tsv"
DEFAULT_OUT = DATA_DIR / "esm2_features.npz"
ACC_DTYPE = "<U24"


def load_existing(path: Path) -> tuple[list[int], list[np.ndarray], list[str]]:
    if not path.exists():
        return [], [], []
    npz = np.load(path)
    return (
        npz["bacdive_ids"].tolist(),
        list(npz["features"]),
        [str(a) for a in npz["accessions"]],
    )


def save_npz(path: Path, ids: list[int], feats: list[np.ndarray], accs: list[str]) -> None:
    np.savez(
        path,
        bacdive_ids=np.array(ids, dtype=np.int64),
        features=np.array(feats, dtype=np.float32),
        accessions=np.array(accs, dtype=ACC_DTYPE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accessions", type=Path, default=DEFAULT_ACCESSIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D",
                        help="HuggingFace ESM-2 model id")
    parser.add_argument("--sample-n", type=int, default=50,
                        help="Sample N proteins per genome (None = all). Default: 50")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="ESM-2 batch size (raise on GPU for speed)")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N new genomes")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.accessions.exists():
        sys.exit(f"Accessions file not found: {args.accessions}. Run extract_genome_accessions.py first.")

    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: running on CPU. Use --model facebook/esm2_t6_8M_UR50D for fastest CPU baseline.")

    ids_out, feats_out, accs_out = load_existing(args.out)
    done = set(ids_out)
    print(f"resumed: {len(done):,} embeddings already in {args.out}")

    acc_df = pd.read_csv(args.accessions, sep="\t")
    todo = acc_df[~acc_df.bacdive_id.isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"to compute: {len(todo):,} genomes")
    if len(todo) == 0:
        print("nothing to do.")
        return

    print(f"loading {args.model} ...")
    tokenizer, model, device = load_esm2(args.model, device=device)
    print(f"  embed_dim = {model.config.hidden_size}")

    rng = np.random.default_rng(args.seed)
    n_ok = n_fetch_fail = n_pred_fail = n_embed_fail = 0
    start = time.time()

    for i, row in enumerate(todo.itertuples(), start=1):
        bid = int(row.bacdive_id)
        acc = str(row.accession)

        contigs = _fetch_fasta_bytes(acc)
        if not contigs:
            n_fetch_fail += 1
            continue
        try:
            proteins, _cds, _total_nt = predict_genes(contigs)
        except Exception as e:
            print(f"  [warn] {bid} ({acc}) predict_genes failed: {type(e).__name__}: {e}", flush=True)
            n_pred_fail += 1
            continue
        if not proteins:
            n_pred_fail += 1
            continue
        try:
            emb = embed_genome(
                proteins, tokenizer, model, device,
                sample_n=args.sample_n,
                batch_size=args.batch_size,
                rng=rng,
            )
        except Exception as e:
            print(f"  [warn] {bid} ({acc}) embed_genome failed: {type(e).__name__}: {e}", flush=True)
            n_embed_fail += 1
            continue

        ids_out.append(bid)
        feats_out.append(emb)
        accs_out.append(acc)
        n_ok += 1

        if i % args.checkpoint_every == 0:
            save_npz(args.out, ids_out, feats_out, accs_out)
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-6)
            eta_min = (len(todo) - i) / max(rate, 1e-6) / 60
            print(
                f"  [{i:>6,}/{len(todo):,}]  ok={n_ok:,}  fetch_fail={n_fetch_fail:,}  "
                f"pred_fail={n_pred_fail:,}  embed_fail={n_embed_fail:,}  "
                f"rate={rate:.2f}/s  eta={eta_min:.1f}min",
                flush=True,
            )

    save_npz(args.out, ids_out, feats_out, accs_out)
    print(f"\ndone. {len(ids_out):,} total embeddings in {args.out}")
    print(f"  this run: ok={n_ok:,}  fetch_fail={n_fetch_fail:,}  "
          f"pred_fail={n_pred_fail:,}  embed_fail={n_embed_fail:,}")


if __name__ == "__main__":
    main()
