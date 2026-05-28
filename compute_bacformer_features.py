"""
compute_bacformer_features.py — generate Bacformer embeddings for downloaded genomes.

Bacformer (Wiatrak et al. 2025, https://github.com/macwiatrak/Bacformer) is a
prokaryotic genome foundation model that takes a sequence of protein
embeddings (from ESM-2 / ESM-C) and produces context-aware genome
representations. We use its mean-pooled last_hidden_state as features for the
microbe-foundation multi-task heads — a direct peer comparator to BacPT.

Requires GPU. Install:
    pip install "torch>=2.2"
    pip install flash-attn --no-build-isolation
    pip install "faesm[flash_attn]"
    pip install bacformer

Reads:
    data/genome_accessions.tsv     (bacdive_id, accession, ...)
    <genomes_dir>/                  FASTA or GenBank files named by accession.
                                    Accepted: <acc>.fna, <acc>.fna.gz,
                                    <acc>.gbff, <acc>.gbff.gz, or NCBI-Datasets
                                    nested layout <acc>/<acc>_genomic.<ext>.

Writes:
    data/bacformer_features.npz with:
        bacdive_ids: int64 [N]
        features:    float32 [N, embed_dim]
        accessions:  unicode <U24 [N]

Resumable: skips genomes already in the output npz. Checkpoints every 50
genomes so a crash loses at most 50 embeddings.

Usage:
    python compute_bacformer_features.py --genomes-dir data/genomes
    python compute_bacformer_features.py --genomes-dir data/genomes \\
        --model macwiatrak/bacformer-large-masked-MAG \\
        --model-type large \\
        --batch-prot 256 --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError:
    sys.exit("requires: pip install pandas pyarrow torch numpy")

try:
    from transformers import AutoModel
    from bacformer.pp import (  # type: ignore
        preprocess_genome_assembly,
        protein_seqs_to_bacformer_inputs,
    )
except ImportError:
    sys.exit(
        "bacformer not installed. Install with:\n"
        "    pip install 'torch>=2.2'\n"
        "    pip install flash-attn --no-build-isolation\n"
        "    pip install 'faesm[flash_attn]'\n"
        "    pip install bacformer"
    )


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_ACCESSIONS = DATA_DIR / "genome_accessions.tsv"
DEFAULT_OUT = DATA_DIR / "bacformer_features.npz"
ACC_DTYPE = "<U24"  # accessions are typically GCA_XXXXXXXXX.N — 16 chars max; allow headroom


def find_genome_file(genomes_dir: Path, accession: str) -> Path | None:
    """Locate a genome file for the accession across common naming conventions."""
    for ext in (".gbff.gz", ".gbff", ".gbk.gz", ".gbk", ".fna.gz", ".fna"):
        p = genomes_dir / f"{accession}{ext}"
        if p.exists():
            return p
        nested = genomes_dir / accession / f"{accession}_genomic{ext}"
        if nested.exists():
            return nested
    return None


def embed_one(
    model, device: str, genome_path: Path, max_n_proteins: int, model_type: str, batch_prot: int
) -> np.ndarray:
    """Run one genome through Bacformer; return [embed_dim] mean-pooled embedding."""
    genome_info = preprocess_genome_assembly(filepath=str(genome_path))
    inputs = protein_seqs_to_bacformer_inputs(
        genome_info["protein_sequence"],
        device=device,
        batch_size=batch_prot,
        max_n_proteins=max_n_proteins,
        bacformer_model_type=model_type,
    )
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
    return emb


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
    parser.add_argument("--genomes-dir", type=Path, required=True, help="Directory of downloaded genome files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--model",
        default="macwiatrak/bacformer-large-masked-complete-genomes",
        help="HF model id; use the -MAG variant for metagenome-assembled genomes",
    )
    parser.add_argument("--model-type", choices=["base", "large"], default="large")
    parser.add_argument("--max-proteins", type=int, default=6000, help="Max proteins per genome (Bacformer trained at 6000)")
    parser.add_argument("--batch-prot", type=int, default=128, help="Batch size for the ESM protein encoder")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N new genomes")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    if not args.accessions.exists():
        sys.exit(f"Accessions file not found: {args.accessions}. Run extract_genome_accessions.py first.")
    if not args.genomes_dir.exists():
        sys.exit(f"Genomes directory not found: {args.genomes_dir}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu":
        print("WARNING: running on CPU. Bacformer is very slow on CPU; expect minutes per genome.")
    else:
        print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else ''})")

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
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device).to(dtype)
    model.train(False)

    n_ok = n_missing = n_failed = 0
    for i, row in enumerate(todo.itertuples(), start=1):
        bid = int(row.bacdive_id)
        acc = str(row.accession)
        path = find_genome_file(args.genomes_dir, acc)
        if path is None:
            n_missing += 1
            continue
        try:
            emb = embed_one(model, device, path, args.max_proteins, args.model_type, args.batch_prot)
        except Exception as e:
            print(f"  [warn] {bid} ({acc}) failed: {type(e).__name__}: {e}", flush=True)
            n_failed += 1
            continue
        ids_out.append(bid)
        feats_out.append(emb)
        accs_out.append(acc)
        n_ok += 1

        if i % args.checkpoint_every == 0:
            save_npz(args.out, ids_out, feats_out, accs_out)
            print(
                f"  [{i:>6,}/{len(todo):,}]  ok={n_ok:,}  missing={n_missing:,}  failed={n_failed:,}  "
                f"checkpoint at {args.out}",
                flush=True,
            )

    save_npz(args.out, ids_out, feats_out, accs_out)
    print(f"\ndone. {len(ids_out):,} total embeddings in {args.out}")
    print(f"  this run: ok={n_ok:,}  missing={n_missing:,}  failed={n_failed:,}")


if __name__ == "__main__":
    main()
