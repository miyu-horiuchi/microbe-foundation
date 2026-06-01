"""
modal_bacformer.py — Bacformer genome-level embeddings on Modal.

Each container processes a batch of (bacdive_id, accession) pairs:
  fetch FASTA  →  pyrodigal predict proteins  →  ESM-2 per-protein embed
              →  Bacformer transformer aggregation  →  one genome vector

Bacformer (Wiatrak et al. 2025) is a pretrained transformer over per-protein
embeddings — beats mean-pooling because it learns content-aware aggregation
on millions of MAGs.

Run:
    modal run modal_bacformer.py --limit 50               # smoke test
    modal run modal_bacformer.py                          # full corpus
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Image — Python 3.10 + GPU torch + Bacformer + ESM-2 + pyrodigal
# ---------------------------------------------------------------------------

image = (
    # CUDA-devel base because flash-attn compiles a C++/CUDA extension at install time
    # and needs nvcc + the full toolkit, which debian_slim doesn't have.
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10",
    )
    .apt_install("git", "build-essential", "wget")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",
        "torch==2.4.0",
        "numpy<2",
        "pandas",
        "pyrodigal>=3.5",
        "biopython",
        "requests",
        "transformers>=4.40",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("faesm[flash_attn]")
    .pip_install("bacformer")
    .add_local_dir(str(ROOT / "microbe_model"), "/root/microbe_model", copy=True)
)

# Volume to cache the Bacformer + ESM-2 HuggingFace checkpoints so each
# container doesn't redownload (~2-3 GB total).
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_HOME = "/root/.cache/huggingface"

app = modal.App("microbe-bacformer", image=image)


# ---------------------------------------------------------------------------
# Per-genome embedding helper — module-level for ProcessPoolExecutor pickle
# ---------------------------------------------------------------------------

def _fetch_predict_one(bid_acc):
    """Worker: fetch genome FASTA + pyrodigal predict, return (bid, list[protein_seq])."""
    import sys
    sys.path.insert(0, "/root")
    from microbe_model.features.genome import predict_genes
    from microbe_model.pipeline import _fetch_fasta_bytes
    bid, acc = bid_acc
    try:
        contigs = _fetch_fasta_bytes(acc)
    except Exception:
        return (bid, acc, None, "FETCH_FAIL")
    if not contigs:
        return (bid, acc, None, "FETCH_FAIL")
    try:
        proteins, _cds, _nt = predict_genes(contigs)
    except Exception as e:
        return (bid, acc, None, f"PRED_FAIL:{type(e).__name__}")
    if not proteins:
        return (bid, acc, None, "PRED_FAIL:empty")
    return (bid, acc, proteins, "OK")


# ---------------------------------------------------------------------------
# Batch processor — runs on GPU container
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",  # 40 GB VRAM, $2.10/hr — ~3x faster than L4 at similar total cost given the 10-GPU concurrent cap
    cpu=8,
    memory=32768,
    secrets=[modal.Secret.from_name("ncbi-api-key")],
    volumes={HF_HOME: hf_cache},
    timeout=60 * 60,
    retries=2,
    max_containers=100,
)
def process_batch(pairs: list[tuple[int, str]], max_proteins: int = 6000) -> dict:
    """Process a batch of genomes → dict {bid -> {accession, features (list of floats)}}."""
    import os, sys, time
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    import torch
    sys.path.insert(0, "/root")

    # Phase A: parallel fetch + predict (CPU-bound)
    t0 = time.time()
    fetched: list[tuple[int, str, list[str]]] = []
    n_ff = n_pf = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        for bid, acc, proteins, status in ex.map(_fetch_predict_one, pairs):
            if status == "OK":
                fetched.append((bid, acc, proteins))
            elif status == "FETCH_FAIL":
                n_ff += 1
            else:
                n_pf += 1
    fetch_dt = time.time() - t0
    print(f"  fetch+predict: {len(fetched)} ok, {n_ff} ff, {n_pf} pf in {fetch_dt:.0f}s", flush=True)
    if not fetched:
        return {}

    # Phase B: load Bacformer (cached in HF volume after first run)
    t_load = time.time()
    from transformers import AutoModel
    from bacformer.pp import protein_seqs_to_bacformer_inputs
    model_id = "macwiatrak/bacformer-large-masked-MAG"
    model_type = "large"  # this checkpoint is the 'large' variant
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cuda").eval()
    print(f"  bacformer loaded in {time.time()-t_load:.0f}s", flush=True)

    # Phase C: per-genome Bacformer forward pass — NaN-safe pooling
    out: dict[int, dict] = {}
    embed_t0 = time.time()
    n_nan_skipped = 0
    for bid, acc, proteins in fetched:
        if len(proteins) > max_proteins:
            proteins = proteins[:max_proteins]
        try:
            inputs = protein_seqs_to_bacformer_inputs(
                proteins,
                device="cuda",
                batch_size=128,
                max_n_proteins=max_proteins,
                bacformer_model_type=model_type,
            )
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
            # Cast to fp32 before any reduction — Bacformer's bf16/fp16 internals
            # routinely produce NaN at mean-reduction time on certain genomes.
            hidden = outputs.last_hidden_state.float()  # [1, N, hidden]

            # Apply attention mask if available, to mean-pool only real (non-pad) positions.
            mask = inputs.get("attention_mask")
            if mask is not None:
                mask = mask.float().unsqueeze(-1)  # [1, N, 1]
                denom = mask.sum(dim=1).clamp(min=1)  # [1, 1]
                emb = (hidden * mask).sum(dim=1) / denom
            else:
                emb = hidden.mean(dim=1)
            emb = emb.squeeze(0).cpu().numpy()

            if not np.all(np.isfinite(emb)):
                n_nan_skipped += 1
                if n_nan_skipped <= 3:
                    n_proteins = len(proteins)
                    hidden_finite = bool(torch.isfinite(hidden).all().item())
                    print(f"  [NaN] bid={bid} n_proteins={n_proteins} "
                          f"hidden_finite={hidden_finite}", flush=True)
                continue
            out[int(bid)] = {"accession": acc, "features": emb.tolist()}
        except Exception as e:
            print(f"  [warn] {bid} bacformer failed: {type(e).__name__}: {e}", flush=True)
    if n_nan_skipped:
        print(f"  [NaN summary] skipped {n_nan_skipped} NaN-producing genomes in this batch", flush=True)
    embed_dt = time.time() - embed_t0
    print(f"  bacformer embed: {len(out)}/{len(fetched)} in {embed_dt:.0f}s "
          f"(total batch {time.time()-t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Local driver
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = 25, only_bids_file: str = "",
         out_name: str = "bacformer_features.npz"):
    """Read genome_accessions.tsv, .map() over batches, write data/bacformer_features.npz.

    --only-bids-file: path to a text file with one bacdive_id per line; only those
    genomes will be processed (used for re-running NaN failures).
    --out-name: name of output file under data/ (default bacformer_features.npz).
    """
    import pandas as pd
    import numpy as np

    acc_path = ROOT / "data" / "genome_accessions.tsv"
    df = pd.read_csv(acc_path, sep="\t")
    if only_bids_file:
        wanted = set(int(line.strip()) for line in open(only_bids_file) if line.strip())
        df = df[df.bacdive_id.astype(int).isin(wanted)].reset_index(drop=True)
        print(f"--only-bids-file: filtered to {len(df):,} genomes from {only_bids_file}")
    if limit:
        df = df.head(limit)
    pairs = [(int(r.bacdive_id), str(r.accession)) for r in df.itertuples()]
    batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]
    print(f"dispatching {len(batches)} batches of up to {batch_size} genomes "
          f"(total {len(pairs):,} genomes)", flush=True)

    all_results: dict[int, dict] = {}
    t0 = time.time()
    n_done = 0
    for batch_result in process_batch.map(batches, order_outputs=False, return_exceptions=False):
        all_results.update({int(k): v for k, v in batch_result.items()})
        n_done += 1
        dt = time.time() - t0
        rate = n_done / max(dt, 1e-6) * batch_size
        eta_min = (len(batches) - n_done) * dt / max(n_done, 1) / 60
        print(f"  [{n_done}/{len(batches)}] {len(all_results):,} genomes done, "
              f"rate={rate:.1f} g/sec, eta={eta_min:.1f}min", flush=True)

    print(f"\nall batches done. {len(all_results):,} genomes embedded.", flush=True)

    bids = sorted(all_results.keys())
    feats = np.array([all_results[b]["features"] for b in bids], dtype=np.float32)
    accs = np.array([all_results[b]["accession"] for b in bids], dtype="<U24")
    out_path = ROOT / "data" / out_name
    np.savez(out_path,
             bacdive_ids=np.array(bids, dtype=np.int64),
             features=feats,
             accessions=accs)
    print(f"wrote {out_path}  shape={feats.shape}", flush=True)
