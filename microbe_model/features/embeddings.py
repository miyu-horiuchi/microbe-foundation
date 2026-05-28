"""Per-genome ESM-2 embeddings.

Pipeline:
  1. Predict CDS via pyrodigal (reuses features.genome.predict_genes)
  2. For each protein: ESM-2 -> per-residue 320/640/1280-dim -> mean-pool over residues
  3. Mean-pool across all proteins in genome -> one fixed-dim vector per genome

Why ESM-2 specifically:
  - Reuses existing pyrodigal-predicted proteins (no DNA-window re-design)
  - Variants from 8M (laptop) to 3B params (cluster) -> easy to scale
  - Industry-standard for protein phenotype tasks
  - Mean-pool across residues + across proteome is the dumb-but-effective baseline

Model choices (set via env or argument):
  - facebook/esm2_t6_8M_UR50D    ->  320-dim, fast (laptop testing)
  - facebook/esm2_t12_35M_UR50D  ->  480-dim
  - facebook/esm2_t30_150M_UR50D ->  640-dim (recommended for GPU)
  - facebook/esm2_t33_650M_UR50D -> 1280-dim (best, needs GPU + 8GB+ VRAM)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
ESM2_MAX_LEN = 1024  # ESM-2's positional embedding limit; longer proteins are truncated


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_esm2(model_name: str = DEFAULT_MODEL, device: torch.device | None = None) -> tuple[Any, Any, torch.device]:
    """Load tokenizer + model on the best available device. Inference mode, fp16 on cuda."""
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.train(False)  # inference mode (equivalent to model.eval())
    return tokenizer, model, device


@torch.inference_mode()
def embed_proteins(
    proteins: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    batch_size: int = 8,
    max_len: int = ESM2_MAX_LEN,
) -> np.ndarray:
    """Mean-pool the per-residue ESM-2 embeddings of each protein.

    Returns (n_proteins, embed_dim) float32 array.
    """
    if not proteins:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    out: list[np.ndarray] = []
    for i in range(0, len(proteins), batch_size):
        batch = proteins[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        last_hidden = outputs.last_hidden_state  # (B, L, D)
        attention_mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_genome(
    proteins: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    sample_n: int | None = None,
    batch_size: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return one fixed-dim vector summarizing the whole proteome.

    If ``sample_n`` is set, only that many proteins are embedded (uniformly sampled
    without replacement) to bound runtime. None = embed every protein.
    """
    if not proteins:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    if sample_n is not None and sample_n < len(proteins):
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(proteins), size=sample_n, replace=False)
        proteins = [proteins[i] for i in idx]

    matrix = embed_proteins(proteins, tokenizer, model, device, batch_size=batch_size)
    return matrix.mean(axis=0).astype(np.float32)
