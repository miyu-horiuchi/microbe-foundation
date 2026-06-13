"""
model.py — multi-task model with masked loss for the microbe-foundation benchmark.

Reads:
    data/traits.parquet         (one row per strain, one column per trait)
    data/splits.parquet         (taxonomic splits per strain)
    data/vocabularies.json      (class vocabularies per head)
    trait_schema.json           (head types per trait)
    <features.npz>              (feature matrix; provide with --features, or
                                 omit to run a random-feature smoke test)

Architecture:
    Input features                          [B, D_feat]
       shared MLP encoder                   [B, hidden//2]
           per-trait linear heads
              - binary       -> 1 logit
              - multiclass   -> K logits
              - multilabel   -> K logits
              - regression_v -> K scalars

Loss:
    Per-head masked loss, summed across heads with equal weight (uniform).
    Masks come from label presence in the parquet; for regression-vector
    (FAME), per-element mask is also applied so only reported FAMEs incur
    loss.

The model is feature-source-agnostic. Features can be:
    - KO presence/absence matrix from eggNOG-mapper
    - ESM-2 mean-pooled embeddings
    - Bacformer embeddings
    - Random (smoke test)
The features.npz file must contain:
    bacdive_ids: int array [N]
    features:    float array [N, D_feat]

Usage:
    python model.py                          # smoke test with random features
    python model.py --features data/koembed.npz --epochs 30
    python model.py --split-level family --epochs 50 --hidden 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:
    sys.exit("requires: pip install pandas pyarrow torch numpy")

# With num_workers>0 the per-protein loader ships large padded batch tensors
# between processes. PyTorch's default 'file_descriptor' strategy burns one FD
# per shared tensor and exhausts the (often 1024) open-file limit -> workers die
# with ConnectionRefusedError. 'file_system' shares by name and avoids that.
if hasattr(torch, "multiprocessing"):
    torch.multiprocessing.set_sharing_strategy("file_system")


DATA_DIR = Path(__file__).parent / "data"
SCHEMA_PATH = Path(__file__).parent / "trait_schema.json"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
TRAITS_PATH = DATA_DIR / "traits.parquet"
SPLITS_PATH = DATA_DIR / "splits.parquet"


# =============================================================================
# Label preparation: parquet + vocab -> dense tensors + masks
# =============================================================================


def _vocab_index(vocab_block: dict) -> dict[str, int]:
    """Build {class_value -> index} from a vocab block's items list."""
    return {item["value"]: i for i, item in enumerate(vocab_block.get("items", []))}


def prepare_labels(
    df: pd.DataFrame, vocab: dict, schema: dict
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict]]:
    """
    Materialize per-head label tensors + masks from the traits parquet.

    Returns three dicts keyed by trait name:
        labels[name]: tensor with appropriate shape and dtype
        masks[name]:  same shape; 1 where labeled, 0 where missing
        specs[name]:  {head_type, size, vocab_index (if applicable)}
    """
    n = len(df)
    labels: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    specs: dict[str, dict] = {}
    vocabs = vocab["vocabularies"]

    for trait in schema["traits"]:
        name = trait["name"]
        head = trait["head"]
        if name not in df.columns:
            continue

        col = df[name]

        if head == "binary":
            y = torch.zeros(n, dtype=torch.float32)
            m = torch.zeros(n, dtype=torch.float32)
            for i, v in enumerate(col):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                y[i] = float(bool(v))
                m[i] = 1.0
            labels[name] = y
            masks[name] = m
            specs[name] = {"head_type": "binary", "size": 1}

        elif head == "multiclass":
            classes = trait.get("classes") or [it["value"] for it in vocabs.get(name, {}).get("items", [])]
            idx = {c: i for i, c in enumerate(classes)}
            y = torch.full((n,), -1, dtype=torch.long)
            m = torch.zeros(n, dtype=torch.float32)
            for i, v in enumerate(col):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                vs = str(v)
                if vs in idx:
                    y[i] = idx[vs]
                    m[i] = 1.0
            labels[name] = y
            masks[name] = m
            specs[name] = {"head_type": "multiclass", "size": len(classes), "classes": classes}

        elif head == "multilabel":
            voc = vocabs.get(name, {})
            vidx = _vocab_index(voc)
            k = len(vidx)
            y = torch.zeros((n, k), dtype=torch.float32)
            m = torch.zeros((n, k), dtype=torch.float32)
            for i, v in enumerate(col):
                if v is None:
                    continue
                if isinstance(v, dict):
                    if not v:
                        continue
                    for key, val in v.items():
                        if key in vidx and val is not None:
                            m[i, vidx[key]] = 1.0
                            if isinstance(val, bool):
                                y[i, vidx[key]] = float(val)
                            elif isinstance(val, str):
                                y[i, vidx[key]] = 1.0 if val.upper() == "R" else 0.0
                elif hasattr(v, "__iter__") and not isinstance(v, str):
                    items = [str(x) for x in v if x is not None]
                    if not items:
                        continue
                    # List-typed multilabel (e.g., cultivation_medium): observed
                    # positives only. Use the implicit-negative assumption — if
                    # the strain has *any* cultivation_medium data, treat unlisted
                    # vocab items as negatives. Without this the loss only sees
                    # positives and the model learns to predict everything.
                    m[i, :] = 1.0
                    for item in items:
                        if item in vidx:
                            y[i, vidx[item]] = 1.0
            labels[name] = y
            masks[name] = m
            specs[name] = {"head_type": "multilabel", "size": k, "vocab_index": vidx}

        elif head == "regression_vector":
            voc = vocabs.get(name, {})
            vidx = _vocab_index(voc)
            k = len(vidx)
            y = torch.zeros((n, k), dtype=torch.float32)
            m = torch.zeros((n, k), dtype=torch.float32)
            for i, v in enumerate(col):
                if not isinstance(v, dict) or not v:
                    continue
                for key, val in v.items():
                    if key in vidx and val is not None:
                        try:
                            y[i, vidx[key]] = float(val)
                            m[i, vidx[key]] = 1.0
                        except (TypeError, ValueError):
                            pass
            y = y / 100.0  # FAME percentages -> fractions for scale-stable MSE
            labels[name] = y
            masks[name] = m
            specs[name] = {"head_type": "regression_vector", "size": k, "vocab_index": vidx}

    return labels, masks, specs


# =============================================================================
# Model
# =============================================================================


class MeanProteinPool(nn.Module):
    """Masked mean pooling over a variable-length set of protein vectors."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


class MaxProteinPool(nn.Module):
    """Masked max pooling over proteins."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        pooled = masked.max(dim=1).values
        return torch.nan_to_num(pooled, neginf=0.0)


class TopKProteinPool(nn.Module):
    """Learned top-k attention pooling over the highest-scoring real proteins."""

    def __init__(self, dim: int, k: int = 8):
        super().__init__()
        self.k = k
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1).masked_fill(mask == 0, float("-inf"))
        k = min(self.k, scores.shape[1])
        vals, idx = torch.topk(scores, k=k, dim=1)
        gathered = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        attn = torch.softmax(vals, dim=1)
        attn = torch.nan_to_num(attn)
        return (gathered * attn.unsqueeze(-1)).sum(dim=1)


class AttentionPool(nn.Module):
    """Learned attention pooling over a variable-length set of protein vectors.

    Collapses a genome's per-protein embeddings [B, P, D] into one [B, D] vector
    by learning a weight per protein, instead of a flat mean. A small scorer maps
    each protein to a scalar; a (padding-masked) softmax over the P proteins turns
    those into weights that sum to 1; the output is the weighted sum.

    This is what lets the model up-weight the few trait-determining proteins
    (e.g. a catalase) and down-weight the thousands of irrelevant ones — the whole
    point of keeping proteins un-pooled.
    """

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Interpretability hook: when store_attn is True, each forward stashes the
        # per-protein softmax weights in last_attn (detached) so an extraction pass
        # can read which proteins the model up-weighted. Off during training (no cost).
        self.store_attn = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, P, D] protein embeddings. mask: [B, P], 1=real protein, 0=padding."""
        scores = self.score(x).squeeze(-1)            # [B, P]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)           # [B, P], sums to 1 over real proteins
        attn = torch.nan_to_num(attn)                 # guard: genome with 0 proteins -> all-zero weights
        if self.store_attn:
            self.last_attn = attn.detach()
        return (x * attn.unsqueeze(-1)).sum(dim=1)    # [B, D]


class GatedAttentionPool(nn.Module):
    """Gated multiple-instance attention pooling from Ilse et al."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.tanh_branch = nn.Linear(dim, hidden)
        self.gate_branch = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)
        self.store_attn = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.tanh_branch(x)) * torch.sigmoid(self.gate_branch(x))
        scores = self.score(gated).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        attn = torch.nan_to_num(attn)
        if self.store_attn:
            self.last_attn = attn.detach()
        return (x * attn.unsqueeze(-1)).sum(dim=1)


class _MAB(nn.Module):
    """Multihead Attention Block (Set Transformer, Lee et al. 2019), mask-aware.

    MAB(Q, K) = LayerNorm(H + rFF(H)), H = LayerNorm(Q_proj + Multihead(Q, K, K)).
    `key_mask` ([B, Nk], 1=real, 0=padding) blocks attention onto padded keys, so
    ragged protein sets pool correctly. Heads are folded into the batch dim.
    """

    def __init__(self, dim_q: int, dim_kv: int, dim_out: int, num_heads: int):
        super().__init__()
        if dim_out % num_heads != 0:
            raise ValueError(f"dim_out={dim_out} not divisible by num_heads={num_heads}")
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim_out)
        self.fc_k = nn.Linear(dim_kv, dim_out)
        self.fc_v = nn.Linear(dim_kv, dim_out)
        self.fc_o = nn.Linear(dim_out, dim_out)
        self.ln0 = nn.LayerNorm(dim_out)
        self.ln1 = nn.LayerNorm(dim_out)

    def forward(self, Q, K, key_mask=None, return_attn=False):
        B = Q.shape[0]
        Qp, Kp, Vp = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        ds = self.dim_out // self.num_heads
        Qh = torch.cat(Qp.split(ds, 2), 0)            # [H*B, Nq, ds]
        Kh = torch.cat(Kp.split(ds, 2), 0)            # [H*B, Nk, ds]
        Vh = torch.cat(Vp.split(ds, 2), 0)
        scores = Qh.bmm(Kh.transpose(1, 2)) / math.sqrt(ds)   # [H*B, Nq, Nk]
        if key_mask is not None:
            km = key_mask.repeat(self.num_heads, 1).unsqueeze(1)  # [H*B, 1, Nk]
            scores = scores.masked_fill(km == 0, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=2))     # guard all-pad rows
        O = Qh + attn.bmm(Vh)
        O = torch.cat(O.split(B, 0), 2)               # back to [B, Nq, dim_out]
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        if return_attn:
            # [H*B, Nq, Nk] -> mean over heads -> [B, Nq, Nk]
            attn_bhn = attn.view(self.num_heads, B, attn.shape[1], attn.shape[2]).mean(0)
            return O, attn_bhn
        return O


class _ISAB(nn.Module):
    """Induced Set-Attention Block: O(P*m) self-attention via m inducing points.

    Lets proteins attend to one another (through the inducing bottleneck) so the
    representation can encode joint presence / interactions, unlike the weighted
    sum of AttentionPool.
    """

    def __init__(self, dim: int, num_heads: int, num_inducing: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, num_inducing, dim))
        nn.init.xavier_uniform_(self.inducing)
        self.mab0 = _MAB(dim, dim, dim, num_heads)   # inducing points attend to proteins
        self.mab1 = _MAB(dim, dim, dim, num_heads)   # proteins attend to inducing summary

    def forward(self, X, mask):
        I = self.inducing.expand(X.shape[0], -1, -1)
        H = self.mab0(I, X, key_mask=mask)           # [B, m, dim]; proteins masked
        return self.mab1(X, H)                        # [B, P, dim]


class _PMA(nn.Module):
    """Pooling by Multihead Attention: k seed vectors attend over the set."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.mab = _MAB(dim, dim, dim, num_heads)

    def forward(self, Z, mask, return_attn=False):
        S = self.seeds.expand(Z.shape[0], -1, -1)
        return self.mab(S, Z, key_mask=mask, return_attn=return_attn)


class SetTransformerPool(nn.Module):
    """Set Transformer pooler (1 ISAB + PMA, single seed) -> one genome vector.

    Drop-in successor to AttentionPool specified in the manuscript (§6). Unlike a
    weighted sum, ISAB lets proteins interact, so the genome vector can reflect
    protein *combinations*. The PMA seed-to-protein attention is exposed via
    `store_attn`/`last_attn` ([B, P]) so the §5 top-k attribution still works.
    """

    def __init__(self, dim: int, num_heads: int = 4, num_inducing: int = 16):
        super().__init__()
        self.isab = _ISAB(dim, num_heads, num_inducing)
        self.pma = _PMA(dim, num_heads, num_seeds=1)
        self.store_attn = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        H = self.isab(x, mask)                        # [B, P, D]
        if self.store_attn:
            pooled, attn = self.pma(H, mask, return_attn=True)  # attn: [B, 1, P]
            self.last_attn = attn.squeeze(1).detach()           # [B, P]
            return pooled.squeeze(1)
        return self.pma(H, mask).squeeze(1)           # [B, D]


def make_pooler(name: str, input_dim: int, topk: int = 8,
                st_heads: int = 4, st_inducing: int = 16) -> nn.Module:
    if name == "mean":
        return MeanProteinPool()
    if name == "max":
        return MaxProteinPool()
    if name == "topk":
        return TopKProteinPool(input_dim, k=topk)
    if name == "attention":
        return AttentionPool(input_dim)
    if name == "gated_attention":
        return GatedAttentionPool(input_dim)
    if name == "set_transformer":
        return SetTransformerPool(input_dim, num_heads=st_heads, num_inducing=st_inducing)
    raise ValueError(f"unknown pooling mode: {name}")


class MicrobeFoundationModel(nn.Module):
    """Shared MLP encoder + per-trait linear heads.

    When `attention_pool=True`, forward expects per-protein input [B, P, D] plus a
    [B, P] padding mask, and pools it to [B, D] before the encoder. When False
    (default), forward takes a pre-pooled [B, D] feature matrix exactly as before —
    the 2-D .npz path (ESM-2 mean-pool, eggNOG, etc.) is unchanged.
    """

    def __init__(self, input_dim: int, head_specs: dict[str, dict], hidden: int = 512,
                 dropout: float = 0.2, attention_pool: bool = False,
                 pooling: str | None = None, topk: int = 8,
                 st_heads: int = 4, st_inducing: int = 16):
        super().__init__()
        if attention_pool and pooling is None:
            pooling = "attention"
        self.pooling = pooling
        self.pool = (make_pooler(pooling, input_dim, topk=topk,
                                 st_heads=st_heads, st_inducing=st_inducing)
                     if pooling else None)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        h_out = hidden // 2
        self.heads = nn.ModuleDict()
        for name, spec in head_specs.items():
            self.heads[name] = nn.Linear(h_out, spec["size"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.pool is not None:
            x = self.pool(x, mask)
        h = self.encoder(x)
        return {name: head(h) for name, head in self.heads.items()}


# =============================================================================
# Masked multi-task loss
# =============================================================================


def compute_class_weights(
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    specs: dict[str, dict],
    train_idx: torch.Tensor,
    threshold: float = 0.0,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute per-head loss-balancing weights from the training subset only.

    Only heads whose imbalance ratio exceeds `threshold` get weights; the rest
    fall back to plain unweighted loss. Imbalance ratio is defined per head:
      binary    : max(pos, neg) / max(min(pos, neg), 1)
      multiclass: max class count / max(min nonzero class count, 1)
      multilabel: median of (neg/pos) over labels with pos > 0

    Weights:
      binary    : scalar pos_weight = neg/pos, clamped to [1, 50]
      multiclass: per-class weight = total / (K * count_c), capped at 50
      multilabel: per-label pos_weight = neg/pos per label, clamped to [1, 50]
      regression_vector: no weighting
    """
    out: dict[str, torch.Tensor] = {}
    for name, spec in specs.items():
        h = spec["head_type"]
        y = labels[name][train_idx]
        m = masks[name][train_idx]

        if h == "binary":
            valid = m > 0
            if valid.sum() == 0:
                continue
            pos = ((y == 1) & valid).sum().clamp(min=1).float()
            neg = ((y == 0) & valid).sum().clamp(min=1).float()
            ratio = (torch.maximum(pos, neg) / torch.minimum(pos, neg).clamp(min=1)).item()
            if ratio < threshold:
                if verbose:
                    print(f"    {name:<24}  binary  ratio={ratio:.2f}  -> skip (< {threshold})")
                continue
            pw = (neg / pos).clamp(min=1.0, max=10.0)
            out[name] = pw
            if verbose:
                print(f"    {name:<24}  binary  ratio={ratio:.2f}  pos_weight={pw.item():.2f}")
        elif h == "multiclass":
            valid = m > 0
            if valid.sum() == 0:
                continue
            k = spec["size"]
            counts = torch.zeros(k)
            for c in range(k):
                counts[c] = ((y == c) & valid).sum()
            nonzero = counts[counts > 0]
            if len(nonzero) < 2:
                continue
            ratio = (nonzero.max() / nonzero.min().clamp(min=1)).item()
            if ratio < threshold:
                if verbose:
                    print(f"    {name:<24}  multiclass  ratio={ratio:.2f}  -> skip")
                continue
            total = counts.sum().clamp(min=1)
            w = torch.zeros(k)
            seen = counts > 0
            w[seen] = total / (seen.sum().float() * counts[seen])
            w = w.clamp(max=10.0)
            out[name] = w
            if verbose:
                print(f"    {name:<24}  multiclass  ratio={ratio:.2f}  weights range "
                      f"[{w[seen].min().item():.2f}, {w[seen].max().item():.2f}]")
        elif h == "multilabel":
            if m.sum() == 0:
                continue
            pos_per_label = ((y == 1) & (m > 0)).sum(0).float()
            neg_per_label = ((y == 0) & (m > 0)).sum(0).float()
            seen = pos_per_label > 0
            if seen.sum() < 2:
                continue
            ratios = neg_per_label[seen] / pos_per_label[seen].clamp(min=1)
            ratio = float(ratios.median().item())
            if ratio < threshold:
                if verbose:
                    print(f"    {name:<24}  multilabel  median_ratio={ratio:.2f}  -> skip")
                continue
            pw = (neg_per_label / pos_per_label.clamp(min=1)).clamp(min=1.0, max=10.0)
            out[name] = pw
            if verbose:
                print(f"    {name:<24}  multilabel  median_ratio={ratio:.2f}  "
                      f"pos_weight range [{pw[seen].min().item():.2f}, {pw[seen].max().item():.2f}]")
    return out


def masked_loss(
    preds: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    specs: dict[str, dict],
    weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Per-head masked loss, equally weighted across heads. Heads with zero
    labeled samples in the batch contribute zero loss and are skipped.
    Optional `weights` carries class-balancing weights from
    compute_class_weights().
    """
    device = next(iter(preds.values())).device
    total = torch.tensor(0.0, device=device)
    n_active = 0
    per_head: dict[str, float] = {}
    weights = weights or {}

    for name, pred in preds.items():
        if name not in labels:
            continue
        y = labels[name].to(device)
        m = masks[name].to(device)
        spec = specs[name]
        h = spec["head_type"]

        if h == "binary":
            valid = m > 0
            if valid.sum() == 0:
                continue
            pw = weights.get(name)
            if pw is not None:
                pw = pw.to(device)
            loss = F.binary_cross_entropy_with_logits(
                pred[valid].squeeze(-1), y[valid], pos_weight=pw,
            )
        elif h == "multiclass":
            valid = m > 0
            if valid.sum() == 0:
                continue
            w = weights.get(name)
            if w is not None:
                w = w.to(device)
            loss = F.cross_entropy(pred[valid], y[valid], weight=w)
        elif h == "multilabel":
            if m.sum() == 0:
                continue
            pw = weights.get(name)
            if pw is not None:
                pw = pw.to(device)
            elem_loss = F.binary_cross_entropy_with_logits(
                pred, y, reduction="none", pos_weight=pw,
            )
            loss = (elem_loss * m).sum() / m.sum().clamp(min=1.0)
        elif h == "regression_vector":
            if m.sum() == 0:
                continue
            elem_loss = (pred - y) ** 2
            loss = (elem_loss * m).sum() / m.sum().clamp(min=1.0)
        else:
            continue

        total = total + loss
        n_active += 1
        per_head[name] = loss.item()

    if n_active > 0:
        total = total / n_active
    return total, per_head


# =============================================================================
# Dataset
# =============================================================================


class StrainDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: dict, masks: dict):
        self.features = features
        self.labels = labels
        self.masks = masks

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            {k: v[idx] for k, v in self.labels.items()},
            {k: v[idx] for k, v in self.masks.items()},
        )


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    label_dict = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    mask_dict = {k: torch.stack([b[2][k] for b in batch]) for k in batch[0][2]}
    return feats, label_dict, mask_dict


class PerProteinDataset(Dataset):
    """Lazily loads one genome's [n_proteins, D] matrix from its .npy file.

    Avoids holding ~100 GB in RAM: only the genomes in the current batch are
    read from disk. `paths` is aligned row-for-row with `labels`/`masks`.
    """

    def __init__(self, paths: list[Path], labels: dict, masks: dict, max_proteins: int | None = None):
        self.paths = paths
        self.labels = labels
        self.masks = masks
        self.max_proteins = max_proteins

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx]).astype(np.float32)  # [P, D] (stored fp16)
        if self.max_proteins and arr.shape[0] > self.max_proteins:
            sel = np.random.choice(arr.shape[0], self.max_proteins, replace=False)
            arr = arr[sel]
        x = torch.from_numpy(arr)
        return (
            x,
            {k: v[idx] for k, v in self.labels.items()},
            {k: v[idx] for k, v in self.masks.items()},
        )


def collate_perprotein(batch):
    """Pad ragged [P, D] genomes to the batch's max P; build a [B, P] padding mask.

    Returns feats as a (padded, mask) tuple so the rest of the loop can tell the
    per-protein path apart from the pre-pooled one.
    """
    xs = [b[0] for b in batch]
    B = len(xs)
    P = max(x.shape[0] for x in xs)
    D = xs[0].shape[1]
    padded = torch.zeros(B, P, D, dtype=torch.float32)
    pmask = torch.zeros(B, P, dtype=torch.float32)
    for i, x in enumerate(xs):
        p = x.shape[0]
        padded[i, :p] = x
        pmask[i, :p] = 1.0
    label_dict = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    mask_dict = {k: torch.stack([b[2][k] for b in batch]) for k in batch[0][2]}
    return (padded, pmask), label_dict, mask_dict


def family_sample_weights(families) -> "np.ndarray":
    """Per-sample weights that equalize family representation in expectation.

    Each genome gets weight 1 / (F * n_f), where n_f is its family's size and F
    is the number of distinct families. A WeightedRandomSampler with these weights
    draws every family with equal expected mass regardless of how many genomes it
    has — directly counteracting the long-tailed family distribution that the
    cross-clade diagnostic (Table 15) identified as the cause of family-test
    collapse. Missing/None families are bucketed together as one group.
    """
    fam = np.array(["<NA>" if (f is None or (isinstance(f, float) and np.isnan(f))) else str(f)
                    for f in families])
    uniq, inv, counts = np.unique(fam, return_inverse=True, return_counts=True)
    n_families = len(uniq)
    return 1.0 / (n_families * counts[inv].astype(np.float64))


def _model_forward(model, feats, device):
    """Run the model on a batch's feats, handling both the pre-pooled ([B,D]) and
    per-protein ((padded [B,P,D], mask [B,P])) cases."""
    # pin_memory / default_convert can turn the (padded, mask) tuple into a list,
    # so accept either — a bare tensor is the pre-pooled path.
    if isinstance(feats, (tuple, list)):
        x, pmask = feats
        return model(x.to(device), pmask.to(device))
    return model(feats.to(device))


# =============================================================================
# Training / evaluation
# =============================================================================


def run_eval(model, loader, specs, device) -> dict[str, dict[str, float]]:
    """Compute per-head metrics on a split.

    Returns {head_name: {metric_kind: score, ...}}. binary/multiclass heads
    emit both `acc` and `f1` (macro for multiclass; positive-class for binary).
    Multilabel emits sample-averaged `f1`. Regression emits `rmse`.
    """
    model.train(False)
    # Accumulators per head — list of (pred, true, mask) tensors.
    buf: dict[str, dict[str, list]] = {}
    with torch.no_grad():
        for feats, labels, masks in loader:
            preds = _model_forward(model, feats, device)
            for name, pred in preds.items():
                spec = specs[name]
                h = spec["head_type"]
                y = labels[name].to(device)
                m = masks[name].to(device)
                if m.sum() == 0:
                    continue
                b = buf.setdefault(name, {"pred": [], "y": [], "m": []})
                b["pred"].append(pred.detach().cpu())
                b["y"].append(y.detach().cpu())
                b["m"].append(m.detach().cpu())

    out: dict[str, dict[str, float]] = {}
    for name, b in buf.items():
        h = specs[name]["head_type"]
        pred = torch.cat(b["pred"], dim=0)
        y = torch.cat(b["y"], dim=0)
        m = torch.cat(b["m"], dim=0)

        if h == "binary":
            valid = m > 0
            p = (torch.sigmoid(pred[valid].squeeze(-1)) > 0.5).long()
            t = y[valid].long()
            acc = (p == t).float().mean().item()
            tp = ((p == 1) & (t == 1)).sum().item()
            fp = ((p == 1) & (t == 0)).sum().item()
            fn = ((p == 0) & (t == 1)).sum().item()
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            out[name] = {"acc": acc, "f1": f1}

        elif h == "multiclass":
            valid = m > 0
            p = pred[valid].argmax(-1).long()
            t = y[valid].long()
            acc = (p == t).float().mean().item()
            k = specs[name]["size"]
            f1s = []
            for c in range(k):
                tp = ((p == c) & (t == c)).sum().item()
                fp = ((p == c) & (t != c)).sum().item()
                fn = ((p != c) & (t == c)).sum().item()
                if tp + fp + fn == 0:
                    continue  # class absent from this split; skip
                f1s.append(2 * tp / (2 * tp + fp + fn))
            f1_macro = sum(f1s) / max(len(f1s), 1)
            out[name] = {"acc": acc, "f1": f1_macro}

        elif h == "multilabel":
            probs = torch.sigmoid(pred)
            preds_bin = (probs > 0.5).long()
            yl = y.long()
            tp_s = ((preds_bin == 1) & (yl == 1) & (m > 0)).float().sum(-1)
            fp_s = ((preds_bin == 1) & (yl == 0) & (m > 0)).float().sum(-1)
            fn_s = ((preds_bin == 0) & (yl == 1) & (m > 0)).float().sum(-1)
            denom = 2 * tp_s + fp_s + fn_s
            f1_sample = torch.where(denom > 0, 2 * tp_s / denom.clamp(min=1), torch.zeros_like(denom))
            has = (m.sum(-1) > 0).float()
            f1 = (f1_sample * has).sum().item() / max(has.sum().item(), 1)
            # Macro F1 across labels: per-label TP/FP/FN summed across samples
            tp_l = ((preds_bin == 1) & (yl == 1) & (m > 0)).float().sum(0)
            fp_l = ((preds_bin == 1) & (yl == 0) & (m > 0)).float().sum(0)
            fn_l = ((preds_bin == 0) & (yl == 1) & (m > 0)).float().sum(0)
            denom_l = 2 * tp_l + fp_l + fn_l
            f1_per_label = torch.where(denom_l > 0, 2 * tp_l / denom_l.clamp(min=1), torch.zeros_like(denom_l))
            seen = denom_l > 0
            f1_macro = f1_per_label[seen].mean().item() if seen.any() else 0.0
            out[name] = {"f1": f1, "f1_macro": f1_macro}

        elif h == "regression_vector":
            sq = ((pred - y) ** 2 * m).sum().item()
            denom = m.sum().item()
            rmse = (sq / max(denom, 1)) ** 0.5
            out[name] = {"rmse": rmse}

    return out


def _primary_metric(head_type: str) -> str:
    return {
        "binary": "acc",
        "multiclass": "acc",
        "multilabel": "f1",
        "regression_vector": "rmse",
    }[head_type]


def predictions_to_frame(bacdive_ids, probs, trues, mask) -> "pd.DataFrame":
    """Per-genome predictions for labeled rows only, preserving order.

    bacdive_ids/probs/trues/mask are aligned 1:1 (loader order). Rows where mask
    is falsey (unlabeled) are dropped.
    """
    import pandas as pd

    keep = np.asarray(mask).astype(bool)
    return pd.DataFrame(
        {
            "bacdive_id": np.asarray(bacdive_ids)[keep],
            "true_label": np.asarray(trues)[keep],
            "pred": np.asarray(probs)[keep],
        }
    )


def collect_binary_predictions(model, loader, specs, device, head: str):
    """Return (probs, trues, mask) for one binary head in loader order.

    Requires an unshuffled loader so the row order matches the caller's id list.
    """
    model.train(False)
    probs, trues, masks = [], [], []
    with torch.no_grad():
        for feats, labels, batch_masks in loader:
            pred = _model_forward(model, feats, device)[head]
            probs.append(torch.sigmoid(pred.squeeze(-1)).detach().cpu().numpy())
            trues.append(labels[head].detach().cpu().numpy())
            masks.append(batch_masks[head].detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(trues), np.concatenate(masks)


def _avg_primary(metrics: dict[str, dict[str, float]], specs) -> float:
    vals = [m[_primary_metric(specs[name]["head_type"])] for name, m in metrics.items()]
    return sum(vals) / max(len(vals), 1)


def train(model, train_loader, val_loader, specs, device, epochs: int, lr: float,
          class_weights: dict | None = None, scheduler: str = "none",
          warmup_frac: float = 0.05):
    """val_loader may be None when the split has zero validation strains (small samples).

    scheduler: 'none' = constant lr; 'cosine' = linear warmup to `lr` over the
    first `warmup_frac` of total batches, then cosine decay to 0.
    """
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * warmup_frac)) if scheduler == "cosine" else 0

    def lr_at(step: int) -> float:
        if scheduler != "cosine":
            return lr
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    step = 0
    val_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train(True)
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for feats, labels, masks in train_loader:
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            preds = _model_forward(model, feats, device)
            loss, _ = masked_loss(preds, labels, masks, specs, weights=class_weights)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            n_batches += 1
            step += 1
        train_loss = running / max(n_batches, 1)
        if val_loader is not None:
            val_metrics = run_eval(model, val_loader, specs, device)
            avg_val = _avg_primary(val_metrics, specs)
            val_s = f"val_avg={avg_val:.4f}"
        else:
            val_s = "val=skipped(empty)"
        elapsed = time.time() - t0
        current_lr = optim.param_groups[0]["lr"]
        print(f"  epoch {epoch:>3}  lr={current_lr:.2e}  train_loss={train_loss:.4f}  {val_s}  ({elapsed:.1f}s)")
    return val_metrics


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=None, help="features.npz (bacdive_ids, features). Omit for random smoke test.")
    parser.add_argument("--per-protein", type=Path, default=None,
                        help="Directory of un-pooled per-protein embeddings (<bid>.npy + manifest.parquet) "
                             "from compute_esm2_perprotein_mp.py. Enables attention pooling. Overrides --features.")
    parser.add_argument("--pooling",
                        choices=["mean", "max", "topk", "attention", "gated_attention", "set_transformer"],
                        default="attention",
                        help="Pooling operator for --per-protein runs. Pre-pooled --features runs ignore this.")
    parser.add_argument("--topk", type=int, default=8,
                        help="Number of proteins to average in --pooling topk mode.")
    parser.add_argument("--st-heads", type=int, default=4,
                        help="Attention heads for --pooling set_transformer (must divide embed dim).")
    parser.add_argument("--st-inducing", type=int, default=16,
                        help="Number of inducing points for the set_transformer ISAB block.")
    parser.add_argument("--max-proteins", type=int, default=0,
                        help="With --per-protein, cap proteins per genome per batch (0 = no cap). Bounds GPU memory.")
    parser.add_argument("--split-level", choices=["species", "genus", "family"], default="family")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker processes. >0 parallelizes the per-protein .npy disk reads "
                             "so the GPU isn't starved (the lazy PerProteinDataset is I/O-bound otherwise).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feat-dim", type=int, default=128, help="Smoke-test feature dim (ignored if --features given)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-metrics", type=Path, default=None, help="Write final test metrics to this JSON path")
    parser.add_argument("--save-predictions", type=Path, default=None,
                        help="Write per-genome test predictions (bacdive_id,true_label,pred) to this parquet. "
                             "Requires exactly one binary head (use --single-task <trait>).")
    parser.add_argument("--save-model", type=Path, default=None,
                        help="Save the trained weights + rebuild config to this .pt path (for attention extraction)")
    parser.add_argument("--run-name", type=str, default="", help="Tag for the saved metrics (e.g., 'esm2-35M-family')")
    parser.add_argument("--class-weights", action="store_true",
                        help="Balance loss by inverse class frequency on training labels. "
                             "Improves macro F1 on imbalanced heads at the cost of plain accuracy.")
    parser.add_argument("--imbalance-threshold", type=float, default=0.0,
                        help="Only apply class weighting to heads whose imbalance ratio "
                             "(majority/minority) exceeds this. 0.0 = weight every head (default, "
                             "matches the original --class-weights behavior). 10.0 leaves balanced "
                             "heads alone and only weights the very skewed ones.")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none",
                        help="Optimizer LR schedule. cosine = warmup-then-decay, helps with longer training.")
    parser.add_argument("--single-task", type=str, default="",
                        help="If set, train only the named head (all other heads dropped). "
                             "Useful for ablations: does multi-task interference hurt this trait?")
    parser.add_argument("--balanced-families", action="store_true",
                        help="Sample training genomes so every taxonomic family is drawn with "
                             "equal expected frequency (WeightedRandomSampler). Counteracts the "
                             "long-tailed family distribution behind family-test collapse (Table 15). "
                             "Affects only the train split.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    schema = json.loads(SCHEMA_PATH.read_text())
    vocab = json.loads(VOCAB_PATH.read_text())
    traits_df = pd.read_parquet(TRAITS_PATH)
    splits_df = pd.read_parquet(SPLITS_PATH)
    df = traits_df.merge(splits_df[["bacdive_id", f"{args.split_level}_split"]], on="bacdive_id", how="left")
    df = df.rename(columns={f"{args.split_level}_split": "split"})
    print(f"Loaded {len(df):,} strains  (split={args.split_level})")
    print(f"  train: {(df.split == 'train').sum():,}   val: {(df.split == 'val').sum():,}   test: {(df.split == 'test').sum():,}")

    per_protein = args.per_protein is not None
    features = None       # pre-pooled [N, D] tensor (2-D path); None when per-protein
    feat_paths = None      # list[Path] aligned to df rows (per-protein path)
    if per_protein:
        manifest = pd.read_parquet(args.per_protein / "manifest.parquet")
        ok = manifest[manifest.status == "ok"]
        id_to_path = {int(b): args.per_protein / p for b, p in zip(ok.bacdive_id, ok.path)}
        keep_mask = df.bacdive_id.map(id_to_path.__contains__).fillna(False).values
        df = df[keep_mask].reset_index(drop=True)
        feat_paths = df.bacdive_id.map(id_to_path).tolist()
        input_dim = int(np.load(feat_paths[0]).shape[1])
        print(f"  per-protein: {len(feat_paths):,} genomes, embed_dim={input_dim} from {args.per_protein}")
    elif args.features:
        npz = np.load(args.features)
        feat_ids = npz["bacdive_ids"]
        feat_mat = npz["features"]
        id_to_row = {int(b): i for i, b in enumerate(feat_ids)}
        keep_mask = df.bacdive_id.map(id_to_row.__contains__).fillna(False).values
        df = df[keep_mask].reset_index(drop=True)
        feat_rows = df.bacdive_id.map(id_to_row).values
        features = torch.tensor(feat_mat[feat_rows], dtype=torch.float32)
        input_dim = features.shape[1]
        print(f"  loaded features [{features.shape[0]}, {input_dim}] from {args.features}")
    else:
        features = torch.randn(len(df), args.feat_dim)
        input_dim = args.feat_dim
        print(f"  using RANDOM features [{features.shape[0]}, {input_dim}] (smoke test)")

    labels, masks, specs = prepare_labels(df, vocab, schema)
    if args.single_task:
        if args.single_task not in specs:
            sys.exit(f"--single-task '{args.single_task}' is not a known head. "
                     f"Available: {sorted(specs.keys())}")
        specs = {args.single_task: specs[args.single_task]}
        labels = {args.single_task: labels[args.single_task]}
        masks = {args.single_task: masks[args.single_task]}
        print(f"  single-task mode: only training head '{args.single_task}'")
    print(f"  built {len(specs)} heads:")
    for name, spec in specs.items():
        print(f"    {name:<24} {spec['head_type']:<18} size={spec['size']}")

    splits = {s: df.index[df.split == s].tolist() for s in ("train", "val", "test")}
    loaders = {}
    for split_name, idx in splits.items():
        if not idx:
            continue
        idx_t = torch.tensor(idx, dtype=torch.long)
        labels_sub = {k: v[idx_t] for k, v in labels.items()}
        masks_sub = {k: v[idx_t] for k, v in masks.items()}
        if per_protein:
            paths_sub = [feat_paths[i] for i in idx]
            dataset = PerProteinDataset(paths_sub, labels_sub, masks_sub,
                                        max_proteins=(args.max_proteins or None))
            collate_fn = collate_perprotein
        else:
            dataset = StrainDataset(features[idx_t], labels_sub, masks_sub)
            collate_fn = collate

        sampler = None
        shuffle = (split_name == "train")
        if split_name == "train" and args.balanced_families:
            fam = df.loc[idx, "family"].tolist()
            w = family_sample_weights(fam)
            sampler = WeightedRandomSampler(
                torch.as_tensor(w, dtype=torch.double), num_samples=len(idx), replacement=True,
            )
            shuffle = False  # sampler and shuffle are mutually exclusive
            print(f"  family-balanced sampling on train: {len(set(fam)):,} families over {len(idx):,} genomes")

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooling = args.pooling if per_protein else None
    model = MicrobeFoundationModel(input_dim, specs, hidden=args.hidden, dropout=args.dropout,
                                   pooling=pooling, topk=args.topk,
                                   st_heads=args.st_heads, st_inducing=args.st_inducing).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    pool_s = pooling or "pre-pooled"
    print(f"\nmodel: {n_params:,} parameters  (device={device}, pooling={pool_s})")

    class_weights = None
    if args.class_weights:
        train_idx = torch.tensor(splits["train"], dtype=torch.long)
        print(f"\nclass-weighted loss enabled — imbalance threshold={args.imbalance_threshold}")
        class_weights = compute_class_weights(
            labels, masks, specs, train_idx,
            threshold=args.imbalance_threshold, verbose=True,
        )
        print(f"  -> applied weights to {len(class_weights)} / {len(specs)} heads")

    print(f"\ntraining for {args.epochs} epochs (scheduler={args.scheduler})...")
    train(model, loaders["train"], loaders.get("val"), specs, device, args.epochs, args.lr,
          class_weights=class_weights, scheduler=args.scheduler)

    test_loader = loaders.get("test")
    if test_loader is None:
        print("\nno test split (sample too small) — skipping test eval")
        test_metrics: dict[str, dict[str, float]] = {}
    else:
        print(f"\nfinal test metrics:")
        test_metrics = run_eval(model, test_loader, specs, device)
        for name, metric_dict in sorted(test_metrics.items()):
            kind = specs[name]["head_type"]
            parts = " ".join(f"{k}={v:.4f}" for k, v in metric_dict.items())
            print(f"  {name:<24} {kind:<18} {parts}")

    if args.save_metrics:
        out = {
            "run_name": args.run_name or (args.features.name if args.features else "smoke"),
            "split_level": args.split_level,
            "seed": args.seed,
            "balanced_families": bool(args.balanced_families),
            "epochs": args.epochs,
            "n_params": n_params,
            "feature_dim": input_dim,
            "pooling": pooling or "pre-pooled",
            "n_train": int((df.split == "train").sum()),
            "n_val": int((df.split == "val").sum()),
            "n_test": int((df.split == "test").sum()),
            "per_head": {
                name: {
                    "metric_kind": _primary_metric(specs[name]["head_type"]),
                    "score": float(metric_dict[_primary_metric(specs[name]["head_type"])]),
                    "metrics": {k: float(v) for k, v in metric_dict.items()},
                    "head_type": specs[name]["head_type"],
                    "head_size": specs[name]["size"],
                }
                for name, metric_dict in test_metrics.items()
            },
        }
        args.save_metrics.parent.mkdir(exist_ok=True)
        args.save_metrics.write_text(json.dumps(out, indent=2))
        print(f"\nwrote test metrics to {args.save_metrics}")

    if args.save_predictions and test_loader is not None:
        binary_heads = [n for n in specs if specs[n]["head_type"] == "binary"]
        if len(binary_heads) != 1:
            print(f"--save-predictions needs exactly one binary head, found {len(binary_heads)}; "
                  "use --single-task <trait>. Skipping.")
        else:
            head = binary_heads[0]
            probs, trues, mask = collect_binary_predictions(model, test_loader, specs, device, head)
            # test_loader is unshuffled, so order matches splits["test"].
            test_ids = df.loc[splits["test"], "bacdive_id"].to_numpy()
            frame = predictions_to_frame(test_ids, probs, trues, mask)
            args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.save_predictions)
            print(f"wrote {len(frame)} per-genome predictions ({head}) to {args.save_predictions}")

    if args.save_model:
        # Everything needed to rebuild the model for attention extraction:
        # weights + the constructor args (head sizes, input_dim, pool flag).
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden": args.hidden,
            "attention_pool": per_protein,
            "pooling": pooling or "pre-pooled",
            "topk": args.topk,
            "st_heads": args.st_heads,
            "st_inducing": args.st_inducing,
            "head_sizes": {name: int(head.out_features) for name, head in model.heads.items()},
            "split_level": args.split_level,
            "seed": args.seed,
        }, args.save_model)
        print(f"wrote model checkpoint to {args.save_model}")


if __name__ == "__main__":
    main()
