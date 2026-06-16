#!/usr/bin/env python3
"""finetune_lora.py -- the decisive lever test: adapt the ESM-2 encoder.

Every analysis so far ran on FROZEN ESM-2 embeddings and showed the cross-clade
(family-split) collapse is an out-of-distribution coverage limit, with the pooling
operator, retrieval, and self-training unable to fix it. The one untested genuine
lever is the encoder itself. This script unfreezes ESM-2 with LoRA adapters at the
rank the profiler measured (lora_rank_profile.py: encoder adaptation is low-rank,
r ~ 8-16) and asks the falsifiable question:

    Does adapting the representation close the family-split gap that a frozen
    encoder cannot?

It runs three modes on identical data/splits/seeds so the comparison is clean:
    frozen   ESM-2 weights frozen (reproduces the cached-embedding baseline,
             but computed live so the only difference vs `lora` is the adapters)
    lora     ESM-2 frozen + LoRA adapters on attention projections (trainable:
             adapters + pooler + encoder MLP + heads)
    full     all ESM-2 weights trainable (expensive upper bound)

Architecture: ESM-2 (HF EsmModel, optionally LoRA) -> residue-mean-pool per
protein -> [B, P, 640] -> the EXISTING model.py stack (make_pooler -> encoder MLP
-> per-trait heads). Loss / labels / eval / metrics / family sampler are reused
verbatim from model.py, so output JSONs drop straight into aggregate_seeds.py /
leaderboard.py.

GPU-ONLY for real runs (backprop through 150M/650M ESM-2). A CPU plumbing smoke
is available via `--model-name mock`, which swaps in a tiny random encoder and
needs no GPU, no network, and no transformers/peft install:

    python3 finetune_lora.py --model-name mock --mode lora --smoke \
        --proteins-dir data/_mock_proteins --max-genomes 64 --epochs 1

Real run (GPU box, after syncing raw protein sequences + `pip install peft`):
    python3 finetune_lora.py --mode lora --split-level family \
        --proteins-dir data/esm2_proteins --pooling set_transformer \
        --model-name facebook/esm2_t30_150M_UR50D --lora-r 16 --lora-alpha 32 \
        --max-proteins 256 --batch 4 --epochs 15 --grad-checkpoint \
        --save-metrics runs/lora/lora_family_s0.json --seed 0
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import model as M  # noqa: E402

ESM2_MAX_LEN = 1024
AA = "ACDEFGHIKLMNPQRSTVWYBXZJUO"  # amino-acid alphabet for the mock tokenizer


# --------------------------------------------------------------------------- #
# Distributed (DDP) plumbing
#
# torchrun launches one process per GPU and exports RANK / WORLD_SIZE /
# LOCAL_RANK. When WORLD_SIZE is unset or 1 we stay in the original
# single-process path so existing single-GPU behaviour is byte-for-byte
# unchanged. On GPU we use the nccl backend; for the CPU gloo smoke test (no
# CUDA, no real network) we fall back to gloo so the exact same code path can be
# exercised on a laptop / CI box.
# --------------------------------------------------------------------------- #
@dataclass
class Dist:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> Dist:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return Dist(enabled=False, rank=0, world_size=1, local_rank=0)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return Dist(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def dist_print(dd: Dist, *args, **kwargs):
    """Print only from rank 0 so a WORLD_SIZE-way run does not N-plicate logs."""
    if dd.is_main:
        print(*args, **kwargs)


class DistributedEvalSampler(Sampler):
    """Disjoint, NON-padded shard of indices for val/test under DDP.

    torch's DistributedSampler pads the last shard with duplicate samples so
    every rank sees an equal count -- which would DOUBLE-COUNT genomes in a
    metric. Eval correctness needs each genome exactly once, so we simply stride
    the index list (indices[rank::world_size]). Shards are uneven, which is fine
    because the evaluator all-gathers raw (pred, y, mask) buffers and computes
    metrics over the exact union == the full split.
    """

    def __init__(self, n: int, world_size: int, rank: int):
        self.indices = list(range(rank, n, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DistributedWeightedSampler(Sampler):
    """Family-balanced sampler that shards the per-epoch work across ranks.

    The single-GPU run draws `num_samples` genomes with replacement from the
    family-balancing weights (model.family_sample_weights). Under DDP each rank
    independently draws num_samples/world_size genomes from the SAME global
    weights, seeded by (seed, epoch, rank) so the ranks cover different genomes.
    Total draws across ranks == num_samples, every draw is family-balanced in
    expectation, and set_epoch reshuffles each epoch -- preserving the science
    of the balanced sampler while giving each rank a disjoint(-in-expectation)
    chunk of work.
    """

    def __init__(self, weights, num_samples_total: int, world_size: int,
                 rank: int, seed: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = max(1, num_samples_total // world_size)
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + 1000 * self.epoch + self.rank)
        idx = torch.multinomial(self.weights, self.num_samples,
                                replacement=True, generator=g)
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


# --------------------------------------------------------------------------- #
# Data: raw protein sequences per genome
# --------------------------------------------------------------------------- #
def read_proteins(fp: Path) -> list[str]:
    """One genome's protein AA sequences from a gzipped (or plain) text file."""
    opener = gzip.open if fp.suffix == ".gz" else open
    with opener(fp, "rt") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


class ProteinSeqDataset(Dataset):
    """Yields (list[str] proteins, label_dict, mask_dict) per genome.

    Tokenization is deferred to the encoder forward so ragged protein counts and
    lengths are handled in one flattened batch.
    """

    def __init__(self, paths, labels, masks, max_proteins: int, seed: int = 0):
        self.paths = paths
        self.labels = labels
        self.masks = masks
        self.max_proteins = max_proteins
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        seqs = read_proteins(Path(self.paths[idx]))
        if not seqs:
            seqs = ["M"]  # guard: never hand the encoder an empty genome
        if self.max_proteins and len(seqs) > self.max_proteins:
            sel = self.rng.choice(len(seqs), self.max_proteins, replace=False)
            seqs = [seqs[i] for i in sel]
        lab = {k: v[idx] for k, v in self.labels.items()}
        msk = {k: v[idx] for k, v in self.masks.items()}
        return seqs, lab, msk


def collate_seqs(batch):
    """Collate to (list_of_genomes, label_dict, mask_dict). No tensorization of
    sequences here -- the encoder flattens + tokenizes."""
    genomes = [b[0] for b in batch]
    keys = batch[0][1].keys()
    labels = {k: torch.stack([b[1][k] for b in batch]) for k in keys}
    masks = {k: torch.stack([b[2][k] for b in batch]) for k in keys}
    return genomes, labels, masks


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class MockEncoder(nn.Module):
    """Tiny CPU/network-free stand-in for ESM-2 to validate the train/eval loop.

    Deterministic char-embedding -> mean over residues -> Linear to out_dim. Has
    real trainable parameters so the optimiser step and grad flow are exercised.
    """

    def __init__(self, out_dim: int = 640):
        super().__init__()
        self.out_dim = out_dim
        self.vocab = {c: i + 1 for i, c in enumerate(AA)}  # 0 = pad
        self.emb = nn.Embedding(len(self.vocab) + 1, 64, padding_idx=0)
        self.proj = nn.Linear(64, out_dim)

    def encode_proteins(self, seqs: list[str], device) -> torch.Tensor:
        maxlen = min(max(len(s) for s in seqs), ESM2_MAX_LEN)
        ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for i, s in enumerate(seqs):
            for j, c in enumerate(s[:maxlen]):
                ids[i, j] = self.vocab.get(c, 0)
        ids = ids.to(device)
        mask = (ids > 0).float().unsqueeze(-1)
        h = self.emb(ids)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(pooled)


class Esm2Encoder(nn.Module):
    """HF ESM-2 (optionally LoRA-adapted) -> per-protein residue-mean-pool."""

    def __init__(self, model_name: str, mode: str, lora_r: int, lora_alpha: int,
                 lora_dropout: float, target_modules, grad_checkpoint: bool):
        super().__init__()
        from transformers import AutoTokenizer

        # Import EsmModel EXPLICITLY (not via AutoModel). AutoModel resolves the
        # class through a lazy mapping whose `hasattr` swallows the real import
        # error and reports the useless "Could not find EsmModel neither in ..."
        # ValueError. That masked a polluted environment on the GPU box (system
        # dist-packages + pip --user collision breaking modeling_esm). The direct
        # import surfaces the true traceback, and we wrap it with a clear hint.
        try:
            from transformers import EsmModel
        except Exception as e:  # pragma: no cover - environment guard
            raise RuntimeError(
                "transformers is installed but EsmModel failed to import. This is "
                "almost always a polluted Python environment (system packages "
                "shadowing/colliding with pip --user installs). Run inside a clean "
                "venv so transformers/peft are isolated. Original error: "
                f"{e!r}") from e

        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base = EsmModel.from_pretrained(model_name)
        self.out_dim = base.config.hidden_size
        # Gradient checkpointing trades compute for memory by recomputing the
        # forward during backward. In `frozen` mode no gradients flow through the
        # encoder, so checkpointing is pure waste (it forces a useless second
        # forward pass -- the source of the "None of the inputs have
        # requires_grad" warning). Only enable it when the encoder is trainable.
        if grad_checkpoint and mode != "frozen":
            # use_reentrant=False is REQUIRED for DDP: reentrant activation
            # checkpointing marks each param ready twice under DistributedDataParallel
            # ("marked ready only once" error) unless static_graph is used. The
            # non-reentrant implementation is numerically identical and the path
            # PyTorch now recommends; it also silences the single-GPU reentrant
            # "inputs have requires_grad" warning. Fall back gracefully on a
            # transformers old enough to lack the kwarg.
            try:
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base.gradient_checkpointing_enable()

        if mode == "frozen":
            for p in base.parameters():
                p.requires_grad_(False)
            self.encoder = base
        elif mode == "full":
            self.encoder = base
        elif mode == "lora":
            from peft import LoraConfig, get_peft_model

            for p in base.parameters():
                p.requires_grad_(False)
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                             lora_dropout=lora_dropout,
                             target_modules=list(target_modules), bias="none")
            self.encoder = get_peft_model(base, cfg)
            self.encoder.print_trainable_parameters()
        else:
            raise ValueError(f"unknown mode {mode}")

    def encode_proteins(self, seqs: list[str], device) -> torch.Tensor:
        enc = self.tokenizer(seqs, return_tensors="pt", padding=True,
                             truncation=True, max_length=ESM2_MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        # Frozen encoder: its weights never change, so the per-protein embeddings
        # carry no gradient. Encode under no_grad to skip building (and, with
        # checkpointing, recomputing) the encoder graph. This is numerically
        # identical to the autograd path -- the trainable pooler/heads downstream
        # still get full gradients from the encoder OUTPUT -- but ~2x faster and
        # far lighter on memory. LoRA/full keep the normal autograd path.
        ctx = torch.no_grad() if self.mode == "frozen" else contextlib.nullcontext()
        with ctx:
            out = self.encoder(**enc).last_hidden_state      # [n, L, D]
            am = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            return (out * am).sum(1) / am.sum(1).clamp(min=1)  # [n, D]


# --------------------------------------------------------------------------- #
# Full model: encoder -> [B,P,D] -> model.py pool/encoder/heads
# --------------------------------------------------------------------------- #
class EsmTraitModel(nn.Module):
    def __init__(self, encoder, specs, pooling, hidden, dropout, st_heads,
                 st_inducing, topk, enc_microbatch: int):
        super().__init__()
        self.encoder = encoder
        self.enc_microbatch = enc_microbatch
        self.head = M.MicrobeFoundationModel(
            encoder.out_dim, specs, hidden=hidden, dropout=dropout,
            pooling=pooling, topk=topk, st_heads=st_heads, st_inducing=st_inducing)

    def forward(self, genomes, device=None):
        device = device or next(self.parameters()).device
        counts = [len(g) for g in genomes]
        flat = [s for g in genomes for s in g]
        # Encode in micro-batches to bound activation memory.
        vecs = []
        mb = self.enc_microbatch or len(flat)
        for i in range(0, len(flat), mb):
            vecs.append(self.encoder.encode_proteins(flat[i:i + mb], device))
        prot = torch.cat(vecs, dim=0)                         # [sum(counts), D]

        B, P, D = len(genomes), max(counts), prot.shape[1]
        x = torch.zeros(B, P, D, device=device, dtype=prot.dtype)
        pmask = torch.zeros(B, P, device=device)
        off = 0
        for b, c in enumerate(counts):
            x[b, :c] = prot[off:off + c]
            pmask[b, :c] = 1.0
            off += c
        return self.head(x, pmask)


def model_forward(model, feats, device):
    """forward_fn for M.run_eval: feats is the list of genomes from collate."""
    return model(feats, device)


def evaluate(model, loader, specs, device, dd: "Dist | None" = None):
    """Metric computation that is correct under DDP.

    Single-process: identical to M.run_eval. Distributed: each rank runs its
    disjoint (non-padded) shard, then we all-gather the raw per-head
    (pred, y, mask) buffers and compute metrics over the UNION via the shared
    M.metrics_from_buffers -- so the reported numbers are the true full-split
    metrics, not rank-0's shard, and match the single-GPU result exactly (modulo
    row order, which none of the metrics depend on).
    """
    if loader is None:
        return {}
    model.train(False)
    buf: dict[str, dict[str, list]] = {}
    with torch.no_grad():
        for feats, labels, masks in loader:
            preds = model_forward(model, feats, device)
            for name, pred in preds.items():
                spec = specs[name]
                y = labels[name].to(device)
                m = masks[name].to(device)
                if m.sum() == 0:
                    continue
                b = buf.setdefault(name, {"pred": [], "y": [], "m": []})
                b["pred"].append(pred.detach().cpu())
                b["y"].append(y.detach().cpu())
                b["m"].append(m.detach().cpu())

    if dd and dd.enabled:
        # Collapse each rank's lists to single tensors, all-gather the per-head
        # buffers (Python objects of CPU tensors), then merge so every rank holds
        # the full split and computes identical metrics.
        local = {n: {k: torch.cat(v, dim=0) for k, v in b.items()}
                 for n, b in buf.items()}
        gathered: list = [None] * dd.world_size
        dist.all_gather_object(gathered, local)
        merged: dict[str, dict[str, list]] = {}
        for part in gathered:
            for name, d in part.items():
                mb = merged.setdefault(name, {"pred": [], "y": [], "m": []})
                mb["pred"].append(d["pred"])
                mb["y"].append(d["y"])
                mb["m"].append(d["m"])
        buf = merged

    return M.metrics_from_buffers(buf, specs)


# --------------------------------------------------------------------------- #
# Data wiring (mirrors model.py main, but resolves raw-sequence paths)
# --------------------------------------------------------------------------- #
def build_loaders(args, seed, dd: "Dist | None" = None):
    schema = json.loads(M.SCHEMA_PATH.read_text())
    vocab = json.loads(M.VOCAB_PATH.read_text())
    traits_df = M.pd.read_parquet(M.TRAITS_PATH)
    splits_df = M.pd.read_parquet(M.SPLITS_PATH)
    df = traits_df.merge(splits_df[["bacdive_id", f"{args.split_level}_split"]],
                         on="bacdive_id", how="left")
    df = df.rename(columns={f"{args.split_level}_split": "split"})

    pdir = Path(args.proteins_dir)
    def path_for(bid):
        for cand in (pdir / f"{bid}.txt.gz", pdir / f"{bid}.txt",
                     pdir / "proteins" / f"{bid}.txt.gz"):
            if cand.exists():
                return cand
        return None
    df["ppath"] = df.bacdive_id.map(path_for)
    df = df[df.ppath.notna()].reset_index(drop=True)
    if args.max_genomes:
        # Seeded random subsample per split (NOT file order: avoids taxonomic
        # bias when the parquet is sorted by lineage, which would skew a family split).
        parts = [g.sample(n=min(len(g), args.max_genomes), random_state=seed)
                 for _, g in df.groupby("split")]
        df = M.pd.concat(parts).reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"no genomes with sequences under {pdir}")

    labels, masks, specs = M.prepare_labels(df, vocab, schema)
    ws = dd.world_size if (dd and dd.enabled) else 1
    rank = dd.rank if (dd and dd.enabled) else 0
    loaders = {}
    samplers = {}   # train sampler kept so the loop can call set_epoch under DDP
    idx_by = {}
    for s in ("train", "val", "test"):
        idx = df.index[df.split == s].tolist()
        idx_by[s] = idx
        if not idx:
            continue
        it = torch.tensor(idx, dtype=torch.long)
        ds = ProteinSeqDataset([df.ppath[i] for i in idx],
                               {k: v[it] for k, v in labels.items()},
                               {k: v[it] for k, v in masks.items()},
                               max_proteins=args.max_proteins, seed=seed)
        if s == "train":
            if args.balanced_families:
                fam = df.loc[idx, "family"].tolist()
                w = M.family_sample_weights(fam)
                if ws > 1:
                    sampler = DistributedWeightedSampler(
                        w, num_samples_total=len(idx), world_size=ws,
                        rank=rank, seed=seed)
                else:
                    sampler = WeightedRandomSampler(
                        torch.as_tensor(w, dtype=torch.double),
                        num_samples=len(idx), replacement=True)
            elif ws > 1:
                sampler = torch.utils.data.DistributedSampler(
                    ds, num_replicas=ws, rank=rank, shuffle=True, seed=seed)
            else:
                sampler = None
            samplers[s] = sampler
            loaders[s] = DataLoader(
                ds, batch_size=args.batch, sampler=sampler,
                shuffle=(sampler is None), collate_fn=collate_seqs,
                num_workers=args.num_workers)
        else:
            # val/test: stride-shard (no duplicate padding) under DDP, else plain.
            sampler = (DistributedEvalSampler(len(idx), ws, rank) if ws > 1 else None)
            loaders[s] = DataLoader(
                ds, batch_size=args.batch, sampler=sampler, shuffle=False,
                collate_fn=collate_seqs, num_workers=args.num_workers)
    return loaders, specs, df, idx_by, labels, masks, samplers


def build_encoder(args):
    if args.model_name == "mock":
        return MockEncoder(out_dim=args.feat_dim)
    return Esm2Encoder(args.model_name, args.mode, args.lora_r, args.lora_alpha,
                       args.lora_dropout, args.target_modules.split(","),
                       args.grad_checkpoint)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proteins-dir", required=True,
                    help="dir of <bacdive_id>.txt.gz raw protein sequences.")
    ap.add_argument("--model-name", default="facebook/esm2_t30_150M_UR50D",
                    help="HF ESM-2 name, or 'mock' for a CPU plumbing smoke.")
    ap.add_argument("--mode", choices=["frozen", "lora", "full"], default="lora")
    ap.add_argument("--split-level", choices=["species", "genus", "family"], default="family")
    ap.add_argument("--pooling",
                    choices=["mean", "max", "topk", "attention", "gated_attention", "set_transformer"],
                    default="set_transformer")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="query,key,value",
                    help="comma-separated ESM-2 Linear names for LoRA adapters.")
    ap.add_argument("--max-proteins", type=int, default=256)
    ap.add_argument("--enc-microbatch", type=int, default=8,
                    help="proteins encoded per ESM-2 forward (caps peak encoder "
                         "activation memory). Lowering is numerically identical "
                         "(same grads/results) -- it only changes how many proteins "
                         "share one forward. Default 8 keeps a full-backprop LoRA "
                         "run (150M + grad-checkpoint) inside an 80GB (or 40GB) GPU; "
                         "the old default of 128 OOM'd on an 80GB H100.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--st-heads", type=int, default=4)
    ap.add_argument("--st-inducing", type=int, default=16)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--feat-dim", type=int, default=640, help="mock encoder out dim.")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--balanced-families", action="store_true")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-genomes", type=int, default=0, help="cap genomes/split (smoke).")
    ap.add_argument("--smoke", action="store_true", help="tiny run for plumbing checks.")
    ap.add_argument("--save-metrics", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_genomes = args.max_genomes or 32
        args.max_proteins = min(args.max_proteins, 16)

    dd = setup_distributed()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dd.local_rank}" if dd.enabled else "cuda")
    else:
        device = torch.device("cpu")
    if dd.enabled:
        dist_print(dd, f"[ddp] world_size={dd.world_size} backend="
                       f"{'nccl' if torch.cuda.is_available() else 'gloo'} "
                       f"device={device}")

    loaders, specs, df, idx_by, labels, masks, samplers = build_loaders(
        args, args.seed, dd)
    dist_print(dd, f"[data] split={args.split_level} "
                   f"train={len(idx_by.get('train', []))} val={len(idx_by.get('val', []))} "
                   f"test={len(idx_by.get('test', []))} (genomes with sequences)")

    encoder = build_encoder(args)
    net = EsmTraitModel(encoder, specs, args.pooling, args.hidden, args.dropout,
                        args.st_heads, args.st_inducing, args.topk,
                        args.enc_microbatch).to(device)

    class_weights = None
    if args.class_weights and idx_by.get("train"):
        train_idx = torch.tensor(idx_by["train"], dtype=torch.long)
        class_weights = M.compute_class_weights(labels, masks, specs, train_idx)
    trainable = [p for p in net.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in net.parameters())
    dist_print(dd, f"[model] mode={args.mode} pooling={args.pooling} "
                   f"trainable={n_train_params:,} / {n_total:,} params")

    # Wrap in DDP so every rank's gradients are all-reduced (== large effective
    # batch sharded across GPUs). DDP broadcasts rank-0's weights at construction
    # so all replicas start identical regardless of per-rank RNG. The eval/forward
    # helpers use the UNWRAPPED module (train_model.module) to skip DDP's forward
    # hooks when no backward follows.
    train_model = net
    if dd.enabled:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[dd.local_rank], output_device=dd.local_rank)
        train_model = DistributedDataParallel(net, **ddp_kwargs)
    eval_model = train_model.module if dd.enabled else net

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        train_model.train(True)
        if dd.enabled and samplers.get("train") is not None \
                and hasattr(samplers["train"], "set_epoch"):
            samplers["train"].set_epoch(epoch)
        t0 = time.time()
        tot, nb = 0.0, 0
        for genomes, lbls, msks in loaders["train"]:
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = train_model(genomes, device)
                loss, _ = M.masked_loss(preds,
                                        {k: v.to(device) for k, v in lbls.items()},
                                        {k: v.to(device) for k, v in msks.items()},
                                        specs, weights=class_weights)
            real = loss.grad_fn is not None
            if dd.enabled:
                # Every rank MUST run a matching backward each step or the
                # all-reduce desyncs/hangs. An all-unlabeled batch yields a
                # detached zero loss (grad_fn is None) on that rank only; anchor
                # the loss to all trainable params (x0) so it always has a graph
                # AND every param gets a grad path -- letting us keep
                # find_unused_parameters=False (cheaper, no static-graph hazard).
                anchor = sum(p.sum() for p in trainable) * 0.0
                loss = loss + anchor
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                if real:
                    tot += float(loss.detach())
                    nb += 1
            else:
                if not real:
                    continue
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                tot += float(loss.detach())
                nb += 1
        # Globally-correct mean train loss: sum the per-rank loss totals and batch
        # counts, then divide (matches a single-process average over all batches).
        if dd.enabled:
            t = torch.tensor([tot, float(nb)], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tot, nb = t[0].item(), int(t[1].item())
        msg = (f"[epoch {epoch+1}/{args.epochs}] train_loss={tot/max(nb,1):.4f} "
               f"({time.time()-t0:.0f}s)")
        if "val" in loaders:
            vm = evaluate(eval_model, loaders["val"], specs, device, dd)
            msg += f" val_avg={M._avg_primary(vm, specs):.4f}"
        dist_print(dd, msg, flush=True)

    # ---- final family-test eval (metrics aggregated across all ranks) ----
    metrics = {}
    if "test" in loaders:
        metrics = evaluate(eval_model, loaders["test"], specs, device, dd)
        dist_print(dd, f"[test] avg_primary={M._avg_primary(metrics, specs):.4f}")
        for name, mm in metrics.items():
            dist_print(dd, f"    {name:24s} {mm}")

    # Only rank 0 writes the JSON / logs; every rank computed identical metrics.
    if args.save_metrics and dd.is_main:
        out = {
            "run_name": args.run_name or f"{args.mode}_{args.split_level}_s{args.seed}",
            "mode": args.mode, "model_name": args.model_name,
            "split_level": args.split_level, "seed": args.seed, "pooling": args.pooling,
            "lora_r": args.lora_r if args.mode == "lora" else None,
            "n_train": len(idx_by.get("train", [])), "n_test": len(idx_by.get("test", [])),
            "n_trainable_params": n_train_params, "n_total_params": n_total,
            "avg_primary": M._avg_primary(metrics, specs) if metrics else None,
            "per_head": {n: {"metric_kind": M._primary_metric(specs[n]["head_type"]),
                             "score": mm.get(M._primary_metric(specs[n]["head_type"])),
                             "metrics": mm, "head_type": specs[n]["head_type"],
                             "head_size": specs[n]["size"]}
                         for n, mm in metrics.items()},
        }
        fp = Path(args.save_metrics)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(out, indent=2))
        print(f"wrote {fp}")

    if dd.enabled:
        # Keep non-zero ranks alive until rank 0 has written, then tear down the
        # process group cleanly so torchrun exits 0 on every rank.
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
