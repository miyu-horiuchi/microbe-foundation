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
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import model as M  # noqa: E402

ESM2_MAX_LEN = 1024
AA = "ACDEFGHIKLMNPQRSTVWYBXZJUO"  # amino-acid alphabet for the mock tokenizer


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


# --------------------------------------------------------------------------- #
# Data wiring (mirrors model.py main, but resolves raw-sequence paths)
# --------------------------------------------------------------------------- #
def build_loaders(args, seed):
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
    loaders = {}
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
        if s == "train" and args.balanced_families:
            fam = df.loc[idx, "family"].tolist()
            w = M.family_sample_weights(fam)
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                            num_samples=len(idx), replacement=True)
            loaders[s] = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                                    collate_fn=collate_seqs, num_workers=args.num_workers)
        else:
            loaders[s] = DataLoader(ds, batch_size=args.batch, shuffle=(s == "train"),
                                    collate_fn=collate_seqs, num_workers=args.num_workers)
    return loaders, specs, df, idx_by, labels, masks


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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, specs, df, idx_by, labels, masks = build_loaders(args, args.seed)
    print(f"[data] split={args.split_level} "
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
    print(f"[model] mode={args.mode} pooling={args.pooling} "
          f"trainable={n_train_params:,} / {n_total:,} params")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        net.train(True)
        t0 = time.time()
        tot, nb = 0.0, 0
        for genomes, lbls, msks in loaders["train"]:
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = net(genomes, device)
                loss, _ = M.masked_loss(preds,
                                        {k: v.to(device) for k, v in lbls.items()},
                                        {k: v.to(device) for k, v in msks.items()},
                                        specs, weights=class_weights)
            if loss.grad_fn is None:
                continue
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
        msg = f"[epoch {epoch+1}/{args.epochs}] train_loss={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)"
        if "val" in loaders:
            vm = M.run_eval(net, loaders["val"], specs, device, forward_fn=model_forward)
            msg += f" val_avg={M._avg_primary(vm, specs):.4f}"
        print(msg, flush=True)

    # ---- final family-test eval ----
    metrics = {}
    if "test" in loaders:
        metrics = M.run_eval(net, loaders["test"], specs, device, forward_fn=model_forward)
        print(f"[test] avg_primary={M._avg_primary(metrics, specs):.4f}")
        for name, mm in metrics.items():
            print(f"    {name:24s} {mm}")

    if args.save_metrics:
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


if __name__ == "__main__":
    main()
