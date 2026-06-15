#!/usr/bin/env python3
"""lora_rank_profile.py -- measure the intrinsic adaptation rank per layer.

Turns LoRA rank from a blind hyperparameter into a *measured* quantity, using
the spectral-structure argument (9D Labs: Kronecker spectral identity + backward
bottleneck). Two complementary estimates, both per trainable Linear layer:

  STEP 1  (cheap, one forward/backward -- the "a-priori Kronecker estimate")
  --------------------------------------------------------------------------
  For a linear layer y = W x, the minibatch gradient is
        G = sum_n  delta_n  x_n^T   =   Delta^T A        (shape [out, in])
  where A = stacked layer inputs [N, in] and Delta = stacked grad-outputs
  (the back-propagated error signal) [N, out]. The K-FAC / Kronecker identity
  factorises the gradient's second moment as
        E[vec(G) vec(G)^T]  ~  Cov[error]  (x)  Cov[activation]
  so the gradient's usable rank is bounded by the *smaller* of the activation
  and error covariance ranks. We therefore report:
        reff(A^T A)   -- activation covariance effective rank
        reff(D^T D)   -- error      covariance effective rank
        kron_estimate = min(the two)            <- predicted before any training
        reff(G)       -- effective rank of the actual minibatch gradient
  reff(G) is what kron_estimate predicts; agreement validates the cheap proxy.

  STEP 2  (ground truth -- a short full fine-tune)
  ------------------------------------------------
  Snapshot W0, run a few hundred optimiser steps, then per layer
        dW = W - W0,   reff(dW)
  This is the empirical "how many directions did adaptation actually use".

  PAYOFF
  ------
  suggested_lora_rank = ceil(reff)  -- set each layer's LoRA adapter to the
  measured rank instead of grid-searching one global value. (The PAC-Bayes side
  argues for rounding *down* toward the smallest rank that still fits, so treat
  reff(dW) as an upper target.)

Effective rank (Roy & Vetterli): for a spectrum of energies e_k (= sigma_k^2),
    p_k = e_k / sum_j e_j ,   reff = exp(-sum_k p_k log p_k).

NOTE ON THIS REPO: the pipeline trains a pooler + encoder MLP + heads on top of
*frozen, precomputed* ESM-2 embeddings; the ESM-2 transformer is not loaded
here. This script therefore profiles the layers we actually train (default:
the encoder + heads). The procedure is model-agnostic -- point --include at
ESM-2's Linear layers once the encoder is unfrozen and it works unchanged.

Usage
-----
    # local dry run (random features, CPU, no embeddings needed):
    python3 paper/lora_rank_profile.py --pooling none --batch 32 --ft-steps 50

    # real run on a GPU box with the per-protein embeddings:
    python3 paper/lora_rank_profile.py --per-protein data/esm2_perprotein \
        --split-level species --pooling set_transformer --include 'encoder|pool' \
        --batch 64 --ft-steps 300
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import the project model + data pipeline (model.py at repo root).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import model as M  # noqa: E402


def eff_rank(energies: torch.Tensor) -> float:
    """Effective rank from non-negative spectral energies (sigma^2 or eigenvalues)."""
    e = energies[energies > 0].double()
    if e.numel() == 0:
        return 0.0
    p = e / e.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def gram_eff_rank(gram: torch.Tensor) -> float:
    """reff of a symmetric PSD gram matrix (eigenvalues are sigma^2 already)."""
    ev = torch.linalg.eigvalsh(gram.double()).clamp(min=0)
    return eff_rank(ev)


def build_train_loader(args):
    """Mirror model.main()'s data wiring for the train split only."""
    schema = M.json.loads(M.SCHEMA_PATH.read_text())
    vocab = M.json.loads(M.VOCAB_PATH.read_text())
    traits_df = M.pd.read_parquet(M.TRAITS_PATH)
    splits_df = M.pd.read_parquet(M.SPLITS_PATH)
    df = traits_df.merge(splits_df[["bacdive_id", f"{args.split_level}_split"]],
                         on="bacdive_id", how="left")
    df = df.rename(columns={f"{args.split_level}_split": "split"})

    per_protein = args.per_protein is not None
    features = None
    feat_paths = None
    if per_protein:
        manifest = M.pd.read_parquet(args.per_protein / "manifest.parquet")
        ok = manifest[manifest.status == "ok"]
        id_to_path = {int(b): args.per_protein / p for b, p in zip(ok.bacdive_id, ok.path)}
        keep = df.bacdive_id.map(id_to_path.__contains__).fillna(False).values
        df = df[keep].reset_index(drop=True)
        feat_paths = df.bacdive_id.map(id_to_path).tolist()
        input_dim = int(np.load(feat_paths[0]).shape[1])
    else:
        # random pre-pooled features -- lets the script dry-run with no embeddings.
        features = torch.randn(len(df), args.feat_dim)
        input_dim = args.feat_dim
        print(f"[dry-run] RANDOM features [{len(df)}, {input_dim}] (no embeddings)")

    labels, masks, specs = M.prepare_labels(df, vocab, schema)
    train_idx = df.index[df.split == "train"].tolist()
    if not train_idx:
        raise SystemExit(f"no train rows for split-level={args.split_level}")
    idx_t = torch.tensor(train_idx, dtype=torch.long)
    labels_sub = {k: v[idx_t] for k, v in labels.items()}
    masks_sub = {k: v[idx_t] for k, v in masks.items()}

    if per_protein:
        ds = M.PerProteinDataset([feat_paths[i] for i in train_idx], labels_sub,
                                 masks_sub, max_proteins=(args.max_proteins or None))
        collate = M.collate_perprotein
    else:
        ds = M.StrainDataset(features[idx_t], labels_sub, masks_sub)
        collate = M.collate

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=(args.num_workers > 0))
    pooling = None if args.pooling == "none" else args.pooling
    return loader, specs, input_dim, pooling


def target_linears(model, include: str):
    pat = re.compile(include)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and pat.search(name):
            out[name] = mod
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-protein", type=Path, default=None)
    ap.add_argument("--split-level", choices=["species", "genus", "family"],
                    default="species")
    ap.add_argument("--pooling",
                    choices=["none", "mean", "attention", "set_transformer"],
                    default="none")
    ap.add_argument("--include", default="encoder|heads",
                    help="regex over layer names to profile (e.g. 'encoder|pool').")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cov-batches", type=int, default=1,
                    help="batches to accumulate covariances/gradient over (Step 1).")
    ap.add_argument("--ft-steps", type=int, default=300,
                    help="optimiser steps for the short full fine-tune (Step 2).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--st-heads", type=int, default=4)
    ap.add_argument("--st-inducing", type=int, default=16)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--feat-dim", type=int, default=640)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=REPO / "paper")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, specs, input_dim, pooling = build_train_loader(args)
    model = M.MicrobeFoundationModel(input_dim, specs, hidden=args.hidden,
                                     dropout=args.dropout, pooling=pooling,
                                     topk=args.topk, st_heads=args.st_heads,
                                     st_inducing=args.st_inducing).to(device)
    layers = target_linears(model, args.include)
    if not layers:
        raise SystemExit(f"no Linear layers matched --include '{args.include}'")
    print(f"device={device}  pooling={pooling}  profiling {len(layers)} layers: "
          f"{list(layers)}")

    # ---- Step 1: covariances + gradient over cov-batches (no optimiser step) ----
    # Accumulators per layer.
    A_gram = {n: 0.0 for n in layers}   # [in, in]
    D_gram = {n: 0.0 for n in layers}   # [out, out]
    G_acc = {n: 0.0 for n in layers}    # [out, in] summed gradient
    cur_input: dict[str, torch.Tensor] = {}

    fwd_handles, bwd_handles = [], []
    for name, mod in layers.items():
        def fwd_hook(m, inp, out, _n=name):
            cur_input[_n] = inp[0].detach().reshape(-1, inp[0].shape[-1])
        def bwd_hook(m, gin, gout, _n=name):
            D = gout[0].detach().reshape(-1, gout[0].shape[-1])  # [N, out]
            A = cur_input[_n]                                     # [N, in]
            A_gram[_n] = A_gram[_n] + (A.T @ A)
            D_gram[_n] = D_gram[_n] + (D.T @ D)
            G_acc[_n] = G_acc[_n] + (D.T @ A)                    # [out, in]
        fwd_handles.append(mod.register_forward_hook(fwd_hook))
        bwd_handles.append(mod.register_full_backward_hook(bwd_hook))

    model.train(True)
    seen = 0
    it = iter(loader)
    for _ in range(args.cov_batches):
        try:
            feats, lbls, msks = next(it)
        except StopIteration:
            break
        model.zero_grad(set_to_none=True)
        preds = M._model_forward(model, feats, device)
        loss, _ = M.masked_loss(preds, {k: v.to(device) for k, v in lbls.items()},
                                {k: v.to(device) for k, v in msks.items()}, specs)
        if loss.grad_fn is None:
            continue
        loss.backward()
        seen += 1
    for h in fwd_handles + bwd_handles:
        h.remove()
    if seen == 0:
        raise SystemExit("no usable batch (all-empty labels); try a larger --batch")

    step1 = {}
    for name in layers:
        step1[name] = {
            "act_reff": gram_eff_rank(A_gram[name]),
            "err_reff": gram_eff_rank(D_gram[name]),
            "grad_reff": eff_rank(torch.linalg.svdvals(G_acc[name].double()) ** 2),
        }
        step1[name]["kron_estimate"] = min(step1[name]["act_reff"],
                                           step1[name]["err_reff"])

    # ---- Step 2: short full fine-tune, then dW effective rank ----
    W0 = {n: layers[n].weight.detach().clone() for n in layers}
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train(True)
    done = 0
    while done < args.ft_steps:
        for feats, lbls, msks in loader:
            preds = M._model_forward(model, feats, device)
            loss, _ = M.masked_loss(preds, {k: v.to(device) for k, v in lbls.items()},
                                    {k: v.to(device) for k, v in msks.items()}, specs)
            if loss.grad_fn is None:
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            done += 1
            if done >= args.ft_steps:
                break
    dW_reff = {}
    for name in layers:
        dW = (layers[name].weight.detach() - W0[name])
        dW_reff[name] = eff_rank(torch.linalg.svdvals(dW.double()) ** 2)

    # ---- assemble table ----
    rows = []
    for name in layers:
        w = layers[name].weight
        min_dim = min(w.shape)
        s = step1[name]
        rows.append({
            "layer": name, "out": w.shape[0], "in": w.shape[1], "min_dim": min_dim,
            "act_reff": round(s["act_reff"], 2),
            "err_reff": round(s["err_reff"], 2),
            "kron_estimate": round(s["kron_estimate"], 2),
            "grad_reff": round(s["grad_reff"], 2),
            "deltaW_reff": round(dW_reff[name], 2),
            "suggested_lora_rank": int(math.ceil(dW_reff[name])),
            "rank_frac_deltaW": round(dW_reff[name] / min_dim, 3),
        })

    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    csv_fp = tables / "28_lora_rank_profile.csv"
    with csv_fp.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    md = ["# Per-layer adaptation-rank profile",
          "",
          f"split={args.split_level}  pooling={pooling}  batch={args.batch}  "
          f"cov_batches={seen}  ft_steps={args.ft_steps}",
          "",
          "`kron_estimate` (cheap, one backward) predicts `grad_reff`; "
          "`deltaW_reff` is the empirical adaptation rank after fine-tuning; "
          "`suggested_lora_rank = ceil(deltaW_reff)`.",
          "",
          "| layer | shape | act_reff | err_reff | kron_est | grad_reff | dW_reff | LoRA r | dW frac |",
          "|-------|-------|---------:|---------:|---------:|----------:|--------:|-------:|--------:|"]
    for r in rows:
        md.append(f"| {r['layer']} | {r['out']}x{r['in']} | {r['act_reff']} | "
                  f"{r['err_reff']} | {r['kron_estimate']} | {r['grad_reff']} | "
                  f"{r['deltaW_reff']} | {r['suggested_lora_rank']} | "
                  f"{r['rank_frac_deltaW']} |")
    (tables / "28_lora_rank_profile.md").write_text("\n".join(md) + "\n")

    # ---- figure: predicted (kron) vs empirical (dW) per layer ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["layer"] for r in rows]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4.6))
        ax.bar(x - 0.2, [r["kron_estimate"] for r in rows], 0.2, label="kron estimate (Step 1, cheap)", color="#2980b9")
        ax.bar(x, [r["grad_reff"] for r in rows], 0.2, label="grad reff (Step 1)", color="#7fb3d5")
        ax.bar(x + 0.2, [r["deltaW_reff"] for r in rows], 0.2, label="dW reff (Step 2, fine-tune)", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("effective rank")
        ax.set_title("Predicted vs empirical adaptation rank per layer")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        figs = args.out_dir / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(figs / "28_lora_rank_profile.png", dpi=200, bbox_inches="tight")
        fig.savefig(figs / "28_lora_rank_profile.pdf", bbox_inches="tight")
        print(f"wrote {figs / '28_lora_rank_profile.png'} (+ .pdf)")
    except Exception as e:  # pragma: no cover
        print(f"[warn] figure skipped: {e}")

    print(f"wrote {csv_fp}")
    print(f"wrote {tables / '28_lora_rank_profile.md'}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
