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
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.exit("requires: pip install pandas pyarrow torch numpy")


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


class MicrobeFoundationModel(nn.Module):
    """Shared MLP encoder + per-trait linear heads."""

    def __init__(self, input_dim: int, head_specs: dict[str, dict], hidden: int = 512, dropout: float = 0.2):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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
) -> dict[str, torch.Tensor]:
    """Compute per-head loss-balancing weights from the training subset only.

    binary: pos_weight scalar (neg/pos), clamped to [1, 50] to avoid blow-up
            on heads with <2% positives.
    multiclass: per-class weight vector = total / (K * count_c).
    multilabel: per-label pos_weight vector (neg_per_label / pos_per_label).
    regression_vector: None (no weighting).
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
            pw = (neg / pos).clamp(min=1.0, max=50.0)
            out[name] = pw  # scalar
        elif h == "multiclass":
            valid = m > 0
            if valid.sum() == 0:
                continue
            k = spec["size"]
            counts = torch.zeros(k)
            for c in range(k):
                counts[c] = ((y == c) & valid).sum()
            total = counts.sum().clamp(min=1)
            # inverse frequency: total / (K * count). Empty classes get weight=0.
            w = torch.zeros(k)
            seen = counts > 0
            w[seen] = total / (seen.sum().float() * counts[seen])
            # Cap to avoid extreme weights from singleton classes
            w = w.clamp(max=50.0)
            out[name] = w
        elif h == "multilabel":
            if m.sum() == 0:
                continue
            pos_per_label = ((y == 1) & (m > 0)).sum(0).float()
            neg_per_label = ((y == 0) & (m > 0)).sum(0).float()
            pw = (neg_per_label / pos_per_label.clamp(min=1)).clamp(min=1.0, max=50.0)
            out[name] = pw
        # regression_vector: no weighting
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
            feats = feats.to(device)
            preds = model(feats)
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


def _avg_primary(metrics: dict[str, dict[str, float]], specs) -> float:
    vals = [m[_primary_metric(specs[name]["head_type"])] for name, m in metrics.items()]
    return sum(vals) / max(len(vals), 1)


def train(model, train_loader, val_loader, specs, device, epochs: int, lr: float,
          class_weights: dict | None = None):
    """val_loader may be None when the split has zero validation strains (small samples)."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    val_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train(True)
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for feats, labels, masks in train_loader:
            feats = feats.to(device)
            preds = model(feats)
            loss, _ = masked_loss(preds, labels, masks, specs, weights=class_weights)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        if val_loader is not None:
            val_metrics = run_eval(model, val_loader, specs, device)
            avg_val = _avg_primary(val_metrics, specs)
            val_s = f"val_avg={avg_val:.4f}"
        else:
            val_s = "val=skipped(empty)"
        elapsed = time.time() - t0
        print(f"  epoch {epoch:>3}  train_loss={train_loss:.4f}  {val_s}  ({elapsed:.1f}s)")
    return val_metrics


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=None, help="features.npz (bacdive_ids, features). Omit for random smoke test.")
    parser.add_argument("--split-level", choices=["species", "genus", "family"], default="family")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feat-dim", type=int, default=128, help="Smoke-test feature dim (ignored if --features given)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-metrics", type=Path, default=None, help="Write final test metrics to this JSON path")
    parser.add_argument("--run-name", type=str, default="", help="Tag for the saved metrics (e.g., 'esm2-35M-family')")
    parser.add_argument("--class-weights", action="store_true",
                        help="Balance loss by inverse class frequency on training labels. "
                             "Improves macro F1 on imbalanced heads at the cost of plain accuracy.")
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

    if args.features:
        npz = np.load(args.features)
        feat_ids = npz["bacdive_ids"]
        feat_mat = npz["features"]
        id_to_row = {int(b): i for i, b in enumerate(feat_ids)}
        keep_mask = df.bacdive_id.map(id_to_row.__contains__).fillna(False).values
        df = df[keep_mask].reset_index(drop=True)
        feat_rows = df.bacdive_id.map(id_to_row).values
        features = torch.tensor(feat_mat[feat_rows], dtype=torch.float32)
        print(f"  loaded features [{features.shape[0]}, {features.shape[1]}] from {args.features}")
    else:
        features = torch.randn(len(df), args.feat_dim)
        print(f"  using RANDOM features [{features.shape[0]}, {features.shape[1]}] (smoke test)")

    labels, masks, specs = prepare_labels(df, vocab, schema)
    print(f"  built {len(specs)} heads:")
    for name, spec in specs.items():
        print(f"    {name:<24} {spec['head_type']:<18} size={spec['size']}")

    splits = {s: df.index[df.split == s].tolist() for s in ("train", "val", "test")}
    loaders = {}
    for split_name, idx in splits.items():
        if not idx:
            continue
        idx_t = torch.tensor(idx, dtype=torch.long)
        feat_sub = features[idx_t]
        labels_sub = {k: v[idx_t] for k, v in labels.items()}
        masks_sub = {k: v[idx_t] for k, v in masks.items()}
        loaders[split_name] = DataLoader(
            StrainDataset(feat_sub, labels_sub, masks_sub),
            batch_size=args.batch,
            shuffle=(split_name == "train"),
            collate_fn=collate,
            num_workers=0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicrobeFoundationModel(features.shape[1], specs, hidden=args.hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {n_params:,} parameters  (device={device})")

    class_weights = None
    if args.class_weights:
        train_idx = torch.tensor(splits["train"], dtype=torch.long)
        class_weights = compute_class_weights(labels, masks, specs, train_idx)
        print(f"\nclass-weighted loss enabled — weights computed for {len(class_weights)} heads")

    print(f"\ntraining for {args.epochs} epochs...")
    train(model, loaders["train"], loaders.get("val"), specs, device, args.epochs, args.lr,
          class_weights=class_weights)

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
            "epochs": args.epochs,
            "n_params": n_params,
            "feature_dim": features.shape[1],
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


if __name__ == "__main__":
    main()
