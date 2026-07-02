#!/usr/bin/env python3
"""overfit_gap.py -- separate overfitting from OOD-generalization gap.

The diagnostic question: when accuracy drops on the family split, is it because
the model MEMORISED the training genomes (overfitting -- a train>>test gap that
shows up even on the easy split), or because it cannot EXTRAPOLATE to unseen
taxa (an OOD-coverage gap -- train~=test on the easy split, but test falls as we
hold out higher taxonomic ranks)?

We train the encoder MLP + heads on pre-pooled (mean-pooled) ESM-2 features for
each cross-clade split, then evaluate the SAME model on its own train set and on
the held-out test set. The macro train-vs-test gap, read across species/genus/
family, tells the two apart:

    big train>>test gap already on species  -> overfitting (variance)
    train~=test on species, test falls on family -> OOD coverage limit

This uses the mean-pool condition (cheap, CPU, runs from local
data/esm2_features.npz). Mean-pool tied the attention / set_transformer poolers
in the tier-1 matrix, so the qualitative conclusion transfers to those models.

Outputs
-------
    paper/tables/27_overfit_gap.csv / .md          per-split macro train/test/gap
    paper/tables/27_overfit_gap_perhead.csv        per-head train/test (find
                                                    data-starved, overfit-prone heads)
    paper/figures/27_overfit_gap.png / .pdf        grouped train-vs-test bars

Usage
-----
    python3 paper/overfit_gap.py --epochs 40 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import model as M  # noqa: E402

SPLIT_ORDER = {"species": 0, "genus": 1, "family": 2}


def build_split_loaders(features_path: Path, split_level: str, batch: int):
    schema = M.json.loads(M.SCHEMA_PATH.read_text())
    vocab = M.json.loads(M.VOCAB_PATH.read_text())
    traits_df = M.pd.read_parquet(M.TRAITS_PATH)
    splits_df = M.pd.read_parquet(M.SPLITS_PATH)
    df = traits_df.merge(splits_df[["bacdive_id", f"{split_level}_split"]],
                         on="bacdive_id", how="left")
    df = df.rename(columns={f"{split_level}_split": "split"})

    npz = np.load(features_path)
    feat_ids = npz["bacdive_ids"]
    feat_mat = npz["features"]
    id_to_row = {int(b): i for i, b in enumerate(feat_ids)}
    keep = df.bacdive_id.map(id_to_row.__contains__).fillna(False).values
    df = df[keep].reset_index(drop=True)
    rows = df.bacdive_id.map(id_to_row).values
    features = torch.tensor(feat_mat[rows], dtype=torch.float32)
    input_dim = features.shape[1]

    labels, masks, specs = M.prepare_labels(df, vocab, schema)
    loaders = {}
    for s in ("train", "val", "test"):
        idx = df.index[df.split == s].tolist()
        if not idx:
            continue
        idx_t = torch.tensor(idx, dtype=torch.long)
        ds = M.StrainDataset(features[idx_t],
                             {k: v[idx_t] for k, v in labels.items()},
                             {k: v[idx_t] for k, v in masks.items()})
        loaders[s] = DataLoader(ds, batch_size=batch, shuffle=(s == "train"),
                                collate_fn=M.collate, num_workers=0)
    return loaders, specs, input_dim


def macro(metrics: dict, specs: dict) -> float:
    return M._avg_primary(metrics, specs)


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in xs) / (n - 1)
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 1.96)
    return m, t * math.sqrt(var) / math.sqrt(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=REPO / "data/esm2_features.npz")
    ap.add_argument("--splits", nargs="+", default=["species", "genus", "family"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--out-dir", type=Path, default=REPO / "paper")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  features={args.features.name}  "
          f"splits={args.splits}  seeds={args.seeds}  epochs={args.epochs}")

    # per (split) -> lists across seeds
    train_macros: dict[str, list[float]] = defaultdict(list)
    test_macros: dict[str, list[float]] = defaultdict(list)
    perhead_rows = []

    for split in args.splits:
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            loaders, specs, input_dim = build_split_loaders(args.features, split, args.batch)
            model = M.MicrobeFoundationModel(input_dim, specs, hidden=args.hidden,
                                             dropout=args.dropout, pooling=None).to(device)
            print(f"\n=== split={split} seed={seed}  "
                  f"(train={len(loaders['train'].dataset)}, "
                  f"test={len(loaders.get('test', loaders['train']).dataset)}) ===")
            M.train(model, loaders["train"], loaders.get("val"), specs, device,
                    epochs=args.epochs, lr=args.lr)

            tr = M.run_eval(model, loaders["train"], specs, device)
            te = M.run_eval(model, loaders["test"], specs, device)
            mtr, mte = macro(tr, specs), macro(te, specs)
            train_macros[split].append(mtr)
            test_macros[split].append(mte)
            print(f"  --> macro train={mtr:.4f}  test={mte:.4f}  gap={mtr - mte:+.4f}")

            for name in specs:
                pk = M._primary_metric(specs[name]["head_type"])
                if name in tr and name in te:
                    perhead_rows.append({
                        "split": split, "seed": seed, "head": name, "metric": pk,
                        "train": round(tr[name][pk], 4), "test": round(te[name][pk], 4),
                        "gap": round(tr[name][pk] - te[name][pk], 4),
                    })

    # ---- summary table ----
    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = []
    for split in sorted(args.splits, key=lambda s: SPLIT_ORDER.get(s, 9)):
        mtr, ctr = mean_ci(train_macros[split])
        mte, cte = mean_ci(test_macros[split])
        gap = mtr - mte
        summary.append({
            "split": split, "n_seeds": len(train_macros[split]),
            "macro_train": round(mtr, 4), "ci_train": round(ctr, 4),
            "macro_test": round(mte, 4), "ci_test": round(cte, 4),
            "gap_train_minus_test": round(gap, 4),
        })

    with (tables / "27_overfit_gap.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with (tables / "27_overfit_gap_perhead.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perhead_rows[0].keys()))
        w.writeheader()
        w.writerows(perhead_rows)

    md = ["# Overfitting vs OOD-gap diagnostic (mean-pool ESM-2)",
          "",
          f"epochs={args.epochs}  seeds={args.seeds}  features={args.features.name}",
          "",
          "Read the gap ACROSS splits: a large train>>test gap on `species` means",
          "overfitting; a small gap on `species` that grows on `family` means an",
          "OOD-coverage limit (the model generalises in-distribution but cannot",
          "extrapolate to unseen taxa).",
          "",
          "| split | n | macro train | macro test | gap (train-test) |",
          "|-------|---|-------------|------------|------------------|"]
    for r in summary:
        md.append(f"| {r['split']} | {r['n_seeds']} | "
                  f"{r['macro_train']:.4f} ±{r['ci_train']:.4f} | "
                  f"{r['macro_test']:.4f} ±{r['ci_test']:.4f} | "
                  f"{r['gap_train_minus_test']:+.4f} |")
    (tables / "27_overfit_gap.md").write_text("\n".join(md) + "\n")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        splits = [r["split"] for r in summary]
        x = np.arange(len(splits))
        tr_m = [r["macro_train"] for r in summary]
        te_m = [r["macro_test"] for r in summary]
        tr_c = [r["ci_train"] for r in summary]
        te_c = [r["ci_test"] for r in summary]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.bar(x - 0.2, tr_m, 0.4, yerr=tr_c, capsize=3, label="train",
               color="#7fb3d5")
        ax.bar(x + 0.2, te_m, 0.4, yerr=te_c, capsize=3, label="test (held-out clade)",
               color="#c0392b")
        for xi, (t, e) in enumerate(zip(tr_m, te_m)):
            ax.text(xi, max(t, e) + 0.012, f"gap {t - e:+.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in splits])
        ax.set_ylabel("Macro trait-prediction score")
        ax.set_title("Train vs held-out test by cross-clade split (mean-pool ESM-2)")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        figs = args.out_dir / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / "27_overfit_gap.png", dpi=200, bbox_inches="tight")
        fig.savefig(figs / "27_overfit_gap.pdf", bbox_inches="tight")
        print(f"\nwrote {figs / '27_overfit_gap.png'} (+ .pdf)")
    except Exception as e:  # pragma: no cover
        print(f"[warn] figure skipped: {e}")

    print("\n" + "\n".join(md))
    print(f"\nwrote {tables / '27_overfit_gap.csv'}, .md, _perhead.csv")


if __name__ == "__main__":
    main()
