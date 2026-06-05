"""
ablate_attention.py — Phase 4 of pathogenicity interpretability (causal test).

Enrichment (Phase 3) shows attention *correlates* with virulence factors. This
tests whether the model *relies* on them: for each confidently-pathogenic genome,
re-predict pathogenicity after masking out the top-k attended proteins, and
compare the prediction drop to masking k random proteins. If removing the
attended proteins collapses the prediction far more than removing random ones,
the attention is causally load-bearing.

    python ablate_attention.py --checkpoint runs/pathogenicity_animal-species.pt \
        --attn runs/attn-pathogenicity_animal-species.parquet --npy-dir /tmp/npy \
        --head pathogenicity_animal --k 5
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

_spec = importlib.util.spec_from_file_location("mm", Path(__file__).parent / "model.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def predict(model, x, drop_idx, head):
    """Sigmoid prediction for `head` with the proteins in drop_idx masked out."""
    P = x.shape[1]
    mask = torch.ones(1, P)
    if len(drop_idx):
        mask[0, list(drop_idx)] = 0.0
    with torch.inference_mode():
        logit = model(x, mask)[head][0, 0].item()
    return 1.0 / (1.0 + np.exp(-logit))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--attn", type=Path, required=True)
    p.add_argument("--npy-dir", type=Path, required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = M.MicrobeFoundationModel(ck["input_dim"], {h: {"size": s} for h, s in ck["head_sizes"].items()},
                                     hidden=ck["hidden"], attention_pool=True)
    model.load_state_dict(ck["state_dict"]); model.eval()

    d = pd.read_parquet(args.attn)
    d = d[(d.true_label == 1) & (d.pred > 0.5)]   # confidently-correct pathogenic genomes
    rows = []
    for _, r in d.iterrows():
        bid = int(r.bacdive_id)
        f = args.npy_dir / f"{bid}.npy"
        if not f.exists():
            continue
        arr = np.load(f).astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0)
        P = arr.shape[0]
        top = [i for i in r.top_idx[:args.k] if i < P]
        randk = rng.choice([i for i in range(P) if i not in set(top)], size=min(args.k, P - 1), replace=False)
        base = predict(model, x, [], args.head)
        rows.append({
            "bacdive_id": bid,
            "base": base,
            "drop_top": base - predict(model, x, top, args.head),
            "drop_rand": base - predict(model, x, randk, args.head),
        })
    df = pd.DataFrame(rows)
    print(f"genomes: {len(df)}  (k={args.k} proteins masked)")
    print(f"  baseline pred (mean):            {df.base.mean():.3f}")
    print(f"  drop after masking TOP-{args.k}:      {df.drop_top.mean():.3f}  (median {df.drop_top.median():.3f})")
    print(f"  drop after masking RANDOM-{args.k}:   {df.drop_rand.mean():.3f}  (median {df.drop_rand.median():.3f})")
    print(f"  ratio top/random:                {df.drop_top.mean() / max(df.drop_rand.mean(), 1e-9):.1f}x")
    w = stats.wilcoxon(df.drop_top, df.drop_rand, alternative="greater", zero_method="zsplit")
    print(f"  Wilcoxon (drop_top > drop_rand): p={w.pvalue:.2e}")
    flipped = (df.base - df.drop_top < 0.5).mean()
    print(f"  genomes flipped to non-pathogenic by removing top-{args.k}: {flipped:.1%}")


if __name__ == "__main__":
    main()
