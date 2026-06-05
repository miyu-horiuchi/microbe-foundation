"""
extract_attention.py — dump per-protein attention weights from a trained
attention-pool checkpoint, for the interpretability analysis (Phase 2).

For each held-out TEST genome it runs the model with AttentionPool.store_attn on,
reads the [P] softmax weights over that genome's proteins, and records the
top-attended protein indices (which map row-for-row to the cached protein
sequences in proteins/<bid>.txt.gz). Downstream (Phase 3) those indices are
annotated against VFDB to ask: are the up-weighted proteins virulence factors?

Run a checkpoint saved by `model.py --save-model` (ideally a --single-task
pathogenicity model so the shared pool specializes to that trait):

    python extract_attention.py --checkpoint runs/patho-animal.pt \
        --per-protein data/esm2_perprotein --split-level species \
        --head pathogenicity_animal --top-k 30 --out runs/attn_patho_animal.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# import model.py as a module (reuse its label/spec machinery + paths)
_spec = importlib.util.spec_from_file_location("microbe_model_main", Path(__file__).parent / "model.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def normalized_entropy(w: np.ndarray) -> float:
    """Entropy of the attention distribution / log(P): 0=all weight on one protein,
    1=uniform. Low values are the precondition for an interpretable spotlight."""
    w = w[w > 0]
    if w.size <= 1:
        return 0.0
    return float(-(w * np.log(w)).sum() / np.log(w.size))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help=".pt from model.py --save-model")
    p.add_argument("--per-protein", type=Path, required=True, help="dir of <bid>.npy + manifest.parquet")
    p.add_argument("--split-level", choices=["species", "genus", "family"], default="species")
    p.add_argument("--head", type=str, required=True, help="head to read predictions for (e.g. pathogenicity_animal)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="which split to extract")
    p.add_argument("--top-k", type=int, default=30, help="how many top-attended protein indices to record per genome")
    p.add_argument("--out", type=Path, required=True, help="output parquet")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # our checkpoint is a state_dict + primitive metadata only, so weights_only=True
    # is safe (no arbitrary-object unpickling / code execution).
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not ck.get("attention_pool"):
        raise SystemExit("checkpoint is not an attention-pool model — nothing to extract")
    model = M.MicrobeFoundationModel(
        ck["input_dim"], {h: {"size": s} for h, s in ck["head_sizes"].items()},
        hidden=ck["hidden"], attention_pool=True,
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    model.pool.store_attn = True
    if args.head not in ck["head_sizes"]:
        raise SystemExit(f"--head {args.head} not in checkpoint heads {list(ck['head_sizes'])}")

    # rebuild the same dataframe + per-protein paths + labels as training
    schema = M.json.loads(M.SCHEMA_PATH.read_text())
    vocab = M.json.loads(M.VOCAB_PATH.read_text())
    traits_df = pd.read_parquet(M.TRAITS_PATH)
    splits_df = pd.read_parquet(M.SPLITS_PATH)
    df = traits_df.merge(splits_df[["bacdive_id", f"{args.split_level}_split"]], on="bacdive_id", how="left")
    df = df.rename(columns={f"{args.split_level}_split": "split"})
    manifest = pd.read_parquet(args.per_protein / "manifest.parquet")
    ok = manifest[manifest.status == "ok"]
    id_to_path = {int(b): args.per_protein / p for b, p in zip(ok.bacdive_id, ok.path)}
    df = df[df.bacdive_id.map(id_to_path.__contains__).fillna(False).values].reset_index(drop=True)
    labels, masks, specs = M.prepare_labels(df, vocab, schema)

    sel = df.index[df.split == args.split].tolist()
    lab = labels[args.head]; msk = masks[args.head]
    print(f"{len(sel):,} {args.split} genomes; head={args.head}")

    rows = []
    with torch.inference_mode():
        for i in sel:
            bid = int(df.bacdive_id.iloc[i])
            arr = np.load(id_to_path[bid]).astype(np.float32)  # [P, D], full proteome (no subsample)
            P = arr.shape[0]
            if P == 0:
                continue
            x = torch.from_numpy(arr).unsqueeze(0).to(device)            # [1, P, D]
            mask = torch.ones(1, P, dtype=torch.float32, device=device)  # all real
            out = model(x, mask)
            w = model.pool.last_attn[0].float().cpu().numpy()            # [P]
            logit = out[args.head][0].float().cpu().numpy()
            pred = float(1 / (1 + np.exp(-logit[0]))) if logit.size == 1 else float(logit.argmax())
            k = min(args.top_k, P)
            top = np.argsort(-w)[:k]
            has_label = bool(msk[i].item()) if hasattr(msk[i], "item") else bool(msk[i])
            true = float(lab[i].item()) if has_label else None
            rows.append({
                "bacdive_id": bid, "n_proteins": P,
                "true_label": true, "pred": pred,
                "attn_entropy_norm": normalized_entropy(w),
                "top_idx": top.tolist(),
                "top_weight": w[top].round(5).tolist(),
                "top_mass": float(w[top].sum()),   # fraction of attention on the top-k
            })
    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"wrote {len(out_df):,} rows -> {args.out}")
    lp = out_df[out_df.true_label == 1]
    print(f"  labeled positive (pathogenic): {len(lp):,}")
    print(f"  median attn entropy: {out_df.attn_entropy_norm.median():.3f} "
          f"(low=concentrated); median top-{args.top_k} mass: {out_df.top_mass.median():.3f}")


if __name__ == "__main__":
    main()
