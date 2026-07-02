"""
encoder_comparison.py — Lever 2: does a stronger genome encoder lift cross-clade transfer?

We hold everything fixed (family split, balanced linear probe, traits, evaluation) and
swap only the frozen genome representation, comparing the 150M-parameter ESM-2 protein
LM (640-d, mean-pooled) against Bacformer (960-d), a bacterial-genome foundation model.
To make it a clean encoder-only contrast, both are evaluated on the *same* genomes (the
intersection of the two feature stores) with identical labels and the same family-held-out
test set, for the solved and coverage-limited binary traits.

A positive Bacformer-minus-ESM2 delta on family-test AUROC/F1 is direct evidence that
encoder quality (not pooling or retrieval) is a live lever for cross-clade generalization,
and motivates re-extracting features from a larger ESM-2 (650M/3B) on the GPU box.

CPU-only; both representations are pre-computed.

Usage:
    python paper/encoder_comparison.py
    python paper/encoder_comparison.py --encoders esm2=data/esm2_features.npz bacformer=data/bacformer_features_all.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_clade_diagnostic import _macro_f1, _probe  # noqa: E402
from ood_error_analysis import binary_trait_labels  # noqa: E402

SOLVED = ["catalase", "cytochrome_oxidase", "sporulation", "pigmentation"]
COVERAGE_LIMITED = ["pathogenicity_human", "pathogenicity_animal", "motility"]
TRAITS = SOLVED + COVERAGE_LIMITED

DEFAULT_ENCODERS = {
    "esm2_150M": "data/esm2_features.npz",
    "bacformer": "data/bacformer_features_all.npz",
}


def load_features(path: str):
    d = np.load(path)
    feats = d["features"]
    ids = [str(i) for i in d["bacdive_ids"]]
    return feats, ids


def common_ids(id_lists: list[list[str]]) -> list[str]:
    """Sorted intersection of bacdive ids present in every encoder."""
    s = set(id_lists[0])
    for ids in id_lists[1:]:
        s &= set(ids)
    return sorted(s)


def delta_row(trait: str, per_enc: dict, base: str, other: str, mode: str) -> dict:
    """One result row with the other-minus-base deltas for a trait."""
    b, o = per_enc[base], per_enc[other]
    return {
        "trait": trait, "mode": mode, "n_test": b["n_test"], "pos_rate": b["pos_rate"],
        f"{base}_f1": b["f1"], f"{other}_f1": o["f1"], "d_f1": o["f1"] - b["f1"],
        f"{base}_auroc": b["auroc"], f"{other}_auroc": o["auroc"], "d_auroc": o["auroc"] - b["auroc"],
    }


def evaluate(trait, feats_by_enc, row_by_enc, tr_common):
    """Family-split probe for each encoder on the shared genome set; per-encoder metrics."""
    y, mask = binary_trait_labels(tr_common[trait])
    sub = tr_common[mask].copy()
    ys = y[mask]
    is_tr = (sub["fsplit"] == "train").to_numpy()
    is_te = (sub["fsplit"] == "test").to_numpy()
    if len(np.unique(ys[is_tr])) < 2 or len(np.unique(ys[is_te])) < 2:
        return None
    bids_tr = sub["bid"].to_numpy()[is_tr]
    bids_te = sub["bid"].to_numpy()[is_te]
    out = {}
    for enc, feats in feats_by_enc.items():
        r2row = row_by_enc[enc]
        Xtr = feats[[r2row[b] for b in bids_tr]]
        Xte = feats[[r2row[b] for b in bids_te]]
        prob = _probe(Xtr, ys[is_tr], Xte)
        out[enc] = {
            "f1": _macro_f1(ys[is_te], prob),
            "auroc": float(roc_auc_score(ys[is_te], prob)),
            "n_test": int(is_te.sum()), "pos_rate": float(ys[is_te].mean()),
        }
    return out


def run(encoders: dict, splits_path, traits_path, traits=TRAITS):
    feats_by_enc, ids_by_enc = {}, {}
    for enc, path in encoders.items():
        feats_by_enc[enc], ids_by_enc[enc] = load_features(path)
    cids = common_ids(list(ids_by_enc.values()))
    row_by_enc = {enc: {b: i for i, b in enumerate(ids)} for enc, ids in ids_by_enc.items()}

    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    fam = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    tr["fsplit"] = tr["bid"].map(lambda b: fam.get(b, "unknown"))
    tr_common = tr[tr["bid"].isin(set(cids))].copy()

    base = list(encoders.keys())[0]
    other = list(encoders.keys())[1]
    rows = []
    for trait in traits:
        if trait not in tr_common.columns:
            continue
        per_enc = evaluate(trait, feats_by_enc, row_by_enc, tr_common)
        if per_enc is None:
            continue
        mode = "solved" if trait in SOLVED else "coverage-limited"
        rows.append(delta_row(trait, per_enc, base, other, mode))
    return rows, base, other, len(cids)


def to_markdown(rows, base, other, n_common) -> str:
    lines = [
        f"# Table 23 — Encoder comparison: {other} vs {base} (family-held-out)",
        "",
        f"Same {n_common} shared genomes, same family split, same balanced linear probe; "
        f"only the frozen genome representation changes. Δ = {other} − {base}. A positive "
        "Δ means the stronger encoder recovers cross-clade signal that the pooling and "
        "retrieval levers could not.",
        "",
        f"| Trait | Mode | Test n | Pos rate | {base} F1 | {other} F1 | ΔF1 | "
        f"{base} AUROC | {other} AUROC | ΔAUROC |",
        "|---|:--|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['trait']}` | {r['mode']} | {r['n_test']} | {r['pos_rate']:.3f} | "
            f"{r[f'{base}_f1']:.3f} | {r[f'{other}_f1']:.3f} | {r['d_f1']:+.3f} | "
            f"{r[f'{base}_auroc']:.3f} | {r[f'{other}_auroc']:.3f} | {r['d_auroc']:+.3f} |"
        )
    if rows:
        df1 = float(np.mean([r["d_f1"] for r in rows]))
        dau = float(np.mean([r["d_auroc"] for r in rows]))
        lines += ["", f"Mean Δ across {len(rows)} traits: ΔF1 {df1:+.3f}, ΔAUROC {dau:+.3f}."]
    return "\n".join(lines) + "\n"


def save_figure(rows, base, other, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["trait"] for r in rows]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.6))
    ax.bar(x - w / 2, [r[f"{base}_auroc"] for r in rows], w, label=base, color="#1f77b4")
    ax.bar(x + w / 2, [r[f"{other}_auroc"] for r in rows], w, label=other, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("family-test AUROC")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_title(f"Lever 2: encoder swap on shared genomes ({other} vs {base})")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_encoders(items: list[str] | None) -> dict:
    if not items:
        return dict(DEFAULT_ENCODERS)
    out = {}
    for it in items:
        name, path = it.split("=", 1)
        out[name] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=None,
                    help="name=path pairs; first is baseline. Default esm2_150M vs bacformer.")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--out-dir", default="paper/tables")
    ap.add_argument("--fig", default="paper/figures/encoder_comparison.png")
    args = ap.parse_args()

    encoders = parse_encoders(args.encoders)
    rows, base, other, n_common = run(encoders, args.splits, args.traits_path)
    if not rows:
        raise SystemExit("no traits evaluated")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, base, other, n_common)
    (Path(args.out_dir) / "23_encoder_comparison.md").write_text(md)
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "23_encoder_comparison.csv", index=False)
    save_figure(rows, base, other, args.fig)
    print(md)
    print(f"wrote {args.out_dir}/23_encoder_comparison.md/.csv and {args.fig}")


if __name__ == "__main__":
    main()
