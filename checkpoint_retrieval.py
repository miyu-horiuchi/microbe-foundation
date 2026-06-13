"""Retrieval-augmented head on the *trained model* (real-system cross-clade).

`retrieval_head.py` (Table 16) blends a fresh logistic-regression probe with
cross-clade k-NN. Reviewers will ask the sharper question: does retrieval still
help the *actual trained encoder*, not a linear probe on raw embeddings? This
script answers it by blending the trained model's own per-genome probabilities
with cross-clade k-NN:

    p_blend = alpha * p_model + (1 - alpha) * p_knn,

with `alpha` tuned on family-val and evaluated once on family-test. `alpha* = 1`
means the model already extracts what the embedding offers and retrieval adds
nothing; `alpha* < 1` means k-NN recovers signal the model misses on novel clades.

It is deliberately torch-free: the model emits its probabilities on the GPU box
via `model.py --save-all-predictions`, and this CPU script consumes that parquet
plus the ESM-2 embeddings used for the k-NN reference. The k-NN reference is the
family-train set only; alpha is chosen on family-val; family-test is scored once.

Usage:
    # 1) on the GPU box, after training a checkpoint:
    python model.py --per-protein data/esm2_perprotein --pooling set_transformer \\
        --split-level family --save-all-predictions runs/preds_st_family.parquet
    # 2) anywhere (CPU):
    python checkpoint_retrieval.py --preds runs/preds_st_family.parquet \\
        --features data/esm2_features.npz --out-dir paper/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cross_clade_diagnostic import _macro_f1, knn_majority_predict, K_NN
from ood_error_analysis import binary_trait_labels
from retrieval_head import blend, tune_alpha


def _train_reference(tr: pd.DataFrame, trait: str, fam: dict,
                     id_to_row: dict, feats: np.ndarray):
    """Family-train embeddings + binary labels for `trait` (the k-NN reference)."""
    y, mask = binary_trait_labels(tr[trait])
    sub = tr.assign(_y=y, _m=mask)
    sub = sub[sub["_m"].astype(bool)]
    sub = sub[sub["bid"].map(lambda b: fam.get(b)) == "train"]
    sub = sub[sub["bid"].isin(id_to_row)]
    if sub.empty:
        return None
    rows = sub["bid"].map(id_to_row).to_numpy()
    return feats[rows], sub["_y"].to_numpy(dtype=int)


def _query_arrays(preds: pd.DataFrame, trait: str, id_to_row: dict, feats: np.ndarray):
    """(model_prob, y_true, X_query) for one trait's genomes that have embeddings."""
    sub = preds[(preds["trait"] == trait)].copy()
    sub = sub[sub["bid"].isin(id_to_row)]
    if sub.empty:
        return None
    rows = sub["bid"].map(id_to_row).to_numpy()
    return (sub["pred"].to_numpy(dtype=float),
            sub["true_label"].to_numpy(dtype=int),
            feats[rows])


def evaluate_trait(model_val, y_val, X_val, model_test, y_test, X_test,
                   X_ref, y_ref, k: int = K_NN) -> dict:
    """Blend trained-model probs with k-NN; tune alpha on val, score test once."""
    scaler = StandardScaler().fit(X_ref)
    Xref_s = scaler.transform(X_ref)
    knn_val = knn_majority_predict(Xref_s, y_ref, scaler.transform(X_val), k=k)
    knn_test = knn_majority_predict(Xref_s, y_ref, scaler.transform(X_test), k=k)

    alpha_star = tune_alpha(model_val, knn_val, y_val)
    blend_test = blend(model_test, knn_test, alpha_star)

    model_f1 = _macro_f1(y_test, model_test)
    blend_f1 = _macro_f1(y_test, blend_test)
    return {
        "alpha_star": alpha_star,
        "n_ref": int(len(y_ref)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "pos_rate_test": float(np.mean(y_test)),
        "model_f1": model_f1,
        "model_auroc": float(roc_auc_score(y_test, model_test)),
        "knn_f1": _macro_f1(y_test, knn_test),
        "knn_auroc": float(roc_auc_score(y_test, knn_test)),
        "blend_f1": blend_f1,
        "blend_auroc": float(roc_auc_score(y_test, blend_test)),
        "delta_f1": float(blend_f1 - model_f1),
    }


def run(preds_path, features_path, splits_path, traits_path) -> dict:
    data = np.load(features_path)
    feats = data["features"]
    ids = np.array([str(i) for i in data["bacdive_ids"]])
    id_to_row = {b: i for i, b in enumerate(ids)}

    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    fam = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))

    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)

    preds = pd.read_parquet(preds_path)
    preds["bid"] = preds["bacdive_id"].astype(str)
    val = preds[preds["split"] == "val"]
    test = preds[preds["split"] == "test"]

    results = {}
    for trait in sorted(set(test["trait"]) & set(val["trait"])):
        if trait not in tr.columns:
            continue
        ref = _train_reference(tr, trait, fam, id_to_row, feats)
        qv = _query_arrays(val, trait, id_to_row, feats)
        qt = _query_arrays(test, trait, id_to_row, feats)
        if ref is None or qv is None or qt is None:
            print(f"  skip {trait}: missing reference or query genomes")
            continue
        X_ref, y_ref = ref
        model_val, y_val, X_val = qv
        model_test, y_test, X_test = qt
        if any(len(np.unique(y)) < 2 for y in (y_ref, y_val, y_test)):
            print(f"  skip {trait}: single-class split")
            continue
        results[trait] = evaluate_trait(model_val, y_val, X_val,
                                        model_test, y_test, X_test, X_ref, y_ref)
    return results


def to_markdown(results: dict) -> str:
    lines = [
        "# Table 19 — Retrieval-augmented head on the trained model (cross-clade)",
        "",
        "Convex blend of the *trained model's* per-genome probabilities and "
        "cross-clade k-NN, `alpha * model + (1 - alpha) * knn`, with `alpha` tuned on "
        "family-val and evaluated on family-test. `alpha* = 1` means the trained model "
        "already extracts what the embedding offers; `alpha* < 1` means k-NN recovers "
        "signal the model misses on novel families. `delta F1` is blend minus model-alone.",
        "",
        "| Trait | Test n | Pos rate | alpha* | Model F1 | k-NN F1 | Blend F1 | delta F1 | "
        "Model AUROC | k-NN AUROC | Blend AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trait, r in results.items():
        lines.append(
            f"| `{trait}` | {r['n_test']} | {r['pos_rate_test']:.3f} | {r['alpha_star']:.1f} | "
            f"{r['model_f1']:.3f} | {r['knn_f1']:.3f} | {r['blend_f1']:.3f} | {r['delta_f1']:+.3f} | "
            f"{r['model_auroc']:.3f} | {r['knn_auroc']:.3f} | {r['blend_auroc']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True,
                    help="Parquet from model.py --save-all-predictions "
                         "(bacdive_id, split, trait, true_label, pred).")
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits", default="data/traits.parquet")
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    results = run(args.preds, args.features, args.splits, args.traits)
    if not results:
        raise SystemExit("no traits scored — check that --preds has val+test rows with "
                         "both classes and matching embeddings.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(results)
    (out_dir / "19_checkpoint_retrieval.md").write_text(md)
    pd.DataFrame([{"trait": t, **r} for t, r in results.items()]).to_csv(
        out_dir / "19_checkpoint_retrieval.csv", index=False
    )
    print(md)
    print(f"wrote {out_dir / '19_checkpoint_retrieval.md'} and .csv")


if __name__ == "__main__":
    main()
