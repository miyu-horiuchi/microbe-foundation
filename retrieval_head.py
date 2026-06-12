"""Retrieval-augmented head for cross-clade trait transfer.

The cross-clade diagnostic (Table 15) ruled out a frozen-representation wall:
k-NN label transfer in the 640-d ESM-2 space matches the trained probe and beats
chance, so the embedding geometry carries trait signal across novel families. This
script tests whether *exploiting* that geometry helps: a convex blend of the linear
probe and cross-clade k-NN,

    p_blend = alpha * p_probe + (1 - alpha) * p_knn,

with alpha tuned on a held-out family-val set (families disjoint from both train and
test) and evaluated on family-test. We report the blend against probe-alone (alpha=1)
and k-NN-alone (alpha=0). A null result is informative: if alpha* ~= 1 everywhere, the
probe already extracts what the embedding offers and retrieval adds nothing.

Leakage control: the standardizer and both the probe and the k-NN reference manifold
are fit on family-train only; alpha is chosen on family-val; family-test is touched
once, for the final numbers.

CPU-only. Reuses the tested helpers from cross_clade_diagnostic and ood_error_analysis.

Usage:
    python retrieval_head.py
    python retrieval_head.py --out-dir paper/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cross_clade_diagnostic import _macro_f1, knn_majority_predict, K_NN
from ood_error_analysis import binary_trait_labels

TRAITS = ["sporulation", "motility", "catalase"]
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def blend(probe_prob, knn_prob, alpha: float) -> np.ndarray:
    """Convex combination: alpha * probe + (1 - alpha) * knn."""
    probe_prob = np.asarray(probe_prob, dtype=float)
    knn_prob = np.asarray(knn_prob, dtype=float)
    return alpha * probe_prob + (1.0 - alpha) * knn_prob


def tune_alpha(probe_val, knn_val, y_val, grid=ALPHA_GRID) -> float:
    """Return the grid alpha maximizing val macro-F1 of the blend.

    Ties resolve to the smallest alpha, so when k-NN alone is at least as good the
    chosen alpha leans toward retrieval rather than the probe.
    """
    f1s = [_macro_f1(y_val, blend(probe_val, knn_val, a)) for a in grid]
    return float(grid[int(np.argmax(f1s))])


def prepare_trait_tvt(feats, tr, trait):
    """Train/val/test feature+label arrays for the family split (disjoint families).

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    y, mask = binary_trait_labels(tr[trait])
    sub = tr[mask]
    ys = y[mask]
    out = []
    for part in ("train", "val", "test"):
        sel = (sub["fsplit"] == part).to_numpy()
        out.append(feats[sub["row"].to_numpy()[sel]])
        out.append(ys[sel])
    return tuple(out)


def evaluate_retrieval(X_train, y_train, X_val, y_val, X_test, y_test, k: int = K_NN) -> dict:
    """Fit probe + k-NN on train, tune alpha on val, evaluate the blend on test.

    The standardizer and probe are fit on train only; the k-NN reference is the
    standardized train set. alpha is chosen on val; test is scored once.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xva_s = scaler.transform(X_val)
    Xte_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, y_train)
    probe_val = clf.predict_proba(Xva_s)[:, 1]
    probe_test = clf.predict_proba(Xte_s)[:, 1]

    knn_val = knn_majority_predict(Xtr_s, y_train, Xva_s, k=k)
    knn_test = knn_majority_predict(Xtr_s, y_train, Xte_s, k=k)

    alpha_star = tune_alpha(probe_val, knn_val, y_val)
    blend_test = blend(probe_test, knn_test, alpha_star)

    probe_f1 = _macro_f1(y_test, probe_test)
    blend_f1 = _macro_f1(y_test, blend_test)
    return {
        "alpha_star": alpha_star,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "pos_rate_test": float(np.mean(y_test)),
        "probe_f1": probe_f1,
        "probe_auroc": float(roc_auc_score(y_test, probe_test)),
        "knn_f1": _macro_f1(y_test, knn_test),
        "knn_auroc": float(roc_auc_score(y_test, knn_test)),
        "blend_f1": blend_f1,
        "blend_auroc": float(roc_auc_score(y_test, blend_test)),
        "delta_f1": float(blend_f1 - probe_f1),
    }


def run(features_path, splits_path, traits_path, traits=TRAITS) -> dict:
    data = np.load(features_path)
    feats = data["features"]
    ids = np.array([str(i) for i in data["bacdive_ids"]])
    id_to_row = {b: i for i, b in enumerate(ids)}
    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    tr = tr[tr["bid"].isin(id_to_row)].copy()
    sp = pd.read_parquet(splits_path)[["bacdive_id", "family_split"]]
    fam = dict(zip(sp["bacdive_id"].astype(str), sp["family_split"]))
    tr["fsplit"] = tr["bid"].map(lambda b: fam.get(b, "unknown"))
    tr["row"] = tr["bid"].map(id_to_row)

    results = {}
    for trait in traits:
        if trait not in tr.columns:
            continue
        Xtr, ytr, Xva, yva, Xte, yte = prepare_trait_tvt(feats, tr, trait)
        # Need both classes in every split to fit a probe and score AUROC.
        if any(len(np.unique(y)) < 2 for y in (ytr, yva, yte)):
            print(f"  skip {trait}: single-class split")
            continue
        results[trait] = evaluate_retrieval(Xtr, ytr, Xva, yva, Xte, yte)
    return results


def to_markdown(results: dict) -> str:
    lines = [
        "# Table 16 — Retrieval-augmented head (cross-clade)",
        "",
        "Convex blend of the linear probe and cross-clade k-NN, `alpha * probe + "
        "(1 - alpha) * knn`, with `alpha` tuned on family-val and evaluated on "
        "family-test. `alpha* = 1` means the probe alone wins (retrieval adds nothing); "
        "`alpha* = 0` means k-NN alone wins. `delta F1` is blend minus probe-alone on "
        "family-test.",
        "",
        "| Trait | Test n | Pos rate | alpha* | Probe F1 | k-NN F1 | Blend F1 | delta F1 | "
        "Probe AUROC | k-NN AUROC | Blend AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trait, r in results.items():
        lines.append(
            f"| `{trait}` | {r['n_test']} | {r['pos_rate_test']:.3f} | {r['alpha_star']:.1f} | "
            f"{r['probe_f1']:.3f} | {r['knn_f1']:.3f} | {r['blend_f1']:.3f} | {r['delta_f1']:+.3f} | "
            f"{r['probe_auroc']:.3f} | {r['knn_auroc']:.3f} | {r['blend_auroc']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits", default="data/traits.parquet")
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    results = run(args.features, args.splits, args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(results)
    (out_dir / "16_retrieval_head.md").write_text(md)
    pd.DataFrame([{"trait": t, **r} for t, r in results.items()]).to_csv(
        out_dir / "16_retrieval_head.csv", index=False
    )
    print(md)


if __name__ == "__main__":
    main()
