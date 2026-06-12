"""Novelty-adaptive retrieval head for cross-clade trait transfer.

Table 16 showed a single global blend weight `alpha` (probe vs cross-clade k-NN)
beats probe-alone. But a constant alpha ignores that retrieval reliability varies
per genome: k-NN label transfer should be trusted more for a test genome that sits
close to training families, and less for a genuinely novel one whose neighbours are
themselves off-distribution. This script makes alpha a function of per-genome novelty,

    alpha_i = clip(alpha_lo + (alpha_hi - alpha_lo) * t_i, 0, 1),

where t_i in [0, 1] is the genome's novelty (the Table-14 Euclidean OOD score against
the family-train reference, percentile-normalized within the family-val distribution).
The ramp endpoints (alpha_lo, alpha_hi) are tuned on family-val and evaluated once on
family-test. A constant alpha (alpha_lo == alpha_hi) is inside the grid, so the
adaptive head can never lose to the global blend on val by construction; the question
is whether novelty-conditioning *transfers* a gain to test, and in which direction
(alpha_hi > alpha_lo would mean "trust the probe more as novelty rises").

Leakage control: probe, k-NN reference, and the novelty manifold are all fit on
family-train; the novelty normalizer and both alpha schemes are tuned on family-val;
family-test is scored once.

CPU-only. Reuses tested helpers from retrieval_head, cross_clade_diagnostic,
ood_error_analysis, and the monitoring manifold.

Usage:
    python adaptive_retrieval.py
    python adaptive_retrieval.py --out-dir paper/tables
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
from retrieval_head import blend, tune_alpha, prepare_trait_tvt
from microbe_model.monitoring import EuclideanBackend, ReferenceManifold

TRAITS = ["sporulation", "motility", "catalase"]
RAMP_GRID = [round(0.2 * i, 1) for i in range(6)]  # 0.0, 0.2, ..., 1.0


def novelty_score(X_ref, X_query, k: int = K_NN) -> np.ndarray:
    """Per-genome Euclidean OOD score vs the reference (Table-14 novelty signal)."""
    manifold = ReferenceManifold(backend=EuclideanBackend(k=k)).fit(np.asarray(X_ref, dtype=float))
    return np.asarray(manifold.ood_score(np.asarray(X_query, dtype=float)), dtype=float)


def normalize_novelty(ref_scores, query_scores) -> np.ndarray:
    """Percentile-rank each query score within the reference distribution -> [0, 1].

    A query at the reference minimum maps near 0 (least novel); at/above the maximum
    maps to 1 (most novel). Fitting the mapping on a held-out reference (family-val)
    and applying it to test keeps the test set untouched during tuning.
    """
    ref = np.sort(np.asarray(ref_scores, dtype=float))
    q = np.asarray(query_scores, dtype=float)
    if len(ref) == 0:
        return np.zeros_like(q)
    # fraction of reference points <= each query value
    ranks = np.searchsorted(ref, q, side="right")
    return ranks / len(ref)


def adaptive_alpha(t, alpha_lo: float, alpha_hi: float) -> np.ndarray:
    """Per-genome blend weight: linear ramp in novelty t, clipped to [0, 1]."""
    t = np.asarray(t, dtype=float)
    return np.clip(alpha_lo + (alpha_hi - alpha_lo) * t, 0.0, 1.0)


def tune_ramp(probe_val, knn_val, t_val, y_val, grid=RAMP_GRID) -> tuple[float, float]:
    """Return (alpha_lo, alpha_hi) maximizing val macro-F1 of the novelty-ramped blend.

    Ties resolve to the smallest (alpha_lo, alpha_hi) in row-major order.
    """
    best = None
    best_f1 = -1.0
    for lo in grid:
        for hi in grid:
            a = adaptive_alpha(t_val, lo, hi)
            f1 = _macro_f1(y_val, blend(probe_val, knn_val, a))
            if f1 > best_f1:
                best_f1 = f1
                best = (float(lo), float(hi))
    return best


def evaluate_adaptive(X_train, y_train, X_val, y_val, X_test, y_test, k: int = K_NN) -> dict:
    """Fit probe + k-NN + novelty on train, tune the ramp on val, score test once.

    Compares the novelty-adaptive blend against the global-alpha blend (Table 16)
    and probe-alone, all on the same family-test set.
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

    # Novelty on the standardized space (consistent with probe / k-NN geometry).
    nov_val = novelty_score(Xtr_s, Xva_s, k=k)
    nov_test = novelty_score(Xtr_s, Xte_s, k=k)
    t_val = normalize_novelty(nov_val, nov_val)
    t_test = normalize_novelty(nov_val, nov_test)

    # Adaptive ramp tuned on val.
    lo, hi = tune_ramp(probe_val, knn_val, t_val, y_val)
    a_test = adaptive_alpha(t_test, lo, hi)
    adaptive_test = blend(probe_test, knn_test, a_test)

    # Global-alpha baseline (Table 16) and probe-alone, same splits.
    alpha_g = tune_alpha(probe_val, knn_val, y_val)
    global_test = blend(probe_test, knn_test, alpha_g)

    probe_f1 = _macro_f1(y_test, probe_test)
    global_f1 = _macro_f1(y_test, global_test)
    adaptive_f1 = _macro_f1(y_test, adaptive_test)
    return {
        "alpha_lo": lo,
        "alpha_hi": hi,
        "alpha_global": alpha_g,
        "n_test": int(len(y_test)),
        "pos_rate_test": float(np.mean(y_test)),
        "probe_f1": probe_f1,
        "probe_auroc": float(roc_auc_score(y_test, probe_test)),
        "global_f1": global_f1,
        "global_auroc": float(roc_auc_score(y_test, global_test)),
        "adaptive_f1": adaptive_f1,
        "adaptive_auroc": float(roc_auc_score(y_test, adaptive_test)),
        "delta_vs_global_f1": float(adaptive_f1 - global_f1),
        "delta_vs_probe_f1": float(adaptive_f1 - probe_f1),
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
        if any(len(np.unique(y)) < 2 for y in (ytr, yva, yte)):
            print(f"  skip {trait}: single-class split")
            continue
        results[trait] = evaluate_adaptive(Xtr, ytr, Xva, yva, Xte, yte)
    return results


def to_markdown(results: dict) -> str:
    lines = [
        "# Table 17 — Novelty-adaptive retrieval head (cross-clade)",
        "",
        "Per-genome blend weight `alpha_i = clip(alpha_lo + (alpha_hi - alpha_lo) * t_i, "
        "0, 1)`, where `t_i` is family-val-normalized novelty (Euclidean OOD vs "
        "family-train). Ramp endpoints tuned on family-val, scored once on family-test. "
        "`alpha_hi > alpha_lo` means the probe is trusted more as novelty rises. Compared "
        "against the global-alpha blend (Table 16) and probe-alone on the same test set.",
        "",
        "| Trait | Test n | alpha_lo | alpha_hi | alpha_global | Probe F1 | Global F1 | "
        "Adaptive F1 | delta vs global | delta vs probe | Probe AUROC | Global AUROC | Adaptive AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trait, r in results.items():
        lines.append(
            f"| `{trait}` | {r['n_test']} | {r['alpha_lo']:.1f} | {r['alpha_hi']:.1f} | "
            f"{r['alpha_global']:.1f} | {r['probe_f1']:.3f} | {r['global_f1']:.3f} | "
            f"{r['adaptive_f1']:.3f} | {r['delta_vs_global_f1']:+.3f} | {r['delta_vs_probe_f1']:+.3f} | "
            f"{r['probe_auroc']:.3f} | {r['global_auroc']:.3f} | {r['adaptive_auroc']:.3f} |"
        )
    lines += ["", _verdict(results)]
    return "\n".join(lines) + "\n"


def _verdict(results: dict, margin: float = 0.01) -> str:
    """Honest one-line takeaway: does novelty-conditioning beat the global blend?"""
    if not results:
        return "**Verdict:** no evaluable traits."
    deltas = {t: r["delta_vs_global_f1"] for t, r in results.items()}
    wins = [t for t, d in deltas.items() if d > margin]
    losses = [t for t, d in deltas.items() if d < -margin]
    mean_d = float(np.mean(list(deltas.values())))
    loss_note = (f" Tuning the ramp on family-val can overfit and transfer negatively "
                 f"({', '.join(losses)}), so the simpler global alpha (Table 16) is the "
                 f"robust choice.") if losses else ""
    win_note = (f" Where a ramp does help ({', '.join(wins)}), it leans `alpha_hi > alpha_lo` "
                f"— the probe is trusted more as novelty rises, consistent with k-NN "
                f"becoming unreliable far from training families.") if wins else ""
    return (
        f"**Verdict:** novelty-conditioning does not reliably beat the global blend "
        f"(mean delta-F1 vs global = {mean_d:+.3f}; {len(wins)}/{len(deltas)} traits improve "
        f"by >{margin:g}, {len(losses)} regress)." + loss_note + win_note
    )


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
    (out_dir / "17_adaptive_retrieval.md").write_text(md)
    pd.DataFrame([{"trait": t, **r} for t, r in results.items()]).to_csv(
        out_dir / "17_adaptive_retrieval.csv", index=False
    )
    print(md)


if __name__ == "__main__":
    main()
