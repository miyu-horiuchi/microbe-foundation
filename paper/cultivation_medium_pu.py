"""
cultivation_medium_pu.py — Lever 3c: turn cultivation_medium's ranking into usable F1.

The multilabel reformulation (Table 25) showed cultivation_medium is highly *rankable*
(macro-AUROC 0.891) but its thresholded macro-F1 stays near the single-label collapse.
Two things suppress F1: (i) every medium is rare, so a fixed 0.5 threshold is wrong, and
(ii) the labels are positive-only (a genome's unlisted media are *unlabeled*, not known
negatives), so naive precision counts true-but-untested positives as false positives.

This script fixes both, per medium:
  1. Per-label threshold tuning: pick the decision threshold on a validation split that
     maximizes F1, converting good ranking into good thresholded predictions.
  2. Elkan--Noto PU correction (SCAR): estimate the labeling propensity c = P(labeled |
     positive) from held-out positives, then correct precision/recall by treating each
     unlabeled example as positive with its estimated posterior probability instead of a
     hard negative.

We report the macro/micro-F1 progression naive@0.5 -> tuned -> PU-adjusted on the species
test split, over the well-supported media. CPU-only.

Usage:
    python paper/cultivation_medium_pu.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLLAPSE_BASELINE_F1 = 0.278  # single-label collapse (Table 20)
MULTILABEL_05_MACRO_F1 = 0.289  # naive multilabel @0.5 (Table 25)


def en_conditional_pos(p_s: np.ndarray, c: float) -> np.ndarray:
    """Elkan-Noto P(y=1 | x, unlabeled) for unlabeled examples (SCAR).

    From p(y=1|x) = p(s=1|x)/c:
        P(y=1 | s=0, x) = p_s (1 - c) / (c (1 - p_s)), clipped to [0, 1].
    """
    p_s = np.clip(np.asarray(p_s, dtype=float), 0.0, 1.0 - 1e-9)
    c = float(min(max(c, 1e-6), 1.0))
    q = p_s * (1.0 - c) / (c * (1.0 - p_s))
    return np.clip(q, 0.0, 1.0)


def f1_from_pr(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def naive_prf(s: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Precision/recall/F1 treating labeled (s=1) as the only positives."""
    tp = float(np.sum(pred & (s == 1)))
    fp = float(np.sum(pred & (s == 0)))
    fn = float(np.sum((~pred) & (s == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, f1_from_pr(prec, rec)


def pu_prf(s: np.ndarray, p: np.ndarray, pred: np.ndarray, c: float) -> tuple[float, float, float]:
    """Elkan-Noto-corrected precision/recall/F1 for positive-unlabeled test data.

    Labeled positives count as true positives; each unlabeled example contributes its
    estimated posterior q = P(y=1|x, unlabeled) instead of being a hard negative.
    """
    s = np.asarray(s)
    pred = np.asarray(pred, dtype=bool)
    unl = s == 0
    q = np.zeros(len(s))
    q[unl] = en_conditional_pos(p[unl], c)

    tp = float(np.sum(pred & (s == 1)) + np.sum(q[pred & unl]))
    fp = float(np.sum((1.0 - q[pred & unl])))
    fn = float(np.sum((~pred) & (s == 1)) + np.sum(q[(~pred) & unl]))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, f1_from_pr(prec, rec)


def tune_threshold(s: np.ndarray, p: np.ndarray) -> float:
    """Threshold on validation scores that maximizes naive (labeled) F1."""
    s = np.asarray(s)
    p = np.asarray(p, dtype=float)
    cands = np.unique(np.concatenate([[0.0], p, [1.0]]))
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        _, _, f1 = naive_prf(s, p >= t)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _fit_predict(Xtr, ytr, Xval, Xte):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    p_val = clf.predict_proba(scaler.transform(Xval))[:, 1]
    p_te = clf.predict_proba(scaler.transform(Xte))[:, 1]
    return p_val, p_te


def eval_medium(pos_bids, all_bids, feats, row_map, split_map, min_pos=50):
    """Per-medium naive@0.5, tuned, and PU-adjusted F1 on the species test split."""
    pos = set(pos_bids)
    df = pd.DataFrame({"bid": list(all_bids)})
    df = df[df["bid"].isin(row_map)]
    df["s"] = df["bid"].isin(pos).astype(int)
    df["split"] = df["bid"].map(lambda b: split_map.get(b, "unknown"))
    tr, va, te = (df[df["split"] == k] for k in ("train", "val", "test"))
    if tr["s"].sum() < min_pos or va["s"].sum() < 3 or te["s"].sum() < 3:
        return None
    if tr["s"].nunique() < 2:
        return None
    Xtr = feats[[row_map[b] for b in tr["bid"]]]
    Xva = feats[[row_map[b] for b in va["bid"]]]
    Xte = feats[[row_map[b] for b in te["bid"]]]
    p_val, p_te = _fit_predict(Xtr, tr["s"].to_numpy(), Xva, Xte)

    c = float(np.clip(np.mean(p_val[va["s"].to_numpy() == 1]), 1e-3, 1.0))
    t_star = tune_threshold(va["s"].to_numpy(), p_val)
    s_te = te["s"].to_numpy()

    _, _, f1_naive = naive_prf(s_te, p_te >= 0.5)
    _, _, f1_tuned = naive_prf(s_te, p_te >= t_star)
    pu_p, pu_r, f1_pu = pu_prf(s_te, p_te, p_te >= t_star, c)
    return {
        "auroc": float(roc_auc_score(s_te, p_te)),
        "f1_naive": f1_naive, "f1_tuned": f1_tuned, "f1_pu": f1_pu,
        "pu_precision": pu_p, "pu_recall": pu_r,
        "c": c, "threshold": t_star,
        "n_pos_test": int(s_te.sum()), "n_test": int(len(s_te)),
    }


def collect_medium(arrays, bids):
    pres = defaultdict(set)
    all_bids = []
    for bid, arr in zip(bids, arrays):
        if arr is None:
            continue
        all_bids.append(bid)
        for m in list(arr):
            pres[str(m)].add(bid)
    return pres, all_bids


def macro(per_label, key):
    return float(np.mean([r[key] for r in per_label.values()])) if per_label else 0.0


def run(features_path, splits_path, traits_path, min_pos=50, max_labels=60):
    data = np.load(features_path, allow_pickle=True)
    feats = data["features"]
    ids = [str(i) for i in data["bacdive_ids"]]
    row_map = {b: i for i, b in enumerate(ids)}
    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    sp = pd.read_parquet(splits_path)[["bacdive_id", "species_split"]]
    split_map = dict(zip(sp["bacdive_id"].astype(str), sp["species_split"]))

    c = tr[tr["cultivation_medium"].notna()]
    pres, all_bids = collect_medium(c["cultivation_medium"].tolist(), c["bid"].tolist())
    ranked = sorted(pres.items(), key=lambda kv: -len(kv[1]))
    per_label = {}
    for name, pos in ranked:
        if len(pos) < min_pos:
            continue
        res = eval_medium(pos, all_bids, feats, row_map, split_map, min_pos=min_pos)
        if res is not None:
            per_label[name] = res
        if len(per_label) >= max_labels:
            break
    return per_label


def to_markdown(per_label: dict) -> str:
    lines = [
        "# Table 26 — cultivation_medium: turning rank quality into usable predictions",
        "",
        "Per-medium F1 on the species test split as we (1) tune the decision threshold on "
        "validation and (2) apply an Elkan--Noto positive-unlabeled correction. AUROC is "
        "unchanged by thresholding and shown for reference; `c` is the estimated labeling "
        "propensity P(listed | grows), so low `c` means many true-but-unlisted media.",
        "",
        f"Single-label collapse macro-F1 (Table 20): {COLLAPSE_BASELINE_F1:.3f}. "
        f"Naive multilabel @0.5 macro-F1 (Table 25): {MULTILABEL_05_MACRO_F1:.3f}.",
        "",
        "| Metric (macro over media) | Value |",
        "|---|---:|",
        f"| #media | {len(per_label)} |",
        f"| AUROC | {macro(per_label, 'auroc'):.3f} |",
        f"| F1 @0.5 (naive) | {macro(per_label, 'f1_naive'):.3f} |",
        f"| F1 + threshold tuning | {macro(per_label, 'f1_tuned'):.3f} |",
        f"| F1 + tuning + PU correction | {macro(per_label, 'f1_pu'):.3f} |",
        f"| mean labeling propensity c | {macro(per_label, 'c'):.3f} |",
        "",
        "## Best-recovered media (by PU-adjusted F1)",
        "",
        "| Medium | AUROC | F1@0.5 | F1 tuned | F1 PU | c | n pos (test) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    top = sorted(per_label.items(), key=lambda kv: -kv[1]["f1_pu"])[:10]
    for name, r in top:
        lines.append(f"| `{name}` | {r['auroc']:.3f} | {r['f1_naive']:.3f} | "
                     f"{r['f1_tuned']:.3f} | {r['f1_pu']:.3f} | {r['c']:.3f} | {r['n_pos_test']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--min-pos", type=int, default=50)
    ap.add_argument("--max-labels", type=int, default=60)
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    per_label = run(args.features, args.splits, args.traits_path,
                    min_pos=args.min_pos, max_labels=args.max_labels)
    if not per_label:
        raise SystemExit("no media met support threshold")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    md = to_markdown(per_label)
    (Path(args.out_dir) / "26_cultivation_medium_pu.md").write_text(md)
    flat = [{"medium": n, **{k: v for k, v in r.items()}} for n, r in per_label.items()]
    pd.DataFrame(flat).to_csv(Path(args.out_dir) / "26_cultivation_medium_pu.csv", index=False)
    print(md)
    print(f"wrote {args.out_dir}/26_cultivation_medium_pu.md/.csv")


if __name__ == "__main__":
    main()
