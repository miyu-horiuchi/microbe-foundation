"""
multilabel_reformulation.py — Lever 3b: predict the *set*, not a collapsed single label.

Two label-ceiling traits (Table 24) are structured fields that were scored as single-class
multiclass, which destroys their signal:

  * metabolite_production: a per-compound ternary (True = produces, False = does not,
    None = untested). The correct formulation is a masked multi-label task — one binary
    head per compound, trained and scored only on the genomes actually tested for it.
  * cultivation_medium: a positive-only set of media IDs that support growth. The correct
    formulation is multi-label presence — one binary head per medium (listed = positive).

For each trait we build the per-label targets, fit a balanced one-vs-rest linear probe on
frozen ESM-2 features per label (species split), and report per-label AUROC/F1 plus
macro/micro aggregates over the well-supported labels. We compare against the single-label
collapse baseline (species-split macro-F1 from Table 20: metabolite_production 0.102,
cultivation_medium 0.278). A large jump shows the ceiling was a label-schema artifact: the
frozen representation carries the signal once the task is posed correctly.

CPU-only.

Usage:
    python paper/multilabel_reformulation.py
    python paper/multilabel_reformulation.py --max-labels 40
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_clade_diagnostic import _probe  # noqa: E402

# Single-label collapse baselines (species-split macro-F1, Table 20).
COLLAPSE_BASELINE = {"metabolite_production": 0.102, "cultivation_medium": 0.278}


def collect_metabolite(dicts, bids):
    """compound -> (positive bids, negative bids) from the per-compound ternary dicts."""
    pos, neg = defaultdict(list), defaultdict(list)
    for bid, d in zip(bids, dicts):
        if not isinstance(d, dict):
            continue
        for c, v in d.items():
            sv = str(v)
            if sv == "True":
                pos[c].append(bid)
            elif sv == "False":
                neg[c].append(bid)
    return pos, neg


def collect_medium(arrays, bids):
    """medium -> set(positive bids); plus the full labeled-genome list (negatives = rest)."""
    pres = defaultdict(set)
    all_bids = []
    for bid, arr in zip(bids, arrays):
        if arr is None:
            continue
        all_bids.append(bid)
        for m in list(arr):
            pres[str(m)].add(bid)
    return pres, all_bids


def eval_label(pos_bids, neg_bids, feats, row_map, split_map):
    """Species-split balanced probe for one binary label; returns metrics + raw test arrays."""
    df = pd.DataFrame([(b, 1) for b in pos_bids] + [(b, 0) for b in neg_bids],
                      columns=["bid", "y"])
    df = df[df["bid"].isin(row_map)]
    df["split"] = df["bid"].map(lambda b: split_map.get(b, "unknown"))
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    if tr["y"].nunique() < 2 or te["y"].nunique() < 2:
        return None
    Xtr = feats[[row_map[b] for b in tr["bid"]]]
    Xte = feats[[row_map[b] for b in te["bid"]]]
    prob = _probe(Xtr, tr["y"].to_numpy(), Xte)
    yte = te["y"].to_numpy()
    pred = (prob > 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(yte, prob)),
        "f1": float(f1_score(yte, pred, pos_label=1, average="binary", zero_division=0)),
        "n_pos_train": int(tr["y"].sum()), "n_pos_test": int(yte.sum()),
        "n_test": int(len(yte)), "yte": yte, "pred": pred,
    }


def aggregate(per_label: dict) -> dict:
    """Macro (mean over labels) and micro (pooled) metrics across evaluated labels."""
    if not per_label:
        return {"n_labels": 0}
    aurocs = [r["auroc"] for r in per_label.values()]
    f1s = [r["f1"] for r in per_label.values()]
    yt = np.concatenate([r["yte"] for r in per_label.values()])
    pr = np.concatenate([r["pred"] for r in per_label.values()])
    return {
        "n_labels": len(per_label),
        "macro_auroc": float(np.mean(aurocs)),
        "macro_f1": float(np.mean(f1s)),
        "micro_f1": float(f1_score(yt, pr, pos_label=1, average="binary", zero_division=0)),
        "median_auroc": float(np.median(aurocs)),
    }


def _run_trait(labels_pos, labels_neg_fn, feats, row_map, split_map,
               min_pos, max_labels):
    """Evaluate the best-supported labels (by positive count) for one trait."""
    ranked = sorted(labels_pos.items(), key=lambda kv: -len(kv[1]))
    per_label = {}
    for name, pos in ranked:
        if len(pos) < min_pos:
            continue
        neg = labels_neg_fn(name, pos)
        res = eval_label(pos, neg, feats, row_map, split_map)
        if res is not None:
            per_label[name] = res
        if len(per_label) >= max_labels:
            break
    return per_label


def run(features_path, splits_path, traits_path, min_pos=20, max_labels=60):
    data = np.load(features_path, allow_pickle=True)
    feats = data["features"]
    ids = [str(i) for i in data["bacdive_ids"]]
    row_map = {b: i for i, b in enumerate(ids)}
    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    sp = pd.read_parquet(splits_path)[["bacdive_id", "species_split"]]
    split_map = dict(zip(sp["bacdive_id"].astype(str), sp["species_split"]))

    results = {}

    # metabolite_production: masked binary per compound (negatives = tested-False genomes)
    m = tr[tr["metabolite_production"].notna()]
    mp_pos, mp_neg = collect_metabolite(m["metabolite_production"].tolist(), m["bid"].tolist())
    mp_labels = _run_trait(mp_pos, lambda name, pos: mp_neg.get(name, []),
                           feats, row_map, split_map, min_pos, max_labels)
    results["metabolite_production"] = {"per_label": mp_labels, **aggregate(mp_labels)}

    # cultivation_medium: multilabel presence per medium (negatives = labeled genomes not listing it)
    c = tr[tr["cultivation_medium"].notna()]
    med_pos, med_all = collect_medium(c["cultivation_medium"].tolist(), c["bid"].tolist())
    med_all_set = set(med_all)
    cm_labels = _run_trait(med_pos, lambda name, pos: list(med_all_set - set(pos)),
                           feats, row_map, split_map, max(min_pos, 50), max_labels)
    results["cultivation_medium"] = {"per_label": cm_labels, **aggregate(cm_labels)}
    return results


def to_markdown(results: dict, top_n=8) -> str:
    lines = [
        "# Table 25 — Multi-label reformulation of the structured ceiling traits",
        "",
        "Scored as a single collapsed label these traits look unsolvable (Table 20); posed "
        "correctly as multi-label they are not. Per trait: balanced one-vs-rest linear probes "
        "on frozen ESM-2, species split, over the well-supported labels. The single-label "
        "collapse baseline is the species-split macro-F1 from Table 20.",
        "",
        "| Trait | Formulation | #labels | Macro-AUROC | Macro-F1 | Micro-F1 | Collapse F1 (Table 20) |",
        "|---|:--|---:|---:|---:|---:|---:|",
    ]
    formul = {"metabolite_production": "masked binary / compound",
              "cultivation_medium": "presence / medium"}
    for trait, r in results.items():
        if not r.get("n_labels"):
            lines.append(f"| `{trait}` | {formul.get(trait,'')} | 0 | — | — | — | "
                         f"{COLLAPSE_BASELINE.get(trait, float('nan')):.3f} |")
            continue
        lines.append(
            f"| `{trait}` | {formul.get(trait,'')} | {r['n_labels']} | {r['macro_auroc']:.3f} | "
            f"{r['macro_f1']:.3f} | {r['micro_f1']:.3f} | {COLLAPSE_BASELINE.get(trait, float('nan')):.3f} |"
        )
    # per-label highlights
    for trait, r in results.items():
        pl = r.get("per_label", {})
        if not pl:
            continue
        top = sorted(pl.items(), key=lambda kv: -kv[1]["auroc"])[:top_n]
        lines += ["", f"## Best-predicted labels — `{trait}`", "",
                  "| Label | AUROC | F1 | n pos (train/test) |", "|---|---:|---:|---:|"]
        for name, res in top:
            lines.append(f"| `{name}` | {res['auroc']:.3f} | {res['f1']:.3f} | "
                         f"{res['n_pos_train']}/{res['n_pos_test']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--max-labels", type=int, default=60)
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    results = run(args.features, args.splits, args.traits_path,
                  min_pos=args.min_pos, max_labels=args.max_labels)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    md = to_markdown(results)
    (Path(args.out_dir) / "25_multilabel_reformulation.md").write_text(md)
    flat = [{"trait": t, "label": name, "auroc": res["auroc"], "f1": res["f1"],
             "n_pos_train": res["n_pos_train"], "n_pos_test": res["n_pos_test"],
             "n_test": res["n_test"]}
            for t, r in results.items() for name, res in r.get("per_label", {}).items()]
    pd.DataFrame(flat).to_csv(Path(args.out_dir) / "25_multilabel_reformulation.csv", index=False)
    print(md)
    print(f"wrote {args.out_dir}/25_multilabel_reformulation.md/.csv")


if __name__ == "__main__":
    main()
