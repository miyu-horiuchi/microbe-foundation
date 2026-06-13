"""
label_quality_audit.py — Lever 3: are the label-ceiling traits a model problem or a label problem?

Six traits are poor even in-distribution (Table 20, label-ceiling). This script asks why,
and fixes the ones that are fixable.

Part A (audit): for each ceiling trait, quantify the label pathology on the non-null
labels — number of classes, majority-class share, fraction of singleton classes,
normalized entropy, and whether the field is structured/multilabel — and name the
dominant failure (free-text explosion, structured/multilabel, geography, or extreme
imbalance).

Part B (fix + re-probe): for the two genuinely fixable categorical traits we apply a
principled consolidation —
  * oxygen_tolerance: 6 noisy classes (some with <100 examples) -> 3 canonical
    {aerobe, anaerobe, facultative};
  * isolation_source: 24k free-text strings -> ~8 ecological categories by keyword
    canonicalization —
and re-probe the in-distribution (species-split) macro-F1 before vs after with the same
balanced multiclass linear probe on frozen ESM-2 features. A rise shows the ceiling was
partly a label-schema artifact, not a representational limit.

CPU-only.

Usage:
    python paper/label_quality_audit.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CEILING_TRAITS = ["country", "isolation_source", "cell_shape", "oxygen_tolerance",
                  "metabolite_production", "cultivation_medium"]

OXYGEN_MAP = {
    "obligate_aerobe": "aerobe",
    "facultative_aerobe": "aerobe",
    "microaerophile": "aerobe",
    "obligate_anaerobe": "anaerobe",
    "aerotolerant": "anaerobe",
    "facultative_anaerobe": "facultative",
}

# Ordered (category, keyword) rules; first match wins. Order resolves overlaps
# (e.g. "human blood" -> human before water; "cheese" -> food before animal).
ISOLATION_RULES = [
    ("human/clinical", ["human", "patient", "blood", "clinical", "urine", "wound",
                         "feces", "faeces", "stool", "sputum", "skin", "nasal", "oral"]),
    ("food/dairy", ["food", "cheese", "milk", "dairy", "ferment", "meat", "yogurt",
                    "yoghurt", "wine", "beer", "sausage", "kimchi"]),
    ("plant", ["plant", "root", "leaf", "rhizosphere", "flower", "seed", "crop",
               "wood", "grass", "rice", "wheat", "fruit", "phyllosphere"]),
    ("soil", ["soil"]),
    ("sediment", ["sediment", "mud", "sludge"]),
    ("water/marine", ["water", "marine", "sea", "ocean", "lake", "river", "fresh",
                      "pond", "spring", "hydrotherm", "aquatic", "brine"]),
    ("animal", ["animal", "bovine", "chicken", "pig", "cattle", "fish", "insect",
                "rumen", "mouse", "cow", "poultry", "gut", "intestine", "bird",
                "shrimp", "mosquito", "tick"]),
]


def normalized_entropy(counts) -> float:
    """Shannon entropy of the class distribution, normalized to [0, 1]."""
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if len(c) <= 1:
        return 0.0
    p = c / c.sum()
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(c)))


def audit_metrics(values: pd.Series) -> dict:
    """Label-pathology metrics for one trait's non-null values."""
    s = values.dropna()
    structured = len(s) > 0 and isinstance(s.iloc[0], (dict, list, np.ndarray))
    if structured:
        return {"n_labeled": int(len(s)), "structured": True, "n_classes": None,
                "top1_share": None, "singleton_frac": None, "norm_entropy": None}
    vc = s.astype(str).value_counts()
    n = int(vc.sum())
    singleton = int((vc == 1).sum())
    return {
        "n_labeled": n,
        "structured": False,
        "n_classes": int(len(vc)),
        "top1_share": float(vc.iloc[0] / n),
        "singleton_frac": float(singleton / len(vc)),
        "norm_entropy": normalized_entropy(vc.values),
    }


def diagnose(m: dict) -> str:
    """Name the dominant label pathology from audit metrics."""
    if m["structured"]:
        return "structured/multilabel (collapsed to one label)"
    if m["n_classes"] > 0.2 * m["n_labeled"]:
        return "free-text explosion (sparse classes)"
    if m["singleton_frac"] > 0.4:
        return "long tail of singleton classes"
    if m["n_classes"] > 50 and m["top1_share"] < 0.5:
        return "many classes, no genomic basis (e.g. geography)"
    if m["top1_share"] > 0.8:
        return "extreme class imbalance"
    return "few classes, weak genomic signal"


def canon_isolation(text: str) -> str:
    """Map a free-text isolation source to a small ecological category."""
    t = str(text).lower()
    for cat, kws in ISOLATION_RULES:
        if any(kw in t for kw in kws):
            return cat
    return "other"


def clean_labels(trait: str, values: pd.Series) -> pd.Series:
    """Apply the consolidation map for a fixable categorical trait."""
    if trait == "oxygen_tolerance":
        return values.map(lambda v: OXYGEN_MAP.get(str(v)) if pd.notna(v) else None)
    if trait == "isolation_source":
        return values.map(lambda v: canon_isolation(v) if pd.notna(v) else None)
    raise ValueError(f"no cleaner for {trait}")


def _probe_multiclass(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    return clf.predict(scaler.transform(Xte))


def reprobe(labels: pd.Series, bids: pd.Series, feats, row_map, split_map, min_count=20):
    """Species-split balanced multiclass macro-F1 for a label series.

    Keeps only classes with >= min_count training examples (otherwise the raw free-text
    'before' condition is undefined); restricts test to those classes too.
    """
    df = pd.DataFrame({"bid": bids.astype(str), "y": labels})
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(str)
    df = df[df["bid"].isin(row_map)]
    df["split"] = df["bid"].map(lambda b: split_map.get(b, "unknown"))
    tr = df[df["split"] == "train"]
    keep = set(tr["y"].value_counts()[lambda c: c >= min_count].index)
    tr = tr[tr["y"].isin(keep)]
    te = df[(df["split"] == "test") & (df["y"].isin(keep))]
    if tr["y"].nunique() < 2 or len(te) < 10:
        return None
    Xtr = feats[[row_map[b] for b in tr["bid"]]]
    Xte = feats[[row_map[b] for b in te["bid"]]]
    pred = _probe_multiclass(Xtr, tr["y"].to_numpy(), Xte)
    return {
        "macro_f1": float(f1_score(te["y"].to_numpy(), pred, average="macro", zero_division=0)),
        "n_classes": int(tr["y"].nunique()),
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }


def run(features_path, splits_path, traits_path):
    data = np.load(features_path)
    feats = data["features"]
    ids = [str(i) for i in data["bacdive_ids"]]
    row_map = {b: i for i, b in enumerate(ids)}
    tr = pd.read_parquet(traits_path)
    tr["bid"] = tr["bacdive_id"].astype(str)
    sp = pd.read_parquet(splits_path)[["bacdive_id", "species_split"]]
    split_map = dict(zip(sp["bacdive_id"].astype(str), sp["species_split"]))

    audit = {t: {**audit_metrics(tr[t]), "trait": t,
                 "diagnosis": diagnose(audit_metrics(tr[t]))}
             for t in CEILING_TRAITS if t in tr.columns}

    fixes = {}
    for trait in ["oxygen_tolerance", "isolation_source"]:
        if trait not in tr.columns:
            continue
        before = reprobe(tr[trait], tr["bid"], feats, row_map, split_map)
        after = reprobe(clean_labels(trait, tr[trait]), tr["bid"], feats, row_map, split_map)
        fixes[trait] = {"before": before, "after": after}
    return audit, fixes


def to_markdown(audit, fixes) -> str:
    lines = [
        "# Table 24 — Label-quality audit of the ceiling traits",
        "",
        "## A. Why each label-ceiling trait is hard",
        "",
        "| Trait | n labeled | n classes | Top-1 share | Singleton frac | Norm. entropy | Dominant pathology |",
        "|---|---:|---:|---:|---:|---:|:--|",
    ]
    def f(x, p=3):
        return "—" if x is None else f"{x:.{p}f}"
    for t, m in audit.items():
        nclasses = "structured" if m["structured"] else str(m["n_classes"])
        lines.append(
            f"| `{t}` | {m['n_labeled']} | {nclasses} | {f(m['top1_share'])} | "
            f"{f(m['singleton_frac'])} | {f(m['norm_entropy'])} | {m['diagnosis']} |"
        )
    lines += [
        "",
        "## B. Fixing the fixable: consolidate the label schema, re-probe in-distribution",
        "",
        "Same balanced multiclass linear probe on frozen ESM-2, species-split macro-F1, "
        "before vs after schema consolidation. A rise means the ceiling was partly a "
        "label artifact, not a representational wall.",
        "",
        "| Trait | Before: classes → F1 | After: classes → F1 | ΔF1 |",
        "|---|---:|---:|---:|",
    ]
    for t, fx in fixes.items():
        b, a = fx["before"], fx["after"]
        if not b or not a:
            lines.append(f"| `{t}` | n/a | n/a | n/a |")
            continue
        d = a["macro_f1"] - b["macro_f1"]
        lines.append(
            f"| `{t}` | {b['n_classes']} → {b['macro_f1']:.3f} | "
            f"{a['n_classes']} → {a['macro_f1']:.3f} | {d:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/esm2_features.npz")
    ap.add_argument("--splits", default="data/splits.parquet")
    ap.add_argument("--traits-path", default="data/traits.parquet")
    ap.add_argument("--out-dir", default="paper/tables")
    args = ap.parse_args()

    audit, fixes = run(args.features, args.splits, args.traits_path)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    md = to_markdown(audit, fixes)
    (Path(args.out_dir) / "24_label_quality_audit.md").write_text(md)
    pd.DataFrame(list(audit.values())).to_csv(Path(args.out_dir) / "24_label_audit.csv", index=False)
    print(md)
    print(f"wrote {args.out_dir}/24_label_quality_audit.md")


if __name__ == "__main__":
    main()
