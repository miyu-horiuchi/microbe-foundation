"""
vocab.py — discover class vocabularies for multilabel and regression heads.

The Phase 0 schema declared `classes_source` for several heads (cultivation
medium IDs from MediaDive, FAMEs from BacDive frequency, carbon substrates
from Madin, etc.) but the actual class vocabularies must be derived from the
data. This script discovers them from data/traits.parquet and writes a
locked spec to data/vocabularies.json.

The output is the single source of truth for Phase 2 model heads:
    - Each multilabel head learns one logit per item in its vocab.
    - Each regression-vector head outputs one scalar per item in its vocab.
    - Scalar multiclass heads keep their schema-enumerated vocab.

Reads:  data/traits.parquet
Writes: data/vocabularies.json

Usage:
    python vocab.py
    python vocab.py --top-media 200 --top-fame 30 --top-carbon 80 --top-amr 30 \\
                    --top-metabolite 50 --top-isolation 20 --top-country 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas pyarrow")


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_IN = DATA_DIR / "traits.parquet"
DEFAULT_OUT = DATA_DIR / "vocabularies.json"


def _as_iterable_strings(v) -> list[str] | None:
    """
    Parquet roundtrip turns Python lists into numpy ndarrays; both should be
    treated as iterables here. Returns None if v isn't a usable sequence.
    """
    if v is None:
        return None
    if isinstance(v, str) or isinstance(v, dict):
        return None
    if hasattr(v, "__iter__"):
        items = [str(x) for x in v if x is not None]
        return items if items else None
    return None


def discover_dict_keys(series: pd.Series, top_n: int) -> dict:
    """For columns whose cells are dict[item -> value]. Count item frequency across strains."""
    counts: Counter[str] = Counter()
    n_labeled = 0
    for v in series:
        if isinstance(v, dict) and v:
            real_keys = [k for k, val in v.items() if val is not None]
            if real_keys:
                n_labeled += 1
                counts.update(real_keys)
    items = [
        {"value": k, "n_strains": c, "frac": round(c / max(n_labeled, 1), 4)}
        for k, c in counts.most_common(top_n)
    ]
    return {
        "size": len(items),
        "n_labeled_strains": n_labeled,
        "n_unique_in_data": len(counts),
        "items": items,
    }


def discover_list_items(series: pd.Series, top_n: int) -> dict:
    """For columns whose cells are list[str] or ndarray (e.g., cultivation_medium)."""
    counts: Counter[str] = Counter()
    n_labeled = 0
    for v in series:
        items = _as_iterable_strings(v)
        if items:
            n_labeled += 1
            counts.update(items)
    items_out = [
        {"value": k, "n_strains": c, "frac": round(c / max(n_labeled, 1), 4)}
        for k, c in counts.most_common(top_n)
    ]
    return {
        "size": len(items_out),
        "n_labeled_strains": n_labeled,
        "n_unique_in_data": len(counts),
        "items": items_out,
    }


def discover_scalar(series: pd.Series, top_n: int | None = None) -> dict:
    """For categorical scalar columns. Returns value-count distribution."""
    vc = series.value_counts(dropna=True)
    if top_n:
        vc = vc.head(top_n)
    n_labeled = int(series.notna().sum())
    items = [
        {"value": str(k), "n_strains": int(c), "frac": round(c / max(n_labeled, 1), 4)}
        for k, c in vc.items()
    ]
    return {
        "size": len(items),
        "n_labeled_strains": n_labeled,
        "n_unique_in_data": int(series.dropna().nunique()),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_IN, help="Input traits parquet")
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUT, help="Output vocabularies json")
    parser.add_argument("--top-media", type=int, default=200, help="Max MediaDive media in vocab")
    parser.add_argument("--top-fame", type=int, default=30, help="Max FAMEs in regression vector")
    parser.add_argument("--top-carbon", type=int, default=80, help="Max carbon substrates")
    parser.add_argument("--top-amr", type=int, default=30, help="Max antibiotics in AMR vocab")
    parser.add_argument("--top-metabolite", type=int, default=50, help="Max metabolites in production vocab")
    parser.add_argument("--top-isolation", type=int, default=20, help="Max isolation source classes")
    parser.add_argument("--top-country", type=int, default=100, help="Max countries")
    args = parser.parse_args()

    if not args.input_path.exists():
        sys.exit(f"Input file not found: {args.input_path}. Run parse_bacdive.py first.")

    df = pd.read_parquet(args.input_path)
    print(f"Loaded {len(df):,} strains from {args.input_path}\n")

    out: dict = {
        "schema_version": "0.1.0",
        "n_strains_basis": len(df),
        "vocabularies": {},
    }

    # === Multilabel / regression-vector heads (data-derived vocab) ===
    print("Multilabel / regression-vector heads (data-derived vocab):")
    for trait, kind, top_n in [
        ("cultivation_medium", "list", args.top_media),
        ("carbon_utilization", "dict", args.top_carbon),
        ("metabolite_production", "dict", args.top_metabolite),
        ("amr_phenotype", "dict", args.top_amr),
        ("fatty_acid_profile", "dict", args.top_fame),
    ]:
        if trait not in df.columns:
            continue
        if kind == "list":
            v = discover_list_items(df[trait], top_n)
        else:
            v = discover_dict_keys(df[trait], top_n)
        out["vocabularies"][trait] = v
        print(
            f"  {trait:<24} size={v['size']:>4}  "
            f"n_unique_seen={v['n_unique_in_data']:>5}  "
            f"n_labeled={v['n_labeled_strains']:>5,}"
        )

    # === Categorical scalar heads (data-derived vocab — free-text fields) ===
    print("\nCategorical scalar heads (data-derived vocab):")
    for trait, top_n in [
        ("isolation_source", args.top_isolation),
        ("country", args.top_country),
    ]:
        if trait not in df.columns:
            continue
        v = discover_scalar(df[trait], top_n)
        out["vocabularies"][trait] = v
        print(
            f"  {trait:<24} size={v['size']:>4}  "
            f"n_unique_seen={v['n_unique_in_data']:>5}  "
            f"n_labeled={v['n_labeled_strains']:>5,}"
        )

    # === Categorical scalar heads (enumerated in schema) — report distribution for sanity ===
    print("\nEnumerated heads (vocab fixed in schema — reporting actual distribution):")
    for trait in (
        "gram_stain",
        "cell_shape",
        "oxygen_tolerance",
        "halophily",
        "temperature_class",
        "ph_class",
        "biosafety_level",
    ):
        if trait not in df.columns:
            continue
        v = discover_scalar(df[trait])
        out["vocabularies"][trait] = v
        labels = [f"{it['value']}={it['n_strains']}" for it in v["items"][:5]]
        print(
            f"  {trait:<24} size={v['size']:>4}  "
            f"n_labeled={v['n_labeled_strains']:>5,}  "
            f"top: {' '.join(labels)}"
        )

    # === Binary heads — just report positive rate ===
    print("\nBinary heads (positive rate):")
    for trait in ("motility", "sporulation", "pigmentation", "catalase",
                  "cytochrome_oxidase", "pathogenicity_human", "pathogenicity_animal"):
        if trait not in df.columns:
            continue
        n_labeled = int(df[trait].notna().sum())
        n_pos = int(df[trait].fillna(False).sum())
        rate = n_pos / max(n_labeled, 1)
        out["vocabularies"][trait] = {
            "size": 2,
            "n_labeled_strains": n_labeled,
            "n_positive": n_pos,
            "positive_rate": round(rate, 4),
        }
        print(f"  {trait:<24} n_labeled={n_labeled:>5,}  pos={n_pos:>5,}  pos_rate={rate:.2%}")

    args.output_path.parent.mkdir(exist_ok=True)
    args.output_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote vocabularies to {args.output_path}")


if __name__ == "__main__":
    main()
