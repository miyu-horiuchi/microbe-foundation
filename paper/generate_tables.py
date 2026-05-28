"""
generate_tables.py — refresh auto-generated tables in paper/tables/ from the
current data files. Run after fetch/parse/splits/vocab updates.

Tables generated:
    tables/01_trait_inventory.md        from trait_schema.json
    tables/02_label_coverage.md         from data/traits.parquet
    tables/03_split_stats.md            from data/splits.parquet
    tables/04_vocabulary_sizes.md       from data/vocabularies.json
    tables/05_results_template.md       template populated when results land

Usage:
    python paper/generate_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("requires: pip install pandas pyarrow")


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCHEMA_PATH = ROOT / "trait_schema.json"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
TRAITS_PATH = DATA_DIR / "traits.parquet"
SPLITS_PATH = DATA_DIR / "splits.parquet"
TABLES_DIR = Path(__file__).parent / "tables"


def write(path: Path, content: str) -> None:
    path.write_text(content)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(content):,} bytes)")


# =============================================================================
# Table 1: Trait inventory
# =============================================================================


def gen_trait_inventory() -> str:
    schema = json.loads(SCHEMA_PATH.read_text())
    out = ["# Table 1 — Trait Inventory", ""]
    out.append("Schema version: " + schema["schema_version"])
    out.append("")
    out.append("| Block | Trait | Head | Size | Est. labeled strains |")
    out.append("|---|---|---|---:|---:|")
    for t in schema["traits"]:
        block = t["block"]
        name = t["name"]
        head = t["head"]
        size = t.get("n_outputs") or (len(t["classes"]) if t.get("classes") else 1)
        labels = t.get("estimated_label_count", 0)
        out.append(f"| {block} | `{name}` | {head} | {size} | {labels:,} |")
    out.append("")
    out.append(f"_Total: {len(schema['traits'])} prediction heads across "
               f"{len(set(t['block'] for t in schema['traits']))} blocks._")
    return "\n".join(out) + "\n"


# =============================================================================
# Table 2: Per-trait label coverage on the current parquet
# =============================================================================


def gen_label_coverage() -> str:
    if not TRAITS_PATH.exists():
        return "# Table 2 — Label Coverage\n\n_(traits.parquet not yet generated — run parse_bacdive.py)_\n"
    df = pd.read_parquet(TRAITS_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    n = len(df)
    out = ["# Table 2 — Label Coverage", ""]
    out.append(f"Source: {len(df):,} strains parsed from BacDive.")
    out.append("")
    out.append("| Trait | Type | Labeled | Coverage |")
    out.append("|---|---|---:|---:|")
    for t in schema["traits"]:
        name = t["name"]
        if name not in df.columns:
            continue
        col = df[name]
        # For dict/list/ndarray columns, .notna() doesn't tell us if the container is non-empty
        if t["head"] in ("multilabel", "regression_vector"):
            def _nonempty(v):
                if v is None:
                    return False
                if isinstance(v, dict):
                    return len(v) > 0
                if isinstance(v, str):
                    return False
                if hasattr(v, "__len__"):
                    return len(v) > 0
                return False
            labeled = sum(1 for v in col if _nonempty(v))
        else:
            labeled = int(col.notna().sum())
        pct = 100 * labeled / max(n, 1)
        out.append(f"| `{name}` | {t['head']} | {labeled:,} | {pct:.1f}% |")
    return "\n".join(out) + "\n"


# =============================================================================
# Table 3: Split statistics
# =============================================================================


def gen_split_stats() -> str:
    if not SPLITS_PATH.exists():
        return "# Table 3 — Split Statistics\n\n_(splits.parquet not yet generated — run splits.py)_\n"
    df = pd.read_parquet(SPLITS_PATH)
    out = ["# Table 3 — Phylogeny-Aware Split Statistics", ""]
    out.append(f"Total strains: {len(df):,}")
    out.append("")
    out.append("| Level | Unique groups | Train | Val | Test | Unknown |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for level in ("family", "genus", "species"):
        col = f"{level}_split"
        if col not in df.columns:
            continue
        counts = df[col].value_counts().to_dict()
        n_groups = df[level].dropna().nunique()
        train = counts.get("train", 0)
        val = counts.get("val", 0)
        test = counts.get("test", 0)
        unknown = counts.get("unknown", 0)
        out.append(
            f"| {level} | {n_groups:,} | "
            f"{train:,} ({100*train/len(df):.1f}%) | "
            f"{val:,} ({100*val/len(df):.1f}%) | "
            f"{test:,} ({100*test/len(df):.1f}%) | "
            f"{unknown:,} |"
        )
    out.append("")
    out.append("**Family-held-out** is microbe-foundation's primary evaluation protocol "
               "(strictly harder than BacBench's genus-held-out).")
    return "\n".join(out) + "\n"


# =============================================================================
# Table 4: Vocabulary sizes
# =============================================================================


def gen_vocabulary_sizes() -> str:
    if not VOCAB_PATH.exists():
        return "# Table 4 — Vocabulary Sizes\n\n_(vocabularies.json not yet generated — run vocab.py)_\n"
    vocab = json.loads(VOCAB_PATH.read_text())
    out = ["# Table 4 — Discovered Vocabulary Sizes", ""]
    out.append(f"Basis: {vocab.get('n_strains_basis', '?'):,} strains.")
    out.append("")
    out.append("| Trait | Vocab size | Unique seen in data | Labeled strains |")
    out.append("|---|---:|---:|---:|")
    for name, info in vocab["vocabularies"].items():
        size = info.get("size", "—")
        seen = info.get("n_unique_in_data", "—")
        labeled = info.get("n_labeled_strains", "—")
        size_s = f"{size:,}" if isinstance(size, int) else size
        seen_s = f"{seen:,}" if isinstance(seen, int) else seen
        labeled_s = f"{labeled:,}" if isinstance(labeled, int) else labeled
        out.append(f"| `{name}` | {size_s} | {seen_s} | {labeled_s} |")
    return "\n".join(out) + "\n"


# =============================================================================
# Table 5: Results template (filled by model.py output)
# =============================================================================


def gen_results_template() -> str:
    schema = json.loads(SCHEMA_PATH.read_text())
    out = ["# Table 5 — Per-Head Results (template)", ""]
    out.append("Fill from `model.py --features <X.npz> --split-level family` test metrics.")
    out.append("")
    out.append("| Trait | Metric | Majority | KO + MLP | ESM-2 pool | Bacformer | microbe-foundation |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for t in schema["traits"]:
        name = t["name"]
        head = t["head"]
        if head == "binary":
            metric = "ACC"
        elif head == "multiclass":
            metric = "ACC"
        elif head == "multilabel":
            metric = "F1"
        elif head == "regression_vector":
            metric = "RMSE"
        else:
            metric = "?"
        out.append(f"| `{name}` | {metric} | — | — | — | — | — |")
    return "\n".join(out) + "\n"


def main() -> None:
    TABLES_DIR.mkdir(exist_ok=True)
    print(f"Generating tables in {TABLES_DIR.relative_to(ROOT)}/ ...")
    write(TABLES_DIR / "01_trait_inventory.md", gen_trait_inventory())
    write(TABLES_DIR / "02_label_coverage.md", gen_label_coverage())
    write(TABLES_DIR / "03_split_stats.md", gen_split_stats())
    write(TABLES_DIR / "04_vocabulary_sizes.md", gen_vocabulary_sizes())
    write(TABLES_DIR / "05_results_template.md", gen_results_template())
    print("\ndone.")


if __name__ == "__main__":
    main()
