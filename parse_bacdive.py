"""
parse_bacdive.py — convert raw BacDive JSONL to a typed traits parquet.

Reads:  data/bacdive_raw.jsonl  (output of fetch_bacdive.py)
Writes: data/traits.parquet     (one row per strain, one column per v1 trait)

Each trait is extracted by a dedicated function that handles BacDive's
dict-or-list-of-dicts polymorphism, value normalization, and masked-NaN for
missing values. Multilabel and regression-vector traits are stored as
object-dtype columns (lists or dicts). Phase 1.5 can pivot to wide format
once class vocabularies are finalized.

Usage:
    python parse_bacdive.py
    python parse_bacdive.py --in data/bacdive_raw.jsonl --out data/traits.parquet
    python parse_bacdive.py --limit 1000   # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas pyarrow")


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_IN = DATA_DIR / "bacdive_raw.jsonl"
DEFAULT_OUT = DATA_DIR / "traits.parquet"


# =============================================================================
# Generic helpers
# =============================================================================


def as_list(x: Any) -> list:
    """BacDive fields are often dict-or-list-of-dicts. Normalize to a list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


TRUE_TOKENS = {"yes", "true", "positive", "+", "1", "growing", "growth"}
FALSE_TOKENS = {"no", "false", "negative", "-", "0", "not growing", "no growth"}


def norm_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v) if v in (0, 1) else None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in TRUE_TOKENS:
            return True
        if s in FALSE_TOKENS:
            return False
    return None


def majority_bool(values: Iterable[Any]) -> bool | None:
    """Take majority vote across multiple refs; ties resolve True (more conservative for positive traits)."""
    bools = [b for b in (norm_bool(v) for v in values) if b is not None]
    if not bools:
        return None
    return sum(bools) >= len(bools) / 2


def parse_numeric(v: Any) -> float | None:
    """Parse '25', '20-30' (midpoint), '~25', '<5', '>80', '25.5' to a float."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().replace(",", ".").replace("~", "").replace("ca.", "").strip()
    # Range like "20-30"
    m = re.match(r"^[<>]?\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    # Single number, possibly with < or >
    m = re.match(r"^[<>]?\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return None


# =============================================================================
# Per-trait extractors
# Return value:
#   - binary traits  → bool | None
#   - multiclass     → str | None
#   - multilabel     → dict[str, bool|str] | list[str] | None
#   - regression vec → dict[str, float] | None
# =============================================================================


def x_taxonomy(rec: dict) -> dict[str, str | None]:
    """Extract taxonomy + type-strain flag from the name/tax section."""
    nt = rec.get("Name and taxonomic classification", {}) or {}
    if not isinstance(nt, dict):
        return {k: None for k in ("domain", "phylum", "class", "order", "family", "genus", "species", "type_strain")}
    # Prefer the LPSN-curated values when present, else the legacy fields
    lpsn = nt.get("LPSN", {}) if isinstance(nt.get("LPSN"), dict) else {}
    out: dict[str, Any] = {}
    for rank in ("domain", "phylum", "class", "order", "family", "genus", "species"):
        out[rank] = lpsn.get(rank) or nt.get(rank)
    ts = nt.get("type strain")
    out["type_strain"] = norm_bool(ts) if ts is not None else None
    return out


def x_gram_stain(rec: dict) -> str | None:
    morph = rec.get("Morphology", {}) or {}
    if not isinstance(morph, dict):
        return None
    for cm in as_list(morph.get("cell morphology")):
        if not isinstance(cm, dict):
            continue
        v = cm.get("gram stain")
        if not v:
            continue
        s = str(v).strip().lower()
        if "variable" in s or "indetermin" in s:
            return "variable"
        if "negative" in s or s == "-":
            return "negative"
        if "positive" in s or s == "+":
            return "positive"
    return None


def x_cell_shape(rec: dict) -> str | None:
    morph = rec.get("Morphology", {}) or {}
    if not isinstance(morph, dict):
        return None
    for cm in as_list(morph.get("cell morphology")):
        if not isinstance(cm, dict):
            continue
        v = cm.get("cell shape")
        if not v:
            continue
        s = str(v).strip().lower()
        if "coc" in s:
            return "coccus"
        if "rod" in s or "bacill" in s:
            return "rod"
        if "vibrio" in s or "curved" in s:
            return "vibrio"
        if "spir" in s:
            return "spiral"
        if "fila" in s:
            return "filament"
        if "pleomorphic" in s:
            return "pleomorphic"
        return "other"
    return None


def x_motility(rec: dict) -> bool | None:
    morph = rec.get("Morphology", {}) or {}
    if not isinstance(morph, dict):
        return None
    values = []
    for cm in as_list(morph.get("cell morphology")):
        if isinstance(cm, dict) and cm.get("motility") is not None:
            values.append(cm.get("motility"))
    return majority_bool(values)


def x_sporulation(rec: dict) -> bool | None:
    values = []
    morph = rec.get("Morphology", {}) or {}
    if isinstance(morph, dict):
        for mc in as_list(morph.get("multicellular morphology")):
            if isinstance(mc, dict) and mc.get("spore formation") is not None:
                values.append(mc.get("spore formation"))
    pm = rec.get("Physiology and metabolism", {}) or {}
    if isinstance(pm, dict):
        for sp in as_list(pm.get("spore formation")):
            if isinstance(sp, dict) and sp.get("spore formation") is not None:
                values.append(sp.get("spore formation"))
            elif sp is not None and not isinstance(sp, dict):
                values.append(sp)
    return majority_bool(values)


def x_pigmentation(rec: dict) -> bool | None:
    morph = rec.get("Morphology", {}) or {}
    if not isinstance(morph, dict):
        return None
    pigs = as_list(morph.get("pigmentation"))
    if not pigs:
        return None
    productions = [p.get("production") for p in pigs if isinstance(p, dict)]
    if not any(productions):
        return None
    return majority_bool(productions)


def x_oxygen_tolerance(rec: dict) -> str | None:
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    for ot in as_list(pm.get("oxygen tolerance")):
        if not isinstance(ot, dict):
            continue
        v = ot.get("oxygen tolerance")
        if not v:
            continue
        s = str(v).strip().lower()
        # Map to canonical classes; preserve specificity where possible
        if "obligate aerobe" in s:
            return "obligate_aerobe"
        if "facultative aerobe" in s:
            return "facultative_aerobe"
        if "microaerophil" in s:
            return "microaerophile"
        if "aerotolerant" in s:
            return "aerotolerant"
        if "facultative anaerobe" in s or "facultatively anaerobic" in s:
            return "facultative_anaerobe"
        if "obligate anaerobe" in s or "strictly anaerobic" in s:
            return "obligate_anaerobe"
        if "aerobe" in s or "aerobic" in s:
            return "obligate_aerobe"  # default for unqualified "aerobe"
        if "anaerobe" in s or "anaerobic" in s:
            return "obligate_anaerobe"
    return None


def _enzyme_activity(rec: dict, name_substring: str) -> bool | None:
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    values = []
    for e in as_list(pm.get("enzymes")):
        if not isinstance(e, dict):
            continue
        nm = str(e.get("value", "")).lower()
        if name_substring in nm:
            values.append(e.get("activity"))
    return majority_bool(values)


def x_catalase(rec: dict) -> bool | None:
    return _enzyme_activity(rec, "catalase")


def x_cytochrome_oxidase(rec: dict) -> bool | None:
    # Several names used: "cytochrome oxidase", "cytochrome-c oxidase", just "oxidase"
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    values = []
    for e in as_list(pm.get("enzymes")):
        if not isinstance(e, dict):
            continue
        nm = str(e.get("value", "")).lower()
        if "cytochrome" in nm and "oxidase" in nm:
            values.append(e.get("activity"))
        elif nm.strip() == "oxidase":
            values.append(e.get("activity"))
    return majority_bool(values)


def x_halophily(rec: dict) -> str | None:
    """
    Derive halophily class from per-concentration growth tests.
    Heuristic: highest NaCl % that supports growth.
        >15%  → extreme_halophile
        2-15% requires salt → halophile
        2-15% grows but doesn't require → halotolerant
        only 0% → non_halophile
    """
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    tests = as_list(pm.get("halophily"))
    if not tests:
        return None
    grows_at: list[float] = []
    no_growth_at: list[float] = []
    for t in tests:
        if not isinstance(t, dict):
            continue
        if str(t.get("salt", "")).lower() != "nacl":
            continue
        conc = t.get("concentration", "")
        # Parse percentage from strings like "0 %", "2-15 %", "10%"
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:[-–]\s*(-?\d+(?:\.\d+)?))?\s*%?", str(conc))
        if not m:
            continue
        hi = float(m.group(2)) if m.group(2) else float(m.group(1))
        if norm_bool(t.get("growth")):
            grows_at.append(hi)
        else:
            no_growth_at.append(hi)
    if not grows_at:
        return None
    max_growth = max(grows_at)
    min_growth = min(grows_at)
    if max_growth >= 15:
        return "extreme_halophile"
    if min_growth >= 2:
        return "halophile"  # requires salt
    if max_growth >= 2:
        return "halotolerant"
    return "non_halophile"


def x_temperature_class(rec: dict) -> str | None:
    """Bin optimal growth temperature into psychro / psychrotroph / meso / thermo / hyperthermo."""
    cult = rec.get("Culture and growth conditions", {}) or {}
    if not isinstance(cult, dict):
        return None
    temps_supporting_growth: list[float] = []
    for t in as_list(cult.get("culture temp")):
        if not isinstance(t, dict):
            continue
        if not norm_bool(t.get("growth")):
            continue
        # Prefer optimum if present in type or a separate field
        tval = parse_numeric(t.get("temperature"))
        if tval is not None:
            temps_supporting_growth.append(tval)
    if not temps_supporting_growth:
        return None
    # Use the median of growth-supporting temps as proxy for optimum
    tm = sorted(temps_supporting_growth)[len(temps_supporting_growth) // 2]
    if tm < 15:
        return "psychrophile"
    if tm < 20:
        return "psychrotroph"
    if tm < 45:
        return "mesophile"
    if tm < 80:
        return "thermophile"
    return "hyperthermophile"


def x_ph_class(rec: dict) -> str | None:
    cult = rec.get("Culture and growth conditions", {}) or {}
    if not isinstance(cult, dict):
        return None
    # Prefer the curated "PH range" field if present
    for p in as_list(cult.get("culture pH")):
        if not isinstance(p, dict):
            continue
        ph_range = p.get("PH range") or p.get("pH range")
        if ph_range:
            s = str(ph_range).strip().lower()
            if "acid" in s:
                return "acidophile"
            if "alka" in s:
                return "alkaliphile"
            if "neutr" in s:
                return "neutrophile"
    # Else derive from numeric pH supporting growth
    ph_supporting: list[float] = []
    for p in as_list(cult.get("culture pH")):
        if not isinstance(p, dict):
            continue
        if not norm_bool(p.get("ability")):
            continue
        v = parse_numeric(p.get("pH"))
        if v is not None:
            ph_supporting.append(v)
    if not ph_supporting:
        return None
    pm = sorted(ph_supporting)[len(ph_supporting) // 2]
    if pm < 5.5:
        return "acidophile"
    if pm > 8.5:
        return "alkaliphile"
    return "neutrophile"


def x_cultivation_medium(rec: dict) -> list[str] | None:
    """Extract list of MediaDive medium IDs that support growth."""
    cult = rec.get("Culture and growth conditions", {}) or {}
    if not isinstance(cult, dict):
        return None
    media_ids: set[str] = set()
    for m in as_list(cult.get("culture medium")):
        if not isinstance(m, dict):
            continue
        if not norm_bool(m.get("growth")):
            continue
        link = str(m.get("link", ""))
        mm = re.search(r"mediadive\.dsmz\.de/medium/(\d+)", link)
        if mm:
            media_ids.add(mm.group(1))
    return sorted(media_ids) if media_ids else None


def x_carbon_utilization(rec: dict) -> dict[str, bool] | None:
    """Extract per-substrate utilization: {substrate: True|False}."""
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    out: dict[str, bool] = {}
    for m in as_list(pm.get("metabolite utilization")):
        if not isinstance(m, dict):
            continue
        metabolite = m.get("metabolite")
        activity = m.get("utilization activity")
        if not metabolite:
            continue
        b = norm_bool(activity)
        if b is not None:
            # Last value wins if duplicate metabolite (rare)
            out[str(metabolite).lower()] = b
    return out or None


def x_metabolite_production(rec: dict) -> dict[str, bool] | None:
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    out: dict[str, bool] = {}
    for m in as_list(pm.get("metabolite production")):
        if not isinstance(m, dict):
            continue
        metabolite = m.get("metabolite")
        prod = m.get("production")
        if not metabolite:
            continue
        b = norm_bool(prod)
        if b is not None:
            out[str(metabolite).lower()] = b
    return out or None


def x_amr_phenotype(rec: dict) -> dict[str, str] | None:
    """Per-antibiotic phenotype: {drug: 'R'|'S'|'I'}."""
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    out: dict[str, str] = {}
    for a in as_list(pm.get("antibiotic resistance")):
        if not isinstance(a, dict):
            continue
        drug = a.get("metabolite")
        if not drug:
            continue
        is_r = norm_bool(a.get("is resistant"))
        is_s = norm_bool(a.get("is sensitive"))
        if is_r is True:
            out[str(drug).lower()] = "R"
        elif is_s is True:
            out[str(drug).lower()] = "S"
        elif is_r is False and is_s is False:
            out[str(drug).lower()] = "I"
    return out or None


def x_biosafety_level(rec: dict) -> str | None:
    safety = rec.get("Interaction and safety", {}) or {}
    if not isinstance(safety, dict):
        return None
    bsls = []
    for ra in as_list(safety.get("risk assessment")):
        if not isinstance(ra, dict):
            continue
        v = ra.get("biosafety level")
        if v is None:
            continue
        m = re.match(r"\s*(\d)", str(v))
        if m:
            bsls.append(int(m.group(1)))
    if not bsls:
        return None
    return f"BSL-{max(bsls)}"  # most conservative


def _pathogenicity(rec: dict, field_keys: tuple[str, ...]) -> bool | None:
    """
    BacDive pathogenicity fields use freeform strings (e.g., 'yes, in single cases',
    'yes', 'no', 'no, but opportunistic'). Use prefix matching rather than exact
    token matching to capture the long tail of qualifications.
    """
    safety = rec.get("Interaction and safety", {}) or {}
    if not isinstance(safety, dict):
        return None
    for ra in as_list(safety.get("risk assessment")):
        if not isinstance(ra, dict):
            continue
        for key in field_keys:
            if key in ra:
                s = str(ra[key]).strip().lower()
                if s.startswith("yes"):
                    return True
                if s.startswith("no"):
                    return False
    return None


def x_pathogenicity_human(rec: dict) -> bool | None:
    return _pathogenicity(rec, ("pathogenicity human", "pathogenicity_human"))


def x_pathogenicity_animal(rec: dict) -> bool | None:
    return _pathogenicity(rec, ("pathogenicity animal", "pathogenicity_animal"))


def x_isolation_source(rec: dict) -> str | None:
    """v1: return raw `sample type` text; Phase 1.5 will map to ~20-class taxonomy."""
    iso = rec.get("Isolation, sampling and environmental information", {}) or {}
    if not isinstance(iso, dict):
        return None
    for s in as_list(iso.get("isolation")):
        if not isinstance(s, dict):
            continue
        st = s.get("sample type")
        if st:
            return str(st).strip()
    return None


def x_country(rec: dict) -> str | None:
    iso = rec.get("Isolation, sampling and environmental information", {}) or {}
    if not isinstance(iso, dict):
        return None
    for s in as_list(iso.get("isolation")):
        if not isinstance(s, dict):
            continue
        c = s.get("country")
        if c:
            return str(c).strip()
    return None


def x_fatty_acid_profile(rec: dict) -> dict[str, float] | None:
    """Extract FAME profile as {fatty_acid_name: percentage}."""
    pm = rec.get("Physiology and metabolism", {}) or {}
    if not isinstance(pm, dict):
        return None
    profile = pm.get("fatty acid profile")
    if not isinstance(profile, dict):
        return None
    out: dict[str, float] = {}
    for fa in as_list(profile.get("fatty acids")):
        if not isinstance(fa, dict):
            continue
        name = fa.get("fatty acid")
        pct = fa.get("percentage")
        if name is None or pct is None:
            continue
        try:
            out[str(name)] = float(pct)
        except (TypeError, ValueError):
            continue
    return out or None


# =============================================================================
# Trait registry
# =============================================================================

EXTRACTORS = {
    "gram_stain": x_gram_stain,
    "cell_shape": x_cell_shape,
    "motility": x_motility,
    "sporulation": x_sporulation,
    "pigmentation": x_pigmentation,
    "oxygen_tolerance": x_oxygen_tolerance,
    "catalase": x_catalase,
    "cytochrome_oxidase": x_cytochrome_oxidase,
    "halophily": x_halophily,
    "temperature_class": x_temperature_class,
    "ph_class": x_ph_class,
    "cultivation_medium": x_cultivation_medium,
    "carbon_utilization": x_carbon_utilization,
    "metabolite_production": x_metabolite_production,
    "amr_phenotype": x_amr_phenotype,
    "biosafety_level": x_biosafety_level,
    "pathogenicity_human": x_pathogenicity_human,
    "pathogenicity_animal": x_pathogenicity_animal,
    "isolation_source": x_isolation_source,
    "country": x_country,
    "fatty_acid_profile": x_fatty_acid_profile,
}

TAXONOMY_COLS = ["domain", "phylum", "class", "order", "family", "genus", "species", "type_strain"]


# =============================================================================
# Driver
# =============================================================================


def parse_record(rec: dict) -> dict:
    """Apply all extractors to one strain record."""
    out: dict[str, Any] = {"bacdive_id": rec.get("_bacdive_id")}
    out.update(x_taxonomy(rec))
    for name, fn in EXTRACTORS.items():
        try:
            out[name] = fn(rec)
        except Exception as e:
            out[name] = None
            out.setdefault("_parse_errors", []).append(f"{name}: {type(e).__name__}: {e}")

    # Post-derivation: BacDive's `pathogenicity human/animal` fields are
    # positive-only (only filled when documented "yes"). Use BSL-1 ("not a
    # recognized disease-causing agent" — German biosafety classification) as
    # a negative-class proxy when no explicit positive is present. This is
    # the standard convention in microbial-pathogenicity datasets and gives
    # the model proper binary labels instead of positive-only signals.
    if out.get("biosafety_level") == "BSL-1":
        if out.get("pathogenicity_human") is None:
            out["pathogenicity_human"] = False
        if out.get("pathogenicity_animal") is None:
            out["pathogenicity_animal"] = False
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_IN, help="Input JSONL")
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUT, help="Output parquet")
    parser.add_argument("--limit", type=int, default=0, help="Parse only first N records (0 = all)")
    args = parser.parse_args()

    if not args.input_path.exists():
        sys.exit(f"Input file not found: {args.input_path}")

    print(f"Parsing {args.input_path} → {args.output_path}")
    rows = []
    error_count = 0
    with args.input_path.open() as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                error_count += 1
                continue
            row = parse_record(rec)
            if row.get("_parse_errors"):
                error_count += 1
            row.pop("_parse_errors", None)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nParsed {len(df):,} strains; {error_count:,} had extractor errors.")
    print(f"\nPer-trait non-null counts (coverage on this sample):\n")
    for col in TAXONOMY_COLS + list(EXTRACTORS.keys()):
        if col not in df.columns:
            continue
        n = df[col].notna().sum()
        pct = 100 * n / len(df) if len(df) else 0
        print(f"  {col:<24} {n:>6,} ({pct:5.1f}%)")

    args.output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"\nWrote {len(df):,} rows × {len(df.columns)} cols to {args.output_path}")


if __name__ == "__main__":
    main()
