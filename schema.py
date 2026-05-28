"""
microbe-foundation v1 trait schema.

Defines the prediction targets for the multi-task model. Each Trait specifies:
- Where the label comes from (BacDive structured field path)
- The prediction head type (binary, multiclass, multilabel, regression vector)
- Estimated label count (from BACDIVE_COVERAGE.md, 2026-05-28 audit)
- The trait block, used to group losses and report per-block performance

The schema is the source of truth for Phase 1 data merging and Phase 2 model
construction. Run this file to export `trait_schema.json` and print a summary.

Out of scope (deferred or skipped):
- Polar lipids, respiratory quinones — no structured BacDive field (need IJSEM PDF mining)
- Peptidoglycan type — chemotaxonomic only, ~0 functional signal
- mol% G+C — directly computable from sequence, not a prediction task
- Pathogenicity (plant) — borderline label count, defer
- Exact cell dimensions, colony morphology, exact MIC — not predictable from sequence
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = "0.1.0"
SCHEMA_DATE = "2026-05-28"


class HeadType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    REGRESSION_VECTOR = "regression_vector"


class Block(str, Enum):
    MORPHOLOGY = "morphology"
    PHYSIOLOGY = "physiology"
    GROWTH = "growth"
    CULTIVATION = "cultivation"
    SAFETY = "safety"
    ECOLOGY = "ecology"
    CHEMOTAXONOMY = "chemotaxonomy"


class ClassesSource(str, Enum):
    """Where the class vocabulary is defined."""
    ENUMERATED = "enumerated"          # listed in `classes` below
    BACDIVE_ENUM = "bacdive_enum"      # discover from BacDive values during Phase 1
    MADIN_SUBSTRATES = "madin_substrates"  # Madin et al. carbon-substrate catalog
    MEDIADIVE_TOP_N = "mediadive_top_n"    # top-N MediaDive medium IDs by frequency
    BACDIVE_FAME_TOP_N = "bacdive_fame_top_n"  # top-N FAMEs by frequency in BacDive
    CARD_DRUG_CLASSES = "card_drug_classes"    # CARD/RGI drug class taxonomy


@dataclass
class Trait:
    name: str
    block: Block
    head: HeadType
    bacdive_path: str
    description: str
    estimated_label_count: int
    classes_source: ClassesSource = ClassesSource.ENUMERATED
    classes: Optional[list[str]] = None
    n_outputs: Optional[int] = None  # for regression_vector / multilabel when classes are data-derived
    notes: str = ""


# =============================================================================
# v1 trait definitions
# =============================================================================

V1_TRAITS: list[Trait] = [
    # --- Block 1: Morphology ---
    Trait(
        name="gram_stain",
        block=Block.MORPHOLOGY,
        head=HeadType.MULTICLASS,
        bacdive_path="Morphology.cell_morphology.gram_stain",
        description="Gram stain reaction.",
        estimated_label_count=15_900,
        classes=["positive", "negative", "variable"],
    ),
    Trait(
        name="cell_shape",
        block=Block.MORPHOLOGY,
        head=HeadType.MULTICLASS,
        bacdive_path="Morphology.cell_morphology.cell_shape",
        description="Predominant cell morphology.",
        estimated_label_count=15_900,
        classes=["coccus", "rod", "vibrio", "spiral", "filament", "pleomorphic", "other"],
    ),
    Trait(
        name="motility",
        block=Block.MORPHOLOGY,
        head=HeadType.BINARY,
        bacdive_path="Morphology.cell_morphology.motility",
        description="Whether the organism is motile.",
        estimated_label_count=15_300,
    ),
    Trait(
        name="sporulation",
        block=Block.MORPHOLOGY,
        head=HeadType.BINARY,
        bacdive_path="Morphology.multicellular_morphology.spore_formation",
        description="Whether the organism forms spores.",
        estimated_label_count=5_600,
        notes="Field may also appear under Physiology_and_metabolism.spore_formation.",
    ),
    Trait(
        name="pigmentation",
        block=Block.MORPHOLOGY,
        head=HeadType.BINARY,
        bacdive_path="Morphology.pigmentation",
        description="Whether the organism produces pigments.",
        estimated_label_count=6_800,
    ),

    # --- Block 2: Physiology ---
    Trait(
        name="oxygen_tolerance",
        block=Block.PHYSIOLOGY,
        head=HeadType.MULTICLASS,
        bacdive_path="Physiology_and_metabolism.oxygen_tolerance",
        description="Oxygen requirement / tolerance class.",
        estimated_label_count=23_700,
        classes=[
            "obligate_aerobe",
            "facultative_aerobe",
            "microaerophile",
            "aerotolerant",
            "facultative_anaerobe",
            "obligate_anaerobe",
        ],
    ),
    Trait(
        name="catalase",
        block=Block.PHYSIOLOGY,
        head=HeadType.BINARY,
        bacdive_path="Physiology_and_metabolism.enzymes[value=catalase].activity",
        description="Catalase enzyme activity (positive / negative).",
        estimated_label_count=14_600,
    ),
    Trait(
        name="cytochrome_oxidase",
        block=Block.PHYSIOLOGY,
        head=HeadType.BINARY,
        bacdive_path="Physiology_and_metabolism.enzymes[value=cytochrome_oxidase].activity",
        description="Cytochrome oxidase activity (positive / negative).",
        estimated_label_count=13_100,
    ),
    Trait(
        name="halophily",
        block=Block.PHYSIOLOGY,
        head=HeadType.MULTICLASS,
        bacdive_path="Physiology_and_metabolism.halophily",
        description="Salt tolerance class (curated, not raw NaCl optimum).",
        estimated_label_count=13_100,
        classes=["non_halophile", "halotolerant", "halophile", "extreme_halophile"],
    ),

    # --- Block 3: Growth conditions ---
    Trait(
        name="temperature_class",
        block=Block.GROWTH,
        head=HeadType.MULTICLASS,
        bacdive_path="Culture_and_growth_conditions.culture_temp",
        description="Temperature optimum binned into growth class.",
        estimated_label_count=48_900,
        classes=["psychrophile", "psychrotroph", "mesophile", "thermophile", "hyperthermophile"],
        notes=(
            "Bin from numeric optimum: psychro <15C, psychrotroph 15-20C, meso 20-45C, "
            "thermo 45-80C, hyperthermo >80C. Phase 1 must convert raw temp ranges to class."
        ),
    ),
    Trait(
        name="ph_class",
        block=Block.GROWTH,
        head=HeadType.MULTICLASS,
        bacdive_path="Culture_and_growth_conditions.culture_pH",
        description="pH optimum binned into growth class.",
        estimated_label_count=7_800,
        classes=["acidophile", "neutrophile", "alkaliphile"],
        notes="Bin from numeric optimum: acidophile <5.5, neutrophile 5.5-8.5, alkaliphile >8.5.",
    ),

    # --- Block 4: Cultivation ---
    Trait(
        name="cultivation_medium",
        block=Block.CULTIVATION,
        head=HeadType.MULTILABEL,
        bacdive_path="Culture_and_growth_conditions.culture_medium[link~=mediadive]",
        description="Which MediaDive media support growth (set prediction).",
        estimated_label_count=28_000,
        classes_source=ClassesSource.MEDIADIVE_TOP_N,
        n_outputs=200,
        notes="Top-N most-frequent MediaDive medium IDs as multilabel targets. N TBD in Phase 1.",
    ),
    Trait(
        name="carbon_utilization",
        block=Block.CULTIVATION,
        head=HeadType.MULTILABEL,
        bacdive_path="Physiology_and_metabolism.metabolite_utilization",
        description="Per-substrate utilization (positive / not-tested-or-negative).",
        estimated_label_count=29_600,
        classes_source=ClassesSource.MADIN_SUBSTRATES,
        n_outputs=80,
        notes="Use Madin et al. curated ~80-substrate catalog for interoperability with BacBench.",
    ),
    Trait(
        name="metabolite_production",
        block=Block.CULTIVATION,
        head=HeadType.MULTILABEL,
        bacdive_path="Physiology_and_metabolism.metabolite_production",
        description="Per-metabolite production (e.g., indole, acid, gas).",
        estimated_label_count=24_600,
        classes_source=ClassesSource.BACDIVE_ENUM,
        n_outputs=50,
        notes="Class set discovered from BacDive value enumeration in Phase 1; cap at ~50 most frequent.",
    ),
    Trait(
        name="amr_phenotype",
        block=Block.CULTIVATION,
        head=HeadType.MULTILABEL,
        bacdive_path="Physiology_and_metabolism.antibiotic_resistance",
        description="Resistance phenotype per antibiotic drug class.",
        estimated_label_count=10_000,
        classes_source=ClassesSource.CARD_DRUG_CLASSES,
        n_outputs=20,
        notes="Map per-antibiotic R/S results to CARD drug-class taxonomy. Estimate is placeholder; verify in Phase 1.",
    ),

    # --- Block 5: Safety ---
    Trait(
        name="biosafety_level",
        block=Block.SAFETY,
        head=HeadType.MULTICLASS,
        bacdive_path="Application_and_interaction.risk_assessment.biosafety_level",
        description="Assigned biosafety level (BSL-1 through BSL-4).",
        estimated_label_count=26_200,
        classes=["BSL-1", "BSL-2", "BSL-3", "BSL-4"],
    ),
    Trait(
        name="pathogenicity_human",
        block=Block.SAFETY,
        head=HeadType.BINARY,
        bacdive_path="Application_and_interaction.risk_assessment.pathogenicity_human",
        description="Whether the organism is a known human pathogen.",
        estimated_label_count=2_800,
    ),
    Trait(
        name="pathogenicity_animal",
        block=Block.SAFETY,
        head=HeadType.BINARY,
        bacdive_path="Application_and_interaction.risk_assessment.pathogenicity_animal",
        description="Whether the organism is a known animal pathogen.",
        estimated_label_count=2_800,
    ),

    # --- Block 6: Ecology ---
    Trait(
        name="isolation_source",
        block=Block.ECOLOGY,
        head=HeadType.MULTICLASS,
        bacdive_path="Isolation_sampling_environmental_information.isolation",
        description="Habitat category derived from isolation source text.",
        estimated_label_count=60_400,
        classes_source=ClassesSource.BACDIVE_ENUM,
        n_outputs=20,
        notes=(
            "Phase 1 must map free-text isolation source into a curated ~20-class taxonomy "
            "(host-associated, marine, freshwater, soil, sediment, plant-associated, "
            "extreme-environment, food, industrial, clinical, etc.)."
        ),
    ),
    Trait(
        name="country",
        block=Block.ECOLOGY,
        head=HeadType.MULTICLASS,
        bacdive_path="Isolation_sampling_environmental_information.isolation.country",
        description="Country of isolation (ISO-3166 alpha-2 codes).",
        estimated_label_count=53_900,
        classes_source=ClassesSource.BACDIVE_ENUM,
        n_outputs=200,
        notes="Class set is the ISO-3166 alpha-2 codes that appear with >=10 strains in BacDive.",
    ),

    # --- Block 7: Chemotaxonomy (white-space differentiator) ---
    Trait(
        name="fatty_acid_profile",
        block=Block.CHEMOTAXONOMY,
        head=HeadType.REGRESSION_VECTOR,
        bacdive_path="Physiology_and_metabolism.fatty_acid_profile",
        description="Relative abundance (%) of each major fatty acid in the membrane.",
        estimated_label_count=7_200,
        classes_source=ClassesSource.BACDIVE_FAME_TOP_N,
        n_outputs=30,
        notes=(
            "First-ever genome-to-FAME predictor (literature confirmed white-space). "
            "Phase 1 picks top-30 most-frequent FAMEs across BacDive; output is a 30-d "
            "normalized abundance vector. Loss = masked MSE on log-ratio transform."
        ),
    ),
]


# =============================================================================
# Helpers
# =============================================================================

def _to_jsonable(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def export_json(out_path: Path) -> dict:
    data = {
        "schema_version": SCHEMA_VERSION,
        "schema_date": SCHEMA_DATE,
        "n_traits": len(V1_TRAITS),
        "traits": [_to_jsonable(asdict(t)) for t in V1_TRAITS],
    }
    out_path.write_text(json.dumps(data, indent=2))
    return data


def print_summary() -> None:
    by_block: dict[str, list[Trait]] = {}
    for t in V1_TRAITS:
        by_block.setdefault(t.block.value, []).append(t)

    print(f"\nmicrobe-foundation v1 trait schema — {len(V1_TRAITS)} heads across {len(by_block)} blocks\n")

    total_labels = 0
    for block_name in [b.value for b in Block]:
        traits = by_block.get(block_name, [])
        if not traits:
            continue
        block_labels = sum(t.estimated_label_count for t in traits)
        total_labels += block_labels
        print(f"  [{block_name}]  {len(traits)} heads, ~{block_labels:,} cumulative labels")
        for t in traits:
            head_info = t.head.value
            if t.n_outputs:
                head_info += f"({t.n_outputs})"
            elif t.classes:
                head_info += f"({len(t.classes)})"
            print(f"      - {t.name:<22} {head_info:<22} ~{t.estimated_label_count:>6,} labels")
        print()

    print(f"  Total cumulative labels across all heads: ~{total_labels:,}")
    print("  (note: same strain often labels multiple heads — this is sum, not unique)")


if __name__ == "__main__":
    out = Path(__file__).parent / "trait_schema.json"
    export_json(out)
    print_summary()
    print(f"\nWrote {out}")
