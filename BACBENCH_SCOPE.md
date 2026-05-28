# BacBench Scope Analysis

**Investigated:** 2026-05-28
**Repo:** https://github.com/macwiatrak/BacBench
**Created:** 2025-05-15 · **Last push:** 2026-05-26 · **Last commit:** 2026-05-21
**Stars:** 19 · **Forks:** 4 · **Open issues:** 2 · **License:** Apache-2.0

## TL;DR Verdict

**BacBench covers ~25-35% of the planned microbe-foundation trait surface.** It is active, well-engineered, and has solid baselines. The phenotypic-traits task uses the **Madin 2020 + Brbic 2016 + Gideon** label sets (139 traits across 24,462 genomes), which means there is **substantial overlap on Tier-1/Tier-2 traits** (Gram stain, motility, sporulation, cell shape, oxygen metabolism, T/pH/salinity, carbon utilization, isolation source). But it does **not cover** the chemotaxonomy white-space (FAME / polar lipids / peptidoglycan / quinones / mol% G+C), does not use BacDive directly, has **no public leaderboard** yet, and uses a single shared genome pool for all traits (so it cannot study label-shift or per-trait taxonomic bias the way a BacDive-native schema can).

**Headline gap = the chemotaxonomy fields + BacDive provenance + medium/cultivation linkages.** That is exactly where microbe-foundation can claim novelty.

---

## Project Health

| Signal | Value |
|---|---|
| First commit / repo created | 2025-05-15 |
| Most recent push | 2026-05-26 (2 days ago) |
| Stars / forks | 19 / 4 |
| Open / total issues | 2 open, 21 total (mostly feature-branch PRs from the author) |
| Active maintainer | Macwiatrak (sole committer) |
| License | Apache 2.0 |
| News last entry | 2026-05-13: phylogeny-aware genus split added to Essential Genes & PPI; new BacLM model |
| Leaderboard | **No leaderboard yet** (open TODO: "Create model leaderboard for each task") |
| Published paper | **No preprint yet** ("Citation details will be added when the manuscript/preprint is available.") |
| HuggingFace datasets | Public: https://huggingface.co/collections/macwiatrak/bacbench-6819ea4b0a226beef8d29f81 |

Conclusion: BacBench is a one-maintainer project that is being actively developed but is still pre-publication. Low community footprint (19 stars). This means: (a) the benchmark is moving, (b) numbers will likely change before publication, (c) there is room for an independent benchmark to coexist or supersede.

---

## Tasks (5 active, 1 deprecated)

| # | Task | Genomes | Modality | Labels source | Status | Split |
|---|---|---:|---|---|---|---|
| 1 | **Essential genes** | 51 | DNA + protein | DEG (Database of Essential Genes) | Active | **Random 60/20/20** (per HF card); README news says "phylogeny-aware genus split added 2026-05-13" — code path supports both |
| 2 | **Operon identification** | 5 | DNA + protein | Long-read RNA-seq operon calls | Active | Per-contig |
| 3 | **PPI** (small) | 261 | DNA + protein | STRING-DB combined score | Active | **Genus-disjoint 60/10/20** |
| 3b | PPI (full) | >10,000 | Protein only | STRING-DB | Active | Genus-disjoint |
| 4 | **Antibiotic resistance** | ~25,032 (across 39 species) | DNA + protein | NCBI AST Browser (Oct 2024 snapshot) | Active | **5-fold StratifiedKFold for binary / KFold for MIC; split on `genome_id` to prevent leakage** |
| 5 | **Phenotypic traits** | **24,462** (across 15,477 species, 2,668 genera, 640 families) | DNA + protein | Madin 2020 + Weimann 2016 (Traitar) + Brbic 2016 + Gideon | Active | **Default `--split genus` (genus-disjoint group split)** with random fallback; 60/20/20 × 5 seeds recommended |
| - | Strain clustering | n/a | DNA + protein | MAG-based | **Deprecated** as of 2026-05-13 | — |

**Splits — concrete answer:** AMR uses stratified k-fold with `genome_id`-level grouping. Essential genes + PPI moved to **genus-aware splits** in May 2026. Phenotypic traits defaults to **genus-disjoint split** if the `genus` column is present (it is — verified in the labels.csv). No family-level or higher splits offered out of the box. There is no held-out clade for novel-phylum evaluation.

---

## Phenotypic traits task — full label inventory

I downloaded `labels.csv` directly from HuggingFace (24,462 rows × 143 columns = 4 metadata + 139 trait columns) and computed per-trait non-null counts.

### Coverage summary
- **139 / 139 traits** have ≥500 labeled genomes
- **108 / 139 traits** have ≥1,000 labeled genomes
- Total genomes: 24,462 across 15,477 species / 2,668 genera / 640 families

### Top-labeled categorical phenotypes (Madin)
| Trait | Labeled genomes |
|---|---:|
| `isolation_source` | 15,752 |
| `gram_stain` | 14,051 |
| `metabolism` (aerobe/anaerobe/facultative/etc.) | 10,052 |
| `cell_shape` | 9,907 |
| `motility_binary` | 8,508 |
| `growth_tmp` (quant) | 7,921 |
| `sporulation` | 7,624 |
| `optimum_tmp` (quant) | 7,082 |
| `range_tmp` | 4,841 |
| `optimum_ph` (quant) | 2,872 |
| `motility_type` | 846 |
| `range_salinity` | 742 |

### Categorical phenotype categories present
The labels fall into 5 groups (from `phenotypic_traits_grouppings.csv`):
1. **Morphology** — gram stain, cell shape, motility (binary + type), sporulation, coccus, bacillus, pigment, branching filaments, spore formation, cell dimensions (d1_lo, d1_up, d2_lo, d2_up)
2. **RespirationMetabolism** — aerobe, anaerobe, facultative, glucose oxidizer/fermenter, catalase, oxidase, gas from glucose, nitrate→nitrite
3. **GrowthConditions** — temperature optimum/range, pH optimum, salinity range, growth on blood agar, growth on MacConkey agar, isolation source
4. **BiochemicalActivity** — indole, urea hydrolysis, hydrogen sulfide, esculin hydrolysis, gelatin hydrolysis, arginine dihydrolase, β-galactosidase (ONPG), β-hemolysis
5. **CarbonUtilization** — ~80+ carbohydrates, organic acids, amino acids (cellobiose, fructose, glucose, lactose, maltose, sucrose, trehalose, xylose, mannitol, sorbitol, raffinose, melibiose, rhamnose, citrate, malate, succinate, pyruvate, Tween 20/40/60/80, ethanol, glycerol, Adonitol, salicin, etc.)

### What is EXPLICITLY **not** present in BacBench labels
A full grep of the 139 column headers confirms BacBench has **zero** of these chemotaxonomy fields:
- ❌ Fatty acid (FAME) composition
- ❌ Polar lipid composition
- ❌ Peptidoglycan type
- ❌ Respiratory quinones (MK-7, Q-8, etc.)
- ❌ mol% G+C content
- ❌ Cultivation medium (linked to MediaDive)
- ❌ Pathogenicity (no human/animal/plant pathogenicity column — even though Gideon labels include some clinically-relevant traits like β-hemolysis)
- ❌ BacDive provenance (labels come from Madin/Traitar/Brbic/Gideon — not BacDive)

Gram-positive/Gram-negative and gram_stain are duplicated across Madin and Gideon (kept as separate labels by design, per the dataset card).

---

## Baselines & benchmarked models

BacBench supplies embedding scripts and evaluation harnesses for 12 models (table from README):

| Model | Type | Params |
|---|---|---|
| Bacformer / Bacformer Large | Multi-protein context | 27M / ~85M |
| **BacLM** (new, 2026-05-13) | Mixed DNA+protein, masked LM | 350M |
| Evo2 | DNA AR | 1B |
| Evo (1-8k) | DNA AR | 6.5B |
| Nucleotide Transformer v2 | DNA masked | 250M |
| ProkBERT | DNA masked | 27M |
| DNABERT-2 | DNA masked | 117M |
| Mistral-DNA-bacteria | DNA AR | 138M |
| ESM-2 (35M) | Protein masked | 35M |
| ESM-C (300M) | Protein masked | 300M |
| ESMPlusPlus | Protein masked | 300M |
| ProtBert | Protein masked | 420M |
| gLM2 | Mixed DNA+protein | 650M |

**Critically: there are no published baseline numbers in the repo or on the HuggingFace dataset cards.** Users are expected to embed and train themselves with the provided scripts. There is no leaderboard. There is no preprint.

The evaluation protocol for phenotypic traits is: pool whole-genome embedding (mean pool over per-protein or per-DNA-window embeddings), train one MLP per trait via PyTorch Lightning, 3 random seeds, 60/20/20 split, report macro AUROC. This is what microbe-foundation should be benchmarked against if direct comparison is the goal.

---

## Coverage versus microbe-foundation's planned trait surface

Mapping the user's requested trait list against what BacBench actually covers:

### Tier 1 (binary / categorical)
| Trait | BacBench? | Source |
|---|---|---|
| Gram stain | ✅ | Madin + Gideon (14,051 + ~1k) |
| Oxygen tolerance / requirement | ✅ | Madin `metabolism` (10,052), Gideon Aerobe/Anaerobe/Facultative |
| Cell shape | ✅ | Madin (9,907) + Gideon coccus/bacillus |
| Motility | ✅ | Madin (8,508 binary, 846 type) + Gideon motile |
| Sporulation | ✅ | Madin (7,624) + Gideon spore formation |
| Spore type | ❌ | Not present |
| Pigmentation | ⚠️ partial | Only "yellow pigment" (Gideon) — no full color palette |
| Catalase, oxidase | ✅ | Gideon (catalase, oxidase, ONPG) |
| Pathogenicity (human/animal/plant) | ❌ | Not present (β-hemolysis is the closest proxy) |

### Tier 2 (binned-continuous / curated)
| Trait | BacBench? | Source |
|---|---|---|
| Temperature optimum / range | ✅ | Madin quant + categorical (4,841–7,921) |
| pH optimum / range | ⚠️ | Optimum only (2,872) — no pH range column |
| NaCl tolerance / halophily | ⚠️ | Range salinity (742) — barely above threshold |
| Cultivation medium | ❌ | No structured medium column |
| Carbon source utilization | ✅✅ | ~80 carbohydrates/acids/amino-acids columns (each ~2,903 strains, Madin pseudo-binary) |
| Isolation source | ✅ | Madin (15,752) |

### Tier 4 (chemotaxonomy — the white-space)
| Trait | BacBench? |
|---|---|
| Fatty acid (FAME) composition | ❌ |
| Polar lipid composition | ❌ |
| Peptidoglycan type | ❌ |
| Respiratory quinones (MK-7, Q-8) | ❌ |
| mol% G+C of DNA | ❌ |

### Other notable
| Trait | BacBench? |
|---|---|
| AMR (binary + MIC) | ✅ — dedicated task, 36 binary / 56 MIC antibiotics, ~25k genomes |
| PPI | ✅ — dedicated task |
| Essential genes | ✅ — dedicated task |

---

## Positioning recommendation

1. **Do not duplicate** BacBench's phenotypic-traits task on the Madin/Gideon traits unless microbe-foundation can demonstrate a substantially better protocol (per-trait class balancing, family-held-out splits, calibration). BacBench is the obvious comparator on those 139 traits.
2. **Do build** the chemotaxonomy + medium + pathogenicity surface. That is genuinely uncovered by BacBench and is exactly what BacDive's structured fields can support.
3. **Do consider** using BacBench's AMR task as a benchmark axis — microbe-foundation should report AMR macro-AUROC on the BacBench AMR labels even if it adds the chemotaxonomy traits as the novel contribution.
4. **Do report** numbers on BacBench's phenotypic-traits labels too — Bacformer Large + a simple MLP head is the de-facto baseline you must beat. The labels.csv is 6.6 MB and fully public; the embeddings are computable in a day on a single GPU.
5. **Do offer family-level and phylum-level held-out splits** in the new benchmark — BacBench only goes down to genus. A model that generalizes to unseen families is a stronger claim.
6. **Do link to MediaDive** for cultivation medium. BacBench has no equivalent.

---

## Sources

- BacBench README and tasks/ source: https://github.com/macwiatrak/BacBench (read 2026-05-28)
- BacBench phenotypic-traits labels.csv (downloaded 2026-05-28, 24,462 × 143)
- HF dataset cards: phenotypic-traits-protein-sequences, antibiotic-resistance-protein-sequences, ppi-stringdb-protein-sequences-small, essential-genes-protein-sequences
- HF collection: https://huggingface.co/collections/macwiatrak/bacbench-6819ea4b0a226beef8d29f81
- Madin 2020 Sci Data 7:170; Weimann 2016 mSystems 1; Brbic 2016 NAR 44 (referenced as label sources)
