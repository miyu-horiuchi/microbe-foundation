# microbe-foundation: A Unified Benchmark and Multi-Task Baseline for Predicting the Complete Microbial Species Description from Genome

**Authors.** Miyu Horiuchi, …

**Affiliations.** …

**Correspondence.** miyu-horiuchi@…

---

## Abstract

Predicting microbial phenotypes from genome sequence has been a fragmented field: each prior method targets a small slice of the species-description surface (one or a few traits), uses task-specific feature engineering, and reports results on incompatible evaluation splits. Three foundation-model preprints in the last twelve months — MicroGenomer, Bacformer, and BacPT — have begun consolidating this space at the encoder level, but no single benchmark spans the full IJSEM/BacDive trait surface or evaluates at the family-held-out level required to test out-of-distribution generalization. We present **microbe-foundation**, a unified benchmark of **21 prediction heads** spanning morphology, physiology, growth conditions, cultivation, safety, ecology, and **chemotaxonomy** — the last of which is a literature-confirmed white-space (no prior model predicts fatty-acid composition from genome). We define **family-held-out, genus-held-out, and species-held-out splits** stratified by strain count, releasing all three for direct comparison with prior protocols. We provide a multi-task model with **masked loss** over heterogeneous sparse labels, evaluated against three feature encoders (KO presence, ESM-2 mean-pool, Bacformer). All data, splits, vocabularies, baselines, and checkpoints are publicly released. The benchmark anchors a new comparison surface; the contribution is the assembly, not a new encoder.

## 1. Introduction

…

**Contributions.**

1. **A unified BacDive-native trait benchmark** spanning the full IJSEM species-description surface (21 heads, 7 trait blocks) — broader than any existing benchmark.
2. **The first chemotaxonomy-from-genome prediction task** (fatty-acid profile, multi-output regression), addressing a literature-confirmed white-space.
3. **Family-held-out evaluation splits**, strictly harder than the genus-only splits used by BacBench and previous trait-prediction work.
4. **A masked multi-task model architecture** handling the heavy missing-label structure of BacDive (5–95% coverage per trait), with three released feature-encoder baselines.

## 2. Related Work

_See `RELATED_WORK.md` for the full draft. Summary:_

Classical trait predictors (Traitar [@weimann2016traitar], Brbić et al. [@brbic2016landscape], MicroPheno [@asgari2018micropheno], Genome Properties [@richardson2019genome], MiGenPro [@loomans2025migenpro]) tackle per-trait classification with hand-engineered features. Cultivation medium prediction has been dormant since KOMODO [@oberhardt2015komodo] until our prior work. Modern foundation models (MicroGenomer [@wang2025microgenomer], Bacformer [@wiatrak2025bacformer], BacPT [@bacpt2026]) consolidate the encoder side but each targets only a slice of the trait space. microbe-foundation builds on this lineage and closes four currently-unfilled gaps: full BacDive coverage, masked multi-task training, family-level evaluation, and chemotaxonomy prediction.

## 3. The microbe-foundation Benchmark

### 3.1 Trait surface

The benchmark covers 21 prediction heads grouped into seven biological blocks: morphology, physiology, growth conditions, cultivation, safety, ecology, and chemotaxonomy. Each head is defined by a value-extraction rule against BacDive's structured fields (see `parse_bacdive.py`).

<!-- tables/01_trait_inventory.md -->

### 3.2 Data source and parsing

Labels come from **BacDive** [@schober2025bacdive], the curated bacterial-diversity database, accessed via the public REST API (CC-BY 4.0). Each strain is parsed by a per-trait extractor that normalizes BacDive's polymorphic dict-or-list-of-dicts shape, handles freeform string values ("yes", "yes, in single cases", etc.), bins continuous measurements (temperature into psychro/meso/thermo classes; pH into acido/neutro/alkali classes), and derives binary negatives for pathogenicity from biosafety-level evidence. Cultivation medium IDs are joined to MediaDive [@koblitz2023mediadive] for downstream recipe-level enrichment.

### 3.3 Label coverage

<!-- tables/02_label_coverage.md -->

Coverage is heavily imbalanced across traits, ranging from ~5% (FAME profile) to >90% (cultivation medium, temperature). The masked multi-task loss handles this directly: every strain contributes whatever labels it has.

### 3.4 Phylogeny-aware splits

We hold out whole taxonomic groups at the family, genus, and species levels using a **largest-group-first greedy bin-packing** algorithm that stratifies by strain count rather than group count, reaching exact 80/10/10 ratios. Whole groups never span buckets, so cross-group leakage is impossible by construction. The family-held-out split is microbe-foundation's primary protocol — strictly harder than BacBench's genus-only splits because it tests cross-family generalization.

<!-- tables/03_split_stats.md -->

### 3.5 Discovered vocabularies

For multilabel and regression-vector heads, the class vocabulary is data-derived: top-N MediaDive media by frequency, top-30 most-reported fatty acids, top-80 carbon substrates following the Madin et al. [@madin2020synthesis] catalog.

<!-- tables/04_vocabulary_sizes.md -->

## 4. Baseline Models

### 4.1 Multi-task architecture

All baselines share a common architecture: a per-genome feature vector enters a shared MLP encoder (two hidden layers, ReLU, dropout 0.2), then routes to **one linear head per trait** sized by the head type (1 for binary, K for multiclass/multilabel, K for regression-vector). With hidden size 512, the model is ~340k parameters — small enough to run on CPU but expressive enough to fit the trait space.

### 4.2 Masked multi-task loss

For each head, we apply a head-appropriate loss only on the subset of strains with a label for that head:
- **Binary**: BCE on the labeled subset
- **Multiclass**: cross-entropy on the labeled subset
- **Multilabel**: per-element BCE, masked. For list-typed labels (cultivation medium), we adopt the implicit-negative assumption — strains with *any* documentation for the head treat unlisted vocab items as negative.
- **Regression-vector** (FAME): masked MSE over reported fatty acids, with percentages normalized to [0, 1].

Per-head losses are equally weighted and averaged across active heads; per-batch heads with no labeled samples contribute zero.

### 4.3 Feature encoders

We compare three feature sources, all producing one fixed-dimension vector per genome:

1. **KO presence/absence** (planned): eggNOG-mapper annotations → ~10k-feature sparse vector → MLP. The honest baseline.
2. **ESM-2 mean-pool**: pyrodigal CDS prediction → ESM-2 per-protein embeddings → mean-pool over proteome. Multiple sizes (8M to 650M). The strong baseline; reuses the existing microbe-model v0 pipeline.
3. **Bacformer**: frozen Bacformer-large genome embeddings as features. The direct peer comparator — explicitly tests whether a Bacformer pretrain transfers as well as a custom training run.

## 5. Results

<!-- Auto-populate from model.py runs. Template: -->

<!-- tables/05_results_template.md -->

### 5.1 Per-block performance

…

### 5.2 The chemotaxonomy white-space

The fatty-acid (FAME) profile head has no prior published predictor. We report RMSE on the 30-dimensional normalized FAME vector, family-held-out. _(Numbers to fill once trained.)_

### 5.3 Cross-encoder comparison

…

### 5.4 Comparison to prior benchmarks

Where heads overlap with BacBench, Koblitz et al. 2025 [@koblitz2025bacdiveml], or single-trait predictors (Traitar, Engqvist 2018 OGT, Ramoneda 2023 pH), we report side-by-side numbers on shared traits.

## 6. Discussion

…

### Limitations

- **Polar lipids and respiratory quinones** are not in BacDive's structured fields and remain unaddressed in v1.
- **Pathogenicity** is positive-only in BacDive; we derive negatives from BSL-1, a standard but imperfect proxy.
- **Label noise** across BacDive references is unmeasured; some traits (e.g., oxygen tolerance) show multiple disagreeing entries per strain that we resolve by majority vote.

### Future work

- Adding polar lipids / quinones via IJSEM PDF mining (a v2 effort)
- Fine-tuning the encoder end-to-end with attention pooling over proteins
- DNA-LM ablations (Evo 2 [@brixi2026evo2], gLM2 [@cornman2025glm2]) on traits with non-coding signal

## 7. Conclusion

microbe-foundation consolidates the fragmented microbial trait-prediction literature into a unified BacDive-native benchmark, opens chemotaxonomy as a new prediction category, and provides strong masked multi-task baselines under family-held-out evaluation. We hope it anchors the next generation of work the way ImageNet anchored computer vision — through the benchmark, not the model.

## Code and Data Availability

All code, data fetchers, parsers, splits, vocabularies, baseline models, and trained checkpoints are released at **https://github.com/miyu-horiuchi/microbe-foundation** under MIT license. Source data is BacDive (CC-BY 4.0) via the public v2 API (no authentication required). The full pipeline reproduces from raw BacDive in `python fetch_bacdive.py && python parse_bacdive.py && python splits.py && python vocab.py && python compute_esm2_features.py && python model.py`.

## References

_Generated from `references.bib`. See repository._
