# Changelog

All notable changes to this project will be documented here. Format follows [Keep a Changelog](https://keepachangelog.com/); the project itself uses semantic-ish versioning tied to the trait schema version in `trait_schema.json`.

## [Unreleased]

Pending: real-data baselines (ESM-2 features computed on full BacDive corpus, model training, populated leaderboard). The code pipeline is complete; only execution remains.

## [0.1.0] — 2026-05-28

First end-to-end pipeline release. Locks the v1 trait schema and the benchmark protocol.

### Added

- **Trait schema** (`schema.py` + `trait_schema.json`): 21 prediction heads across 7 biological blocks (morphology, physiology, growth conditions, cultivation, safety, ecology, chemotaxonomy). Schema version 0.1.0.
- **Data pipeline**: `fetch_bacdive.py` (stdlib bulk fetcher for the public BacDive v2 API, resumable with exponential-backoff retries), `parse_bacdive.py` (21 trait extractors handling BacDive's dict-or-list-of-dicts polymorphism, BSL-1-derived pathogenicity negatives, FAME percentage extraction), `splits.py` (largest-group-first stratified packing for family/genus/species-held-out splits hitting exact 80/10/10), `vocab.py` (data-derived class vocabularies for multilabel and regression-vector heads), `extract_genome_accessions.py` (best-assembly-per-strain NCBI accession TSV).
- **Vendored microbe-model v0 modules** under `microbe_model/`: `pipeline.py` (parallel in-memory NCBI Datasets fetch), `features/genome.py` (pyrodigal CDS prediction), `features/embeddings.py` (ESM-2 mean-pool), `features/composition.py`, `features/kegg_modules.py`, `features/markers.py`, `data/bacdive.py`, `data/mediadive.py`. Adapted `config.py` for the flat repo layout.
- **Feature extraction**: `compute_esm2_features.py` (orchestrates the vendored modules; in-memory genome fetch through ESM-2 embeddings to NPZ), `compute_bacformer_features.py` (Bacformer alternative, scaffold pending GPU run).
- **Multi-task model** (`model.py`): shared MLP encoder + per-trait linear heads constructed dynamically from schema+vocab. Masked loss per head (binary BCE, multiclass CE, multilabel BCE with per-element mask, regression-vector MSE on reported FAMEs only). Implicit-negative assumption for list-typed multilabel (cultivation_medium). Feature-source-agnostic via `--features X.npz`.
- **Prior-work comparator**: `prior_numbers.json` (curated published scores from CITATION_AUDIT.md and live search) + `compare_to_priors.py` (joins our `model.py` metrics JSONs against prior numbers, produces 3-section paper table: directly-comparable / context / literature white-space).
- **Leaderboard**: `leaderboard.py` aggregates `runs/*.json` into a ranked per-head table with bolded best scores.
- **Paper scaffold** under `paper/`: full markdown draft (abstract, contributions, related work, benchmark description, methods, results template, discussion, limitations) + `generate_tables.py` regenerating 6 auto-tables from the live data files.
- **Tests** (`tests/`, 45 tests, ~2 s): schema invariants, parser correctness on a synthetic record covering all field shapes, split correctness (no group spans buckets, hits target ratios, seed determinism), model construction (heads match schema, masked loss flows gradients).
- **CI** (`.github/workflows/test.yml`): pytest on push/PR/manual for Python 3.11 + 3.12.
- **Pipeline runner**: `scripts/run_all.sh` with `--smoke`, `--skip-fetch`, `--skip-features` flags.
- **Documentation**: top-level `README.md`, submission-protocol `BENCHMARK.md`, narrative `RELATED_WORK.md`, audit docs (`CITATION_AUDIT.md`, `BACBENCH_SCOPE.md`, `BACDIVE_COVERAGE.md`, `LIVE_SEARCH_2026-05-28.md`).
- **Reference data**: `reference_data/media_metadata.parquet` + `reference_data/media_recipes.parquet` (MediaDive catalog, reused from microbe-model v0).

### Positioning

After the citation audit (`CITATION_AUDIT.md`), the project was reframed away from "first microbial foundation model" (no longer defensible — MicroGenomer, Bacformer, BacPT all staked that ground in 2025–2026) toward four currently-unfilled gaps: full BacDive species-description coverage, FAME chemotaxonomy white-space (literature-confirmed zero prior predictors), family-held-out splits strictly harder than BacBench's genus-only protocol, and masked multi-task formulation over sparse heterogeneous labels.

### Fixed during development

- `fetch_bacdive.py` originally crashed on socket.timeout in Python 3.9 (socket.timeout is not a subclass of TimeoutError); now catches socket.timeout, OSError, and any unexpected exception type, with a defensive wrap around `fut.result()` in the driver loop.
- `parse_bacdive.py` pathogenicity extractors initially missed ~95% of positives because BacDive uses freeform strings like "yes, in single cases"; switched from exact-token matching to prefix matching. BSL-1 → derived-negative post-processing added so the heads have proper binary labels rather than positive-only.
- `vocab.py` initially mis-handled `cultivation_medium` (size 0) because parquet roundtrip turns Python lists into numpy ndarrays; now treats any non-string non-dict iterable as a list.
- `model.py` multilabel mask for list-typed heads originally only marked observed positives, yielding inflated F1 with random features; switched to implicit-negative assumption for list-typed heads.
- `model.py` `--run-name` was silently ignored when `--features` was absent (operator precedence in the fallback expression).

### Known limitations

- Polar lipid and respiratory quinone heads not in v1: BacDive's structured fields don't include them; would need IJSEM PDF mining.
- Pathogenicity binary uses BSL-1 as a negative-class proxy (standard convention but imperfect).
- Label noise across BacDive references is unmeasured; some traits with multiple disagreeing entries per strain are resolved by majority vote.
