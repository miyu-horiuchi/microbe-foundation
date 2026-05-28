# microbe-foundation

A unified benchmark and multi-task baseline for predicting the **complete microbial species description from genome sequence** — morphology, physiology, growth conditions, cultivation media, biosafety, ecology, and **chemotaxonomy** (the last of which has no prior published predictor).

Status: work-in-progress. The data pipeline, schema, multi-task model, and paper draft are all in the repository. Final results pending GPU feature extraction; the codebase is reproducible end-to-end today.

---

## What it predicts

**21 prediction heads** across 7 biological blocks. See `paper/tables/01_trait_inventory.md` for the full inventory regenerated from `trait_schema.json`.

| Block | Heads |
|---|---|
| Morphology | gram stain, cell shape, motility, sporulation, pigmentation |
| Physiology | oxygen tolerance, catalase, cytochrome oxidase, halophily |
| Growth conditions | temperature class, pH class |
| Cultivation | cultivation medium (MediaDive-linked), carbon utilization, metabolite production, AMR phenotype |
| Safety | biosafety level, pathogenicity (human), pathogenicity (animal) |
| Ecology | isolation source, country of isolation |
| Chemotaxonomy | fatty-acid profile (FAME) — **first genome-to-FAME predictor in the literature** |

## The four contributions

| | Contribution | Why it's defensible |
|---|---|---|
| 1 | **Full BacDive species-description coverage.** 21 heads across all 7 biological blocks. | MicroGenomer covers ecophysiology only; BacBench has no chemotaxonomy / medium / pathogenicity; BacPT targets metabolic/ecological. None match scope. |
| 2 | **First chemotaxonomy-from-genome predictor (FAME).** | Live literature search (2026-05-28) confirmed zero prior papers. |
| 3 | **Family-held-out splits.** Strictly harder than BacBench's genus-only splits. | Tests cross-family generalization, not within-family memorization. |
| 4 | **Masked multi-task loss** over BacDive's heavily sparse labels (5–95% coverage per head). | None of the three foundation-model competitors explicitly handles label sparsity this way. |

## Pipeline

```
BacDive REST API  --> data/bacdive_raw.jsonl          (fetch_bacdive.py)
                  --> data/traits.parquet              (parse_bacdive.py)
                  --> data/splits.parquet              (splits.py)
                  --> data/vocabularies.json           (vocab.py)
                  --> data/genome_accessions.tsv       (extract_genome_accessions.py)

NCBI Datasets API --> proteins via pyrodigal          (microbe_model/pipeline.py)
                  --> ESM-2 mean-pool embeddings       (compute_esm2_features.py)
                  --> data/esm2_features.npz

Multi-task model  --> runs/<name>.json                 (model.py --save-metrics)
                  --> paper/tables/06_vs_prior.md      (compare_to_priors.py)
                  --> paper/paper.md (full draft)
```

## Reproduce

### Requirements

- Python 3.11+ recommended (3.9 works for everything except ESM-2 feature extraction)
- ~150 GB disk-free not required — genomes stream from NCBI in memory
- GPU recommended for ESM-2 feature extraction at scale; CPU works for the smallest ESM-2 (8M params)

### One-command run

```bash
git clone https://github.com/miyu-horiuchi/microbe-foundation
cd microbe-foundation
pip install -r requirements.txt
bash scripts/run_all.sh                                  # full pipeline
bash scripts/run_all.sh --smoke                          # smoke test (~1000 strains)
```

### Step-by-step

```bash
# 1. Build benchmark
python fetch_bacdive.py                                  # ~2–3 hours, resumable
python parse_bacdive.py                                  # ~30 s
python splits.py                                         # ~2 s
python vocab.py                                          # ~5 s
python extract_genome_accessions.py                      # ~5 s

# 2. Compute features (GPU recommended)
python compute_esm2_features.py \
    --model facebook/esm2_t30_150M_UR50D \
    --sample-n 50 --batch-size 16

# 3. Train and evaluate
python model.py \
    --features data/esm2_features.npz \
    --split-level family --epochs 30 \
    --save-metrics runs/esm2_150M_family.json \
    --run-name esm2-150M

# 4. Refresh paper artifacts
python paper/generate_tables.py
python compare_to_priors.py --our runs/esm2_150M_family.json
```

## Repository layout

```
microbe-foundation/
├── schema.py + trait_schema.json     # 21-head schema (source of truth)
├── fetch_bacdive.py                  # bulk BacDive fetcher (stdlib, resumable)
├── parse_bacdive.py                  # per-trait extractors
├── splits.py                         # family/genus/species-held-out splits
├── vocab.py                          # data-derived class vocabularies
├── extract_genome_accessions.py      # NCBI accessions per strain
├── compute_esm2_features.py          # ESM-2 mean-pool per genome
├── compute_bacformer_features.py     # Bacformer alternative (scaffold)
├── model.py                          # multi-task model + masked loss
├── compare_to_priors.py              # side-by-side vs published priors
├── prior_numbers.json                # curated prior-work scores
├── microbe_model/                    # vendored from microbe-model v0
│   ├── pipeline.py                   #   in-memory NCBI fetch
│   ├── features/                     #   pyrodigal, ESM-2, KEGG, markers
│   └── data/                         #   BacDive, MediaDive clients
├── reference_data/                   # precomputed MediaDive catalogs
├── paper/
│   ├── paper.md                      # draft (skeleton + narrative)
│   ├── generate_tables.py            # auto-refresh from data files
│   └── tables/*.md                   # all auto-generated
├── scripts/
│   └── run_all.sh                    # one-command pipeline
├── requirements.txt
├── references.bib
├── RELATED_WORK.md                   # narrative draft
├── BACBENCH_SCOPE.md                 # BacBench competitor analysis
├── BACDIVE_COVERAGE.md               # per-trait label coverage audit
├── CITATION_AUDIT.md                 # citation verification + threat assessment
└── LIVE_SEARCH_2026-05-28.md         # browser-based prior-art search
```

## Why this exists

Three 2025–2026 preprints (MicroGenomer, Bacformer, BacPT) have claimed the foundation-model framing for microbial genomes. microbe-foundation is **not** trying to be a better encoder than those — it is the missing **benchmark** and **chemotaxonomy white-space**. Anyone (including the three competitor teams) can swap in their encoder as `features.npz` and report numbers on the same 21-head, family-held-out evaluation. The hope: anchor the next 5 years of microbial trait-prediction work through the benchmark rather than any single model. See `RELATED_WORK.md` for the full positioning argument.

## Tests

A pytest suite at `tests/` covers the schema invariants (21 heads, 7 blocks, FAME head never accidentally dropped), parser correctness on a synthetic record exercising all field shapes, split correctness (no group spans buckets, hits target ratios), and model construction (heads match schema, masked loss flows gradients).

```bash
pip install pytest
python -m pytest tests/ -v
```

45 tests, ~2 s on a laptop. Run before any PR touching `schema.py`, `parse_bacdive.py`, `splits.py`, or `model.py`. GitHub Actions runs the same suite on every push to main via `.github/workflows/test.yml` (Python 3.11 + 3.12).

## Submitting to the leaderboard

External submissions are welcome — anyone with a microbial genome encoder (KO, ESM-2, Bacformer, BacPT, Evo 2, your own) can produce a comparable run. See **`BENCHMARK.md`** for the protocol: which features are allowed, what splits to use, what to report, and how to submit a PR. Submissions are aggregated into `paper/tables/07_leaderboard.md` by `python leaderboard.py`.

## Built on

- **[microbe-model v0](https://github.com/miyu-horiuchi/microbe-model)** — the single-task cultivation-medium predictor that established the BacDive + NCBI Datasets + pyrodigal + ESM-2 pipeline. Pipeline modules vendored under `microbe_model/`.
- **[BacDive](https://bacdive.dsmz.de)** — primary label source (CC-BY 4.0, free public REST API).
- **[MediaDive](https://mediadive.dsmz.de)** — cultivation medium recipes joined to BacDive strains.
- **[Bacformer / BacBench](https://github.com/macwiatrak/Bacformer)** — closest peer benchmark; cited as direct comparator.

## License

Code: MIT.

Data: BacDive content is CC-BY 4.0 (cite Schober et al. 2025 in any derivative). MediaDive content is CC-BY 4.0.

## Citation

If you use this benchmark, please cite the eventual paper (currently in `paper/paper.md` as a draft). Until then:

```
Horiuchi, M. (2026). microbe-foundation: a unified benchmark for
predicting microbial species descriptions from genome.
https://github.com/miyu-horiuchi/microbe-foundation
```
