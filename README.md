# microbe-foundation

**Reading the genome of an organism no one has ever grown, and predicting how to grow it.**

Most microbial life is *dark matter*: an estimated **99% of microbial species have never been cultured** in a lab, so we know almost nothing about them — what they need to grow, what they do, or what they could make. Yet their genomes are pouring in by the millions (metagenome-assembled and single-cell genomes). microbe-foundation is a machine-learning effort to **turn those genomes into actionable biology**: predict an organism's complete species description *and* the cultivation conditions likely to bring it into culture, so labs can prioritize the few worth attempting instead of blindly trying thousands.

> **One line:** a genome-conditioned, multi-task model that predicts a microbe's traits and its likely **cultivation medium**, used as a screening tool to triage which dark-matter organisms to try to culture — with a roadmap toward a closed active-learning loop that gets better with every wet-lab result.

**Status:** work-in-progress research. The benchmark, schema, multi-task model, and paper draft are reproducible end-to-end today. The per-protein encoder, hybrid features, generative recipe head, and active-learning loop are in progress / planned — clearly marked as such throughout (✅ built · 🚧 in progress · 📋 planned).

---

## Why this matters

- **The unculturable 99%.** The vast majority of microbes resist cultivation because we don't know their growth requirements. Locked inside that dark matter is most of nature's undiscovered antibiotics, enzymes, and metabolic chemistry.
- **Cultivation is blind and expensive.** Today, bringing a new organism into culture is trial-and-error over a huge space of media and conditions. A model that *ranks candidates and proposes a recipe* converts a blind search into a prioritized shortlist.
- **The genomes already exist.** We don't lack sequence — we lack the map from sequence → phenotype → cultivation conditions for organisms unlike anything previously characterized. That map is the ML problem.

## What it does

1. **Predicts the full species description from genome** — morphology, physiology, growth conditions, cultivation media, biosafety, ecology, and chemotaxonomy (21 prediction heads; see below). ✅
2. **Predicts cultivation requirements** — the medium an organism is likely to grow in, framed as a screening/ranking problem with calibrated confidence rather than a single point guess. ✅ (medium-as-class today) · 📋 (medium-as-recipe next)
3. **(Roadmap) Proposes novel recipes for novel organisms** — a generative recipe head that composes media (ingredients + concentrations) for genomes unlike anything in training, grounded in genome-derived auxotrophy signals. 📋
4. **(Roadmap) Improves itself from experiments** — a Bayesian-optimization / active-learning loop where each wet-lab result is a new label that shrinks the gap between "organisms we could culture" and "organisms we couldn't." 📋

## The scientific thesis: a predictability gradient

Not all traits are predicted the same way. Our working hypothesis — the spine of the modeling work — is that traits live on a **predictability gradient**:

- **Compositional traits** (GC-correlated properties, bulk membrane/cell-wall features) are diffuse signals spread across the whole genome → **mean-pooling** the protein representations is correct.
- **Machinery traits** (motility, specific metabolic capabilities, *nutrient requirements*) are decided by a handful of decisive genes → **attention-pooling** is correct, because averaging dilutes the signal.

Cultivation requirements sit at the *machinery* end: an organism's need for an exogenous amino acid is determined by whether ~8 specific biosynthesis genes are present or absent, not by a genome-wide average. This is why the encoder moves from mean-pool toward **per-protein + attention pooling**, and the gradient is directly testable inside one architecture.

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

## Contributions (what's defensible today)

| | Contribution | Why it's defensible |
|---|---|---|
| 1 | **Full BacDive species-description coverage.** 21 heads across all 7 biological blocks. | MicroGenomer covers ecophysiology only; BacBench has no chemotaxonomy / medium / pathogenicity; BacPT targets metabolic/ecological. None match scope. |
| 2 | **First chemotaxonomy-from-genome predictor (FAME).** | Live literature search (2026-05-28) confirmed zero prior papers. |
| 3 | **Family-held-out splits.** Strictly harder than BacBench's genus-only splits. | Tests cross-family generalization, not within-family memorization — a proxy for the cultured→uncultured shift. |
| 4 | **Masked multi-task loss** over BacDive's heavily sparse labels (5–95% coverage per head). | None of the three foundation-model competitors explicitly handles label sparsity this way. |

---

## How it works (the process)

### 1. Data — labels from the cultured minority

```
BacDive REST API  --> data/bacdive_raw.jsonl          (fetch_bacdive.py)        ✅
                  --> data/traits.parquet              (parse_bacdive.py)         ✅
                  --> data/splits.parquet              (splits.py, family-held-out)✅
                  --> data/vocabularies.json           (vocab.py)                 ✅
                  --> data/genome_accessions.tsv       (extract_genome_accessions.py) ✅
MediaDive         --> cultivation-medium recipes joined to BacDive strains       ✅
```

BacDive provides measured traits for the cultured minority; MediaDive provides the medium each strain grows in. These are the supervision for genome → trait and genome → medium.

### 2. Features — two complementary views of a genome

A genome enters as its set of proteins (predicted with pyrodigal), then is encoded two ways that are designed to be **fused**:

```
proteins --> ESM-2 mean-pool per genome          (compute_esm2_features.py)        ✅ baseline
         --> ESM-2 per-protein embeddings         (modal_esm2_perprotein.py,         🚧 in progress
                                                    compute_esm2_perprotein_mp.py,
                                                    embed_from_cache.py)
         --> eggNOG / KEGG pathway-completeness    (compute_eggnog_features.py,       ✅
                                                    modal_eggnog.py)
```

- **Per-protein ESM-2 embeddings** capture *novel, unannotated* proteins — the dark-matter case where annotation fails.
- **eggNOG / KEGG pathway-completeness** gives a *clean auxotrophy signal* — "is the lysine-biosynthesis module complete?" maps almost directly to "does the medium need lysine?"

The two views are complementary: embeddings see what annotation misses; annotation grounds what embeddings can't easily express.

### 3. Model — multi-task, moving toward a cross-attentive recipe decoder

```
model.py  --> 21-head multi-task model + masked loss (mean-pool features)         ✅
          --> attention pooling over per-protein embeddings                        🚧 (esm2-perprotein-attnpool)
          --> hybrid encoder: ESM-2 protein tokens + eggNOG pathway tokens         📋
          --> generative recipe head (ingredients + concentrations)                📋
```

**Planned architecture (design captured in `docs/`):** a Set-Transformer / Perceiver genome encoder over frozen per-protein ESM-2 embeddings *and* eggNOG pathway tokens, with **ingredient queries that cross-attend to the protein/pathway set** — so each predicted ingredient is justified by the specific genes that imply it (interpretable, auxotrophy-grounded). A parallel mean-pool path serves compositional traits, making the predictability gradient testable in one model.

### 4. Evaluation — screening-shaped, not accuracy-shaped

The deliverable is a *ranked, calibrated shortlist*, so the metrics are precision@k / cultivation-attempts-saved and **calibration under phylogenetic shift**, not raw accuracy. The model must beat three honest baselines under a strict family/clade holdout:

1. **16S nearest-neighbor** ("copy the medium of the closest cultured relative") — a deceptively strong baseline.
2. **Genome-scale metabolic model** (gapseq / CarveMe + flux-balance analysis) — the established mechanistic approach.
3. **eggNOG-only logistic regression** — to prove the learned ESM-2 representation adds value over gene-content alone.

### 5. (Roadmap) The active-learning loop — a self-improving culturomics engine 📋

Framed as **Bayesian optimization**, not RL-against-a-simulator:

- **Surrogate** = the genome-conditioned model with calibrated uncertainty (deep ensemble), scoring `(genome, medium) → P(growth) ± σ`.
- **Acquisition function** = ranks candidate experiments by `P(growth) × information-gain × novelty` to choose what to try next.
- **Oracle** = the wet lab (ground truth). Optionally a **multi-fidelity** setup uses cheap in-silico FBA as a low-fidelity pre-filter and reserves wet-lab budget for confirmation — dry lab as a prior, never as truth.
- **Flywheel** = each result becomes a new label, progressively closing the cultured→uncultured covariate gap.

---

## Reproduce (what runs today)

### Requirements

- Python 3.11+ recommended (3.9 works for everything except ESM-2 feature extraction)
- Genomes stream from NCBI in memory — no large local genome store required
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
    --sample-n 50 --batch-size 16                        # mean-pool baseline
python compute_eggnog_features.py                        # pathway-completeness view
# per-protein extraction (large GPU jobs): see modal_esm2_perprotein.py
#   and docs/RUNNING_PERPROTEIN_ON_LAMBDA.md

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
├── compute_esm2_perprotein_mp.py     # ESM-2 per-protein (multi-GPU, sharded)
├── modal_esm2_perprotein.py          # per-protein extraction on Modal
├── embed_from_cache.py               # embed Modal-cached proteins on a GPU box
├── compute_eggnog_features.py        # eggNOG / KEGG pathway-completeness
├── modal_eggnog.py                   # eggNOG annotation on Modal
├── compute_bacformer_features.py     # Bacformer alternative (scaffold)
├── model.py                          # multi-task model + masked loss
├── compare_to_priors.py              # side-by-side vs published priors
├── prior_numbers.json                # curated prior-work scores
├── microbe_model/                    # vendored from microbe-model v0
│   ├── pipeline.py                   #   in-memory NCBI fetch
│   ├── features/                     #   pyrodigal, ESM-2, KEGG, markers
│   └── data/                         #   BacDive, MediaDive clients
├── reference_data/                   # precomputed MediaDive catalogs
├── docs/                             # design notes + GPU runbooks
├── paper/                            # draft + auto-generated tables
├── scripts/run_all.sh                # one-command pipeline
├── RELATED_WORK.md                   # positioning vs prior work
├── BACBENCH_SCOPE.md                 # BacBench competitor analysis
├── BACDIVE_COVERAGE.md               # per-trait label coverage audit
└── CITATION_AUDIT.md                 # citation verification + threat assessment
```

## Where this fits among prior work

Three 2025–2026 preprints (MicroGenomer, Bacformer, BacPT) claimed the foundation-model framing for microbial genomes; this repo is **not** trying to be a better encoder. Two things distinguish it:

1. **The benchmark + chemotaxonomy white-space (built).** A unified 21-head, family-held-out evaluation anyone can run by swapping in their encoder as `features.npz` — including the first genome-to-FAME predictor. See `RELATED_WORK.md`.
2. **The cultivation-screening direction (the vision).** Reframing the foundation model as a tool that *prioritizes which dark-matter organisms to culture and proposes how* — distinct from purely descriptive trait prediction, and from mechanistic metabolic models (which fail exactly where genomes are novel/incomplete).

## Built on

- **[microbe-model v0](https://github.com/miyu-horiuchi/microbe-model)** — the single-task cultivation-medium predictor that established the BacDive + NCBI Datasets + pyrodigal + ESM-2 pipeline. Pipeline modules vendored under `microbe_model/`.
- **[BacDive](https://bacdive.dsmz.de)** — primary label source (CC-BY 4.0, free public REST API).
- **[MediaDive](https://mediadive.dsmz.de)** — cultivation medium recipes joined to BacDive strains.
- **[ESM-2](https://github.com/facebookresearch/esm)** — protein language model (frozen) providing per-protein representations.
- **[eggNOG-mapper](http://eggnog-mapper.embl.de)** — functional annotation / KEGG pathway-completeness.
- **[Bacformer / BacBench](https://github.com/macwiatrak/Bacformer)** — closest peer benchmark; cited as direct comparator.

## Tests

A pytest suite at `tests/` covers schema invariants (21 heads, 7 blocks, FAME head never dropped), parser correctness, split correctness (no group spans buckets, hits target ratios), and model construction (heads match schema, masked loss flows gradients).

```bash
pip install pytest
python -m pytest tests/ -v
```

Run before any PR touching `schema.py`, `parse_bacdive.py`, `splits.py`, or `model.py`. GitHub Actions runs the same suite on every push to main via `.github/workflows/test.yml`.

## License

Code: MIT. Data: BacDive and MediaDive content are CC-BY 4.0 (cite Schober et al. 2025 in any derivative).

## Citation

If you use this benchmark, please cite the eventual paper (currently a draft in `paper/paper.md`). Until then:

```
Horiuchi, M. (2026). microbe-foundation: predicting microbial species
descriptions and cultivation requirements from genome.
https://github.com/miyu-horiuchi/microbe-foundation
```
