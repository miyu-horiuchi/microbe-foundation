# Per-Protein ESM-2 Embeddings + Attention Pooling

**Date:** 2026-06-02
**Status:** Draft for review
**Topic:** Replace genome-level mean-pooling of ESM-2 embeddings with un-pooled per-protein
embeddings consumed by a learned attention-pooling head.

---

## Goal

Test the hypothesis that **keeping each protein distinct** — and letting the model learn which
proteins matter for each trait — beats the current mean-pooled baseline on the family-held-out
benchmark.

Today the pipeline collapses each genome to a single `[640]` vector by averaging ~50 randomly
sampled proteins. That averaging is lossy: trait-determining proteins (e.g. a catalase) are
diluted into thousands of unrelated ones. This project keeps the **full proteome un-pooled**
(`[n_proteins, 640]` per genome) and adds an **attention-pooling** layer to the model that learns
a weighted combination over proteins before the trait heads.

This is a lightweight, ESM-2-native analogue of Bacformer's "protein-of-proteins" formulation,
and slots directly into the existing leaderboard for apples-to-apples comparison against
`esm2-150M-*` mean-pool rows.

## Non-goals

- **Not** residue-level (within-protein) embeddings. Each protein stays residue-mean-pooled to a
  single `[640]` vector; we only stop pooling *across* proteins. (True residue-level deferred —
  storage/compute infeasible, lower marginal value for genome-level labels.)
- **Not** end-to-end ESM-2 fine-tuning. The encoder stays frozen; we cache embeddings and train
  only the pooling + heads. (Deferred Phase-4 idea.)
- **Not** an encoder-size change. Held at `facebook/esm2_t30_150M_UR50D` (640-dim) to isolate the
  pooling variable. Bumping to 650M/1280 is a separate follow-up.
- **Not** protein deduplication. All proteins kept as-is (option (a)); dedup-to-unique deferred.

## Background — current state (verified against code)

- `microbe_model/features/embeddings.py`
  - `embed_proteins(...)` → returns un-pooled `[n_proteins, 640]` (residue-mean-pooled per protein). **Already exists, reusable as-is.**
  - `embed_genome(...)` → calls `embed_proteins` then `matrix.mean(axis=0)` at **line 113**. This final mean is what we drop.
  - `sample_n` (default 50) uniformly subsamples proteins. We set it to **None** (full proteome).
  - `ESM2_MAX_LEN = 1024` — proteins longer than 1024 residues are truncated. Unchanged.
- `compute_esm2_features_mp.py`
  - ProcessPool workers fetch FASTA + pyrodigal-predict CDS; main process runs ESM-2 on GPU.
  - Saves a single `data/esm2_features.npz` with `bacdive_ids [N]`, `features [N,640]`, `accessions [N]`.
  - Resumable via `load_existing` (skips ids already present).
- All 13 saved `.npz` feature files are 2-D (pooled). **No un-pooled data exists anywhere** → a
  re-extraction is mandatory.
- `model.py` — shared-MLP encoder + per-trait linear heads, consumes `[N, D]` from one `.npz`.

## Design

### 1. Extraction (`compute_esm2_features_mp.py`, run on Lambda H100)

Change behavior to emit **per-genome files** instead of one stacked matrix:

- New flag `--per-protein` (or a new script `compute_esm2_perprotein_mp.py` to avoid regressing the
  pooled path — **decision below**). When set:
  - `sample_n = None` → embed **every** predicted protein.
  - Replace `embed_genome(...)` with a direct `embed_proteins(...)` call → keep `[n_proteins, 640]`.
  - Write one file per genome: `data/esm2_perprotein/<bacdive_id>.npy` (float16 to halve storage;
    embeddings are fp16 on GPU anyway).
  - Append a row to a manifest `data/esm2_perprotein/manifest.parquet`:
    `bacdive_id, accession, n_proteins, path, status`.
- **Resumability:** skip genomes whose `.npy` already exists and is non-empty (mirrors the eggNOG
  self-healing pattern). Checkpoint = the files themselves + manifest append.
- **Truncation guard:** genomes with very large proteomes (>N proteins, e.g. 12,000) are still
  written whole; the model loader caps per-batch length (see §3). Log any genome exceeding the cap.

### 2. Storage format

- ~19,600 genomes × ~4,000 proteins × 640 dims × **2 bytes (fp16)** ≈ **~100 GB**.
- Per-genome `.npy` (ragged-friendly, lazy-loadable, resumable) + a `manifest.parquet` index.
- Rationale over a single `.npz`: a ~100 GB monolith can't be loaded into RAM, isn't resumable
  mid-write, and npz has no ragged support. Per-genome files load lazily in the DataLoader.

### 3. Model (`model.py`)

Add an **attention-pooling** module between the (new) per-protein input and the existing heads:

- Input per genome: `[P, 640]` protein matrix + a length (P varies per genome).
- **Attention pool:** a learned query vector `q [640]` (or a small MLP scorer); scores
  `s_i = q · proj(x_i)`; `softmax` over the P proteins (with padding mask); weighted sum →
  genome vector `[640]`. ~640–2k new params. (Multi-head / per-trait attention is a later refinement.)
- The `[640]` genome vector feeds the **existing** shared-MLP encoder + per-trait heads unchanged.
  All masked-loss logic (binary/multiclass/multilabel/regression-vector) is untouched.
- **DataLoader:** reads per-genome `.npy` via the manifest, pads to the batch's max P, builds a
  padding mask. Cap P at a max (e.g. 6,000) by random/first-N selection *only* if memory forces it;
  default is no cap. Memory-map or load-on-access so RAM stays bounded.
- **Backward-compat:** the existing 2-D `.npz` path stays working. The model detects input kind
  (2-D npz vs per-protein manifest) and only inserts the attention-pool layer for the latter.

### 4. Training & evaluation

- Same `splits.parquet` (family/genus/species, seed 42). No leakage-surface change.
- Train with the existing `model.py` loop + the new pooling layer.
- Emit `runs/esm2-150M-attnpool-{family,genus,species}.json` in the standard schema.
- `leaderboard.py` picks them up automatically; they rank beside `esm2-150M-*` mean-pool rows.
- **Primary success metric:** beat `esm2-150M-family` (mean-pool) on mean per-head rank, family split.

## Cost

| | Pooled baseline (done) | This project |
|---|---|---|
| Proteins embedded | ~1M (50/genome) | ~78M (full proteome) |
| GPU wall, 1× H100 | ~100 min | ~130 hrs |
| GPU wall, 8× H100 | — | ~16–17 hrs |
| GPU $ (total) | ~$10 | ~$400–800 |
| Storage | 50 MB | ~100 GB (fp16) |

NCBI fetch + pyrodigal cost is unchanged (same 19,637 genomes); the GPU embedding is now the
dominant term (~80× more forward passes).

## Risks & mitigations

- **Overfitting / no gain.** More signal ≠ better generalization on ~16k training genomes
  (cf. `hybrid_v3` dim-26454 underperforming). *Mitigation:* attention pooling is parameter-
  efficient (learns weights, not raw dims); keep encoder at 640; this run is itself the cheap test
  of whether granularity helps before escalating.
- **Memory blowup in the loader.** Large proteomes × batch. *Mitigation:* fp16, lazy per-genome
  load, padding mask, optional P cap, modest batch size.
- **Extraction cost wasted if it loses.** *Mitigation:* a smoke run on ~200 genomes first to
  validate format + a quick attention-pool fit, before committing the full multi-GPU run.
- **Truncation of long proteins (>1024 aa).** Pre-existing behavior, accepted; logged.

## Testing

- Unit: attention pool output shape `[B, 640]`; padding mask zeroes absent proteins; gradients flow;
  softmax weights sum to 1 over real (unmasked) proteins.
- Unit: DataLoader pads ragged batch correctly; manifest round-trips; missing-file handling.
- Integration: end-to-end train on the ~200-genome smoke set → metrics JSON in correct schema.
- Regression: existing 2-D `.npz` mean-pool path still trains and scores unchanged.

## Open decisions (resolve before implementation plan)

1. **New flag on existing script vs. new script.** Recommendation: **new script**
   `compute_esm2_perprotein_mp.py` (clean output format, no risk to the proven pooled path).
2. **Smoke-first.** Recommendation: yes — ~200-genome extraction + attention-pool fit before the
   full Lambda run.
3. **P cap.** Recommendation: no cap by default; add only if loader memory forces it.

## Decisions locked

- Full proteome, un-pooled per protein. ✅
- Encoder held at ESM-2 150M / 640-dim. ✅
- Run extraction on Lambda H100 (multi-GPU to bound wall time). ✅
- Attention pooling over the protein set; genome-level labels; existing heads/splits/leaderboard. ✅
- Cost ~$400–800 accepted. ✅
