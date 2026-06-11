# Embedding-Based Drift Monitor & Geometric Verifier — Design

**Date:** 2026-06-11
**Status:** Implemented & validated (Tier 0–1; Tiers 2–4 designed, deferred)
**Home:** `microbe_model/monitoring/` (microbe-foundation research repo)

## Result (go/no-go)

Implemented and merged. Family-split validation on `data/esm2_features.npz`
(reference = held-out genomes of train families; in-distribution negatives = unseen
strains of train families; positives = test/novel families):

| Backend | AUROC |
|---|---|
| **Euclidean k-NN (default)** | **0.757** |
| Diffusion (10 components) | 0.694 |
| Diffusion (100 components) | 0.728 |

**Decision: keep the Euclidean backend** (already the `ReferenceManifold` default). The
diffusion map's dimensionality reduction discards discriminative variance that plain
k-NN distance in standardized space retains; it does not beat Euclidean even when
tuned. The curvature-aware hypothesis was worth testing and is honestly rejected on the
data. `DiffusionBackend` remains in the codebase as a tested, pluggable alternative.

Methodological note discovered while running: the family split assigns whole families
to train/val/test, so **val families are also novel relative to train** — val genomes
must NOT be used as in-distribution negatives. Negatives must be held-out genomes of
*train* families. The harness was corrected accordingly.

## Motivation

The microbial-function-discovery system predicts useful biological functions from
genomes (assembled genomes, MAGs, SAGs, protein FASTA) to surface uncultured
organisms worth taking to the wet lab. Its dominant failure mode is **covariate
shift**: novel organisms occupy regions of ESM2 sequence space unlike the training
distribution, and predictions there are unreliable. This is the predictability-gradient
thesis made operational — covariate shift is the real frontier.

Shifts in the ESM2 embedding distribution can reveal that the model is encountering
novel biological regions of sequence space **before any labels arrive**, providing the
earliest warning of performance degradation.

### Drift taxonomy (governing principle)

- **Data drift** — the inputs changed: P(X) shifts. A *monitoring problem first*.
  Characterize the new distribution, measure OOD rates, and determine whether
  performance is actually affected before retraining.
- **Concept drift** — the input→label relationship changed: P(Y|X) shifts. A
  *model-update problem first*. Requires collecting fresh labels and re-evaluating,
  because the learned mapping may no longer be valid.

**Hard truth this design respects:** concept drift is invisible in embeddings. P(Y|X)
changing leaves P(X) untouched, so the geometric monitor stays silent. Embedding
monitoring buys early warning on *data drift only*; concept drift requires labels.

## Scope

**In scope (this spec):**
- Tier 0 — geometric, label-free embedding monitor.
- Tier 1 — drift classifier that routes on Tier 0 signals.
- Validation harness proving the geometry against the family split.

**Designed, deferred to a later spec:**
- Tier 2 — weighted-conformal verifier gate.
- Tiers 3–4 — label-prioritization loop and performance-proxy tracking.

## Substrate (already present)

- `data/esm2_features.npz` — genome-level embeddings: `features (19608, 640)`,
  `bacdive_ids (19608,)`, `accessions (19608,)`.
- `data/splits.parquet` (from `splits.py`) — family / genus / species held-out splits.
  **Family-level held-out clades are the novel-sequence-space proxy**: train families
  form the reference manifold; held-out families are genuine OOD arrivals.
- `paper/tables/13_matched_clade_controls.md` and existing `runs/` outputs document the
  performance collapse on the family split — real degradation to validate against, not
  synthetic drift.

## Why the geometry (and where it is overkill)

ESM2 embeddings are 640-d Euclidean vectors, but the biologically meaningful structure
is a low-dimensional **curved manifold** inside that space. The failure mode of naïve
Euclidean drift detection: a novel genome can be Euclidean-close but **geodesically far**
(off-manifold), and off-manifold is exactly OOD.

**Engineering decision:** do *not* do literal continuous Riemannian geometry (estimating
a metric tensor and shooting geodesics in 640-d with ~19k samples is noisy, expensive,
brittle). Use the tractable discretization — the **graph Laplacian and its heat kernel**,
which *is* the discrete manifold (Coifman–Lafon diffusion maps). The continuous Riemannian
framing is the conceptual justification; the graph Laplacian is the computation.

This is a hypothesis to be proven, not assumed: the diffusion backend only wins
measurably when the embedding manifold is strongly curved. The validation harness must
demonstrate it beats the Euclidean baseline, or we honestly keep the simpler kernel.

## Architecture — `microbe_model/monitoring/`

Four focused, independently testable units.

### `reference.py` — `ReferenceManifold`

Fit on the reference embedding matrix (train-family genomes).

Responsibilities:
- Standardize embeddings (store mean/scale).
- Build a sparse k-NN affinity graph; normalized graph Laplacian.
- Compute diffusion-map coordinates: top-k eigenvectors of the Laplacian via
  `scipy.sparse.linalg.eigsh`.
- Store kernel bandwidth via the median heuristic.

Interface:
- `fit(X_ref) -> self`
- `ood_score(X) -> np.ndarray` — per-genome score, higher = more out-of-distribution.
- `save(path)` / `load(path)` — persist fitted state (no refit per batch).

Out-of-sample scoring: new genomes embed into diffusion coordinates via **Nyström
extension** — no refit when a batch arrives.

**Pluggable distance backend** (same interface, so the experiment decides the winner):
- `EuclideanBackend` — k-NN distance / Mahalanobis in standardized space (baseline).
- `DiffusionBackend` — heat-kernel diffusion distance to the reference support (contender).

### `mmd.py` — `mmd_permutation_test(X_ref, X_test, kernel, n_perm)`

Pure function returning `(mmd2, p_value)`. The permutation test yields a **valid p-value
with controlled type-I error** — the population-level rigorous verifier.

- Kernels: `rbf` (median-heuristic bandwidth, baseline) and `heat`/diffusion (contender).
- Use a characteristic kernel so MMD is a proper metric on distributions.

### `drift.py` — `DriftReport`

Combines:
- Population signal: MMD permutation p-value.
- Per-genome signal: OOD rate = fraction of batch exceeding a reference-calibrated
  quantile threshold on `ood_score`.

Classifies `{no_drift, data_drift}`. On `data_drift`, emits a recommendation —
*characterize the new distribution → check performance proxies → decide whether to
retrain* — rather than auto-triggering retraining. Concept drift is explicitly reported
as out of embedding scope, with a "fresh labels required" note when proxies degrade.

### `validate.py` — proof harness (go/no-go for the geometry)

Setup: reference = train families; positives = held-out families (novel space);
negatives = held-in validation genomes.

Reports:
1. **AUROC** of `ood_score` separating in- vs out-of-clade genomes.
2. **Correlation** of per-genome `ood_score` with model error on the family split
   (using existing `runs/` outputs).

Run for both backends and both kernels.

## Data flow (Tier 0–1)

```
esm2_features.npz + splits.parquet
  → ReferenceManifold.fit(train-family embeddings)
incoming batch (genome embeddings)
  → standardize → ood_score (per-genome) + mmd_permutation_test (batch p-value)
  → DriftReport: {no_drift | data_drift} + recommendation
validation:
  → validate.py: AUROC + ood-vs-error correlation, Euclidean vs Diffusion
```

## Deferred interfaces (Tiers 2–4)

- **Tier 2 `verifier.py`** — split/weighted conformal prediction. Importance weights from
  the OOD likelihood ratio p_test(x)/p_train(x); high-OOD genomes get widened or
  abstaining intervals. Documented catch: standard conformal assumes exchangeability,
  which breaks precisely under shift, so the gate is OOD-weighted and the monitor tells
  the verifier how much to distrust its own interval.
- **Tier 3 — label-prioritization hook** — rank labeling budget by `ood_score` (novel
  manifold regions first); the only path that addresses concept drift.
- **Tier 4 — performance-proxy tracking** — head disagreement, confidence collapse,
  calibration drift as weak concept-drift signals that *suggest* (never confirm) the need
  to acquire labels.

## Testing

- `mmd`: type-I correctness (null: same distribution → ~uniform p-values across repeated
  draws); power (shifted Gaussians → low p-values).
- `reference`: `ood_score` monotonic with injected distance from the reference;
  Nyström out-of-sample round-trip; `save`/`load` round-trip preserves scores.
- Integration: tiny synthetic embedding fixture end-to-end through `DriftReport`.

## Success criterion (this phase)

The diffusion backend achieves **higher AUROC** at flagging held-out-family genomes
**and stronger correlation with model error** than the Euclidean baseline — or we
honestly report it does not and retain the simpler Euclidean/RBF path. Either outcome is
a valid result; the harness exists to decide it on data, not assumption.
