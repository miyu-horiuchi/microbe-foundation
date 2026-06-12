# Cross-Clade Collapse Diagnostic — Design

**Date:** 2026-06-12
**Status:** Approved
**Home:** `cross_clade_diagnostic.py` (microbe-foundation repo root, mirrors `ood_error_analysis.py`)

## Motivation

The predictability-gradient paper's central negative result: the attention-pooling
advantage collapses at family-level holdout, and cross-clade generalization — not
pooling — is the binding constraint for genomic trait models on uncultured organisms.
Nothing yet *fixes* this; the drift monitor only *detects* the regime.

**Key constraint.** The genome representation is frozen precomputed ESM-2 (mean-pooled
640-d); `model.py` trains only a small head on top. So representation-level fixes
(domain-adversarial encoders, etc.) have a low ceiling — you cannot learn a
clade-invariant encoder that is never trained. Before investing in the real lever
(end-to-end encoder fine-tuning, a large GPU project), we must know **why** family
transfer collapses: is it insufficient training-clade *coverage* (fixable without
fine-tuning) or a *representation wall* (the frozen embedding does not carry the trait
across clades, so only fine-tuning can help)?

This spec is a cheap, falsifiable **diagnostic** that returns that verdict. It does not
attempt a fix.

## Scope

Two complementary diagnostics, CPU-only, reusing existing data (`data/esm2_features.npz`,
`data/splits.parquet`, `data/traits.parquet`) and a linear probe. No new training
architecture, no encoder changes. Produces a committed results table, a figure, and a
written verdict.

## Diagnostic A — Train-family-diversity curve

For a trait, restrict the training set to `k` distinct families (drawn deterministically),
train a `LogisticRegression` probe on those genomes, and evaluate macro-F1 on the
**fixed, full** family-test set. Sweep `k ∈ {5, 10, 20, 40, 80, all}`; average over 3
subsampling seeds per `k` (report mean ± std).

Interpretation:
- **Rising and still climbing at all-families** → coverage-limited: the representation
  transfers given enough clade diversity; acquiring more training families would help.
- **Plateaus early / stays low** → adding clades does not help → representation wall.

Implementation notes:
- "Distinct families" come from the `family` column of `traits.parquet`, restricted to
  genomes in the family-train split that have both an embedding and a non-null trait
  label and that belong to a family with ≥2 labeled classes available overall.
- If a sampled `k`-family subset is single-class, resample with the next seed offset;
  if still single-class after a few tries, skip that `(k, seed)` and note it.
- `k = all` uses every training family (the standard baseline point).

## Diagnostic B — Cross-clade k-NN label transfer

For each novel-family (family-test) genome, take its `k_nn = 10` nearest **training-family**
genomes in standardized 640-d embedding space and predict the trait by neighbour-label
majority (probability = fraction of positive neighbours). Report macro-F1 and AUROC, and
compare to the trained probe's family-test F1/AUROC.

Interpretation:
- **k-NN transfers well (≫ chance, near probe)** → embedding geometry carries the trait
  across clades; a better head/retrieval can exploit it (no fine-tuning strictly needed).
- **k-NN ≈ chance** → cross-clade neighbours do not share the label → representation wall.

Implementation notes:
- Reuse standardized-space k-NN (the same machinery as `EuclideanBackend`); neighbours are
  drawn only from family-train genomes (no leakage from test families).
- Chance baseline = predicting the test-set positive rate; report it alongside.

## Triangulated verdict

| A (diversity) | B (k-NN) | Conclusion | Action |
|---|---|---|---|
| rising | good | coverage problem | acquire more training families; no fine-tuning needed |
| flat | poor | representation wall | fine-tune the encoder (justifies that project) |
| mixed | mixed | partial | targeted follow-up |

## Traits

- `sporulation` (machinery, sharpest + most learnable collapse) — primary.
- `motility` (machinery) — second.
- `catalase` (compositional) — control.

Binary-head materialization matches `ood_error_analysis.py`: `y = float(bool(value))`,
mask = not-null.

## Components

- `cross_clade_diagnostic.py` (root):
  - `family_diversity_curve(...)` → list of `{trait, k_families, seed, n_train, test_f1}`.
  - `knn_transfer(...)` → `{trait, knn_f1, knn_auroc, probe_f1, probe_auroc, chance_f1}`.
  - pure helpers: `binary_trait_labels` (reuse/duplicate the tested one), `sample_families(families, k, seed)`, `knn_majority_predict(X_ref, y_ref, X_query, k)`.
  - `run(features, splits, traits, traits_list)` orchestrates both, writes outputs.
  - `__main__` CLI: `--features --splits --traits --out-dir paper/tables --fig-dir paper/figures`.
- Outputs:
  - `paper/tables/15_cross_clade_diagnostic.{md,csv}` (both diagnostics + the verdict line).
  - `paper/figures/cross_clade_diversity_curve.png` (test-F1 vs #families, one line per trait;
    matplotlib, Agg backend).
- Reuse: `microbe_model.monitoring` standardization/k-NN where natural; `scikit-learn`
  `LogisticRegression`/`roc_auc_score`/`f1_score`.

## Testing (`tests/test_cross_clade_diagnostic.py`, pytest plain-assert)

- `sample_families`: returns exactly `k` distinct families, deterministic per seed,
  subset of input.
- `knn_majority_predict`: a query identical to all-positive neighbours → prob 1.0; a
  query nearest to all-negative neighbours → prob 0.0; output shape matches query count.
- `binary_trait_labels`: null handling + truthiness (as in the OOD test).
- Synthetic end-to-end: a constructed separable dataset where k-NN transfer F1 > chance,
  and a diversity-curve call returns one row per `(k, seed)` with F1 in `[0, 1]`.

## Success criterion

The run produces a table + figure that yield an unambiguous verdict
(coverage-limited vs representation-wall vs mixed) for at least the primary trait
(sporulation), with the chance baseline and probe baseline reported so the k-NN and
diversity numbers are interpretable. The verdict decides whether encoder fine-tuning is
the warranted next investment.
