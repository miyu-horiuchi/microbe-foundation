# Design: Capacity ladder, extra-data fusion, and learned stacking for the baseline-regressor paper

Date: 2026-07-02
Paper: `paper/baseline_regressor_family_shift.md` ("Simple Readouts on Frozen Genome Embeddings…")
Status: approved (brainstorm), pending spec review

## Motivation

The existing Table 32 experiment (`paper/baseline_regressors.py`) already fits a
capacity ladder of simple downstream models on frozen ESM-2 embeddings, sweeps
PCA rank, evaluates per-target (binary traits + top cultivation media), and
combines models with a fixed soft-vote ensemble. This design adds the three
pieces the paper does not yet make explicit, motivated by the "Pfizer pattern"
framing: a heavy upstream model produces rich embeddings, and a light downstream
model (optionally fused with cheap extra data) does the task — with a principled
rule for when a simple readout is enough versus when to escalate to a bigger
model.

The three additions:

1. **Capacity ladder → plateau → escalate.** Formalize "start simple, grow the
   regressor until it plateaus, then you know you need a bigger model" as an
   explicit diagnostic with a plateau-detection rule and an escalation verdict.
2. **Embedding + extra-data fusion.** The literal Pfizer pattern: concatenate the
   frozen embedding with cheap tabular side-data and measure lift.
3. **Learned stack.** Replace the fixed soft-vote "combine at the end" with a
   trained meta-regressor and report whether it beats soft-vote.

## Non-goals

- Not training a transformer / attention pool / LoRA here. Piece 1 produces the
  *decision rule and its verdict*, not the escalated model. The escalation
  targets are exactly what the pooling paper (`predictability_gradient_academic`)
  and encoder-adaptation paper (`encoder_adaptation_family_shift`) already tried
  and found did not help under family shift; those are cross-referenced.
- Not changing Table 32 or `baseline_regressors.py` behavior. The new code
  imports its building blocks; the existing table stays reproducible.
- Not multi-seed. Single seed 0, consistent with Table 32. Multi-seed is noted as
  future work in the paper's limitations.

## Architecture

New module `paper/baseline_capacity_ladder.py`. It imports and reuses from
`paper/baseline_regressors.py`:

- `load_aligned_frame`, `load_features` — feature/trait/split loading.
- `Target`, `collect_targets`, `binary_trait_labels` path — target construction.
- `build_estimator`, `predict_probability`, `metrics_row`, `primary_metric` —
  model fitting and metrics.

It writes to `paper/tables/`:

- `34_capacity_ladder.{csv,md}` + figure `capacity_ladder.{pdf,png}`
- `35_fusion.{csv,md}` + figure `fusion_lift.{pdf,png}`
- `36_stack.{csv,md}`

Figures are added to `paper/figures/make_baseline_figures.py` (same style as the
existing baseline figures) or generated inline in the new script following that
module's conventions. Reuse the existing matplotlib style already used for
`baseline_regressor_*` figures.

Defaults (all overridable by CLI, matching `baseline_regressors.py` conventions):

- Targets: script default set — 7 binary traits + top-5 `cultivation_medium`
  one-vs-rest (the `DEFAULT_BINARY_TARGETS` + `--top-media 5`), not just the
  3-trait smoke subset.
- Features: `data/esm2_features.npz`.
- Split: `data/splits.parquet` family-held-out (`fsplit` train/test).
- Seed: 0. CPU-only.

## Piece 1 — Capacity ladder → plateau → escalate

### Ladder definition (monotone capacity axis)

Per target, evaluate rungs in order and record the primary metric
(`primary_metric`: AUPRC when family-test positive rate < 0.20, else AUROC):

- Rung 0 (floor): `chance` (majority-class DummyClassifier).
- Rungs 1–6 (capacity ↑): `logistic_l2` at PCA rank 6, 10, 25, 50, 100, full.
- Rung 7: `regularized_rf` at full rank.
- Rung 8: `hist_gd` at full rank.

Rungs 1–6 give a clean, monotone "effective dimensions" axis at fixed model
class; rungs 7–8 add nonlinearity at full capacity. All estimators come from the
existing `build_estimator`.

### Plateau detection

Walk rungs 1→8. Track best-so-far metric `m*`. Declare **plateau** at the first
rung r such that the gain over the previous rung is `< EPS` for two consecutive
rungs (`EPS = 0.005` on the primary metric). Report:

- `lower_bound` = metric at rung 1 (simplest non-trivial model).
- `plateau_rung`, `plateau_value` = metric where plateau declared (or the last
  rung if no plateau — flagged `plateau=false`).
- `ceiling` = the Table-33 cosine-kNN coverage oracle for that trait (the
  headroom reference). For traits not in Table 33, `ceiling = NaN` and the verdict
  is `unknown`.

### Escalation verdict

Decision rule per target:

- If not plateaued → `escalate: keep-scaling-simple` (bigger simple model still
  helping; no transformer needed yet).
- If plateaued and `ceiling - plateau_value <= MARGIN` (`MARGIN = 0.02`) →
  `escalate: NO — coverage-limited` (capacity-saturated at the coverage ceiling;
  a bigger model will not help; the bottleneck is cross-family coverage/labels).
- If plateaued and `ceiling - plateau_value > MARGIN` →
  `escalate: YES — capacity-limited` (simple models saturate below achievable
  signal; a higher-capacity model is indicated).

The expected result on these traits (given Table 33) is predominantly
`NO — coverage-limited`, delivering the paper's thesis via a reusable rule rather
than assertion. Report the verdict distribution across targets.

### Outputs

- `34_capacity_ladder.csv`: one row per (target, rung) with metric; plus a
  per-target summary (lower_bound, plateau, ceiling, verdict).
- `34_capacity_ladder.md`: the per-target summary table + a short methods note.
- Figure `capacity_ladder`: metric vs rung per trait, with plateau marker and the
  cosine ceiling drawn as a dashed line.

## Piece 2 — Embedding + extra-data fusion

For each target, fit the downstream ladder (reuse a representative subset:
`logistic_l2` rank 25, `regularized_rf` full, `hist_gd` full, and the soft-vote)
on four feature arms and report the primary metric per arm:

- `embed` — ESM-2 only (baseline; matches Table 32).
- `extra` — extra data only (no embedding).
- `embed+extra` — concatenation.

Extra-data sources (each run as its own `extra`/`embed+extra` pair):

- **taxonomy**: one-hot of `phylum`, `class`, `order`. Encoder fit on TRAIN
  categories only; unseen test categories map to all-zero (honest under family
  shift). `family`/`genus` deliberately excluded (novel families ⇒ dead columns).
- **isolation**: top-K (K=30) `isolation_source` + top-K `country` one-hot, same
  train-fit / unseen→zero handling; rare categories folded to an `other` column.
- **second_embed**: eggNOG (`data/eggnog_features_6738.npz`) and, if aligned rows
  are sufficient, Bacformer (`data/bacformer_features_all.npz`), aligned to the
  ESM-2 rows by `bacdive_id` (intersection only; report n aligned).

Report **lift = metric(embed+extra) − metric(embed)** per (target, source).
Aggregate mean lift per source. Explicitly flag the clade-confound: taxonomy
raising pathogenicity is the confound the pooling/encoder papers warn about, not
a clean generalization gain.

### Outputs

- `35_fusion.csv`: (target, source, arm, metric, lift).
- `35_fusion.md`: per-source mean lift table + best/worst targets.
- Figure `fusion_lift`: grouped bars of mean lift by source, with a zero line.

## Piece 3 — Learned stack vs soft-vote

Per target:

1. Base learners = the current `ENSEMBLE_MEMBERS` set from `baseline_regressors.py`
   (`logistic_l2`@25, `sgd_logistic`@25, `poly2_logistic`@10, `regularized_rf`@25,
   `hist_gd`@25).
2. Generate **out-of-fold** base predictions on TRAIN using GroupKFold on the
   `family` column (no family appears in both a fold's train and validation) — this
   prevents the family leakage that a naive stack would introduce under a
   family-held-out protocol.
3. Fit a logistic-regression meta-learner on the OOF prediction matrix.
4. Refit base learners on full train, predict family-test, feed through the
   meta-learner.
5. Compare stack vs fixed soft-vote (same base set) vs single best base learner.

If a target has too few training families for GroupKFold (< n_splits distinct
families in train), skip stacking for that target and record the reason.

### Outputs

- `36_stack.csv`: (target, method ∈ {best_base, soft_vote, learned_stack},
  primary metric).
- `36_stack.md`: per-target comparison + a one-line verdict (does stacking beat
  soft-vote on average?).

## Paper changes

Add three subsections to `paper/baseline_regressor_family_shift.md`, after the
current Results:

- "A capacity ladder tells you when to stop" (Piece 1, with the escalation rule
  and verdict; references Tables 32–33 and papers 2–3).
- "Does cheap side-data help under family shift?" (Piece 2).
- "Learned stacking vs a fixed vote" (Piece 3).

Update the Reproducibility section to list `paper/baseline_capacity_ladder.py`
and the new tables/figures. Update Limitations to note single-seed and the
taxonomy clade-confound.

## Testing / verification

- Run the new script on the 3-trait smoke subset first
  (`--targets motility sporulation catalase`) to confirm it executes end-to-end
  and the numbers are sane (e.g., sporulation ceiling ≈ 0.92 from Table 33).
- Sanity checks in-code: ladder metric is monotone-ish for logistic rank sweep;
  fusion `embed` arm reproduces Table 32 numbers for the shared targets/models;
  stack GroupKFold produces no family overlap between OOF folds.
- Confirm Table 32 outputs are byte-identical after the change (existing script
  untouched).

## Risks / open questions

- **Ceiling availability**: Table 33 only covers catalase/motility/sporulation.
  For the other binary traits and media, `ceiling = NaN` ⇒ verdict `unknown`;
  optionally extend the cosine diagnostic to those targets later. Acceptable for
  this pass.
- **Alignment loss** for the second-embedding arm (eggNOG/Bacformer cover fewer
  genomes) — report n aligned and treat as a subset result, not the headline.
- **Runtime**: full target set × arms × ladder on CPU. If slow, the fusion and
  stack pieces can be limited to the binary traits and top-3 media without
  changing conclusions; note any such cap in the table (no silent truncation).
