# Interview / Defense Prep — Microbe Foundation (trait prediction from protein embeddings)

Anticipated ML questions with crisp answers, grounded in the tier-1 results
(`paper/tables/18_seed_aggregated.md`, `paper/figures/18_pooling_split_macro.png`).
Each answer is structured: **short answer → why → evidence**.

---

## 1. What is the actual task and setup?

**Short answer.** Given a microbe's *set* of protein sequences, predict ~20 phenotypic
traits (gram stain, motility, oxygen tolerance, catalase, pathogenicity, carbon
utilization, etc.). We encode each protein with a **frozen ESM-2** model, **pool** the
per-protein embeddings into one organism vector, and train lightweight per-trait heads.

**Why this shape.** A genome is a *bag of proteins* of variable size (hundreds to
thousands). The model must be permutation-invariant over proteins and handle variable
cardinality — which is exactly what a pooling layer over a frozen encoder gives us.

**Evidence.** Feature dim 640 (ESM-2 per-protein), up to 2048 proteins/organism, ~1M
trainable params in the head — the encoder itself is never updated.

---

## 2. What do "mean", "attention", and "set_transformer" pooling actually do?

- **Mean pool** — average the per-protein embeddings. Zero parameters, permutation-
  invariant, strong baseline. Every protein contributes equally.
- **Attention pool** — learn a weight per protein (a small gated-attention scorer) and
  take a weighted average. Lets the model *up-weight the few proteins that matter* for a
  trait (e.g. a flagellar gene for motility) and ignore housekeeping proteins.
- **Set Transformer** — a permutation-invariant transformer with **inducing points**
  (learned query vectors, here 16) that attend over the protein set with multi-head
  attention. Most expressive: it can model *interactions between proteins*, not just
  re-weight them.

**One-liner ranking of capacity:** mean < attention < set_transformer.
**One-liner ranking of robustness (our finding):** attention ≥ mean > set_transformer
on the hardest split.

---

## 3. Why freeze ESM-2 instead of fine-tuning it?

**Short answer.** Three reasons: cost, data size, and overfitting risk.

- **Cost / compute.** ESM-2 is hundreds of millions to billions of params; fine-tuning it
  across thousands of proteins × thousands of organisms is enormous. Freezing turns the
  encoder into a one-time **feature extractor** — we embed once, cache to disk (~100 GB),
  and then every experiment trains a ~1M-param head in minutes.
- **Data size.** We have ~16k organisms with labels, many traits sparsely labeled.
  That is far too little signal to safely move a billion-parameter encoder without
  catastrophic overfitting.
- **Scientific cleanliness.** Freezing isolates the question we care about — *how to pool*
  — from confounds in encoder adaptation. Everyone shares the identical frozen features,
  so a pooling comparison is apples-to-apples.

**Evidence.** The entire 36-run matrix (3 poolings × 3 splits × balancing × 3 seeds)
runs in hours on one A100 *because* the encoder is frozen and embeddings are precomputed.

---

## 4. Why does LoRA come up right after "freeze the encoder"?

**Short answer.** LoRA is the *middle ground* between fully frozen and fully fine-tuned.

If a reviewer says "frozen features might be leaving accuracy on the table," the standard
answer is: **LoRA** (Low-Rank Adaptation) injects small trainable low-rank matrices into
the frozen encoder's attention layers. You adapt the encoder with <1% extra params and no
risk of blowing up the pretrained weights — getting *some* task-specific adaptation while
keeping cost and overfitting bounded. It is the natural next experiment if frozen pooling
plateaus, which is why it follows the freezing discussion.

---

## 5. Why macro-F1 (and macro-averaged scores) instead of accuracy?

**Short answer.** The labels are **heavily imbalanced**, so accuracy is misleading.

- For a trait that is 90% one class, a model predicting only the majority gets 90%
  accuracy while learning nothing. **Macro-F1 averages F1 across classes equally**, so
  the model is rewarded only if it gets the *rare* classes right too.
- We report a **macro score across the 20 traits** (mean of each trait's primary metric)
  so no single easy trait dominates the headline number.

**Evidence.** `country` has 100 classes and lands at ~0.02–0.06 — accuracy would never
expose how hard it is, but the macro view makes the difficulty visible and honest.
Class-balanced loss weighting (`--class-weights`) is applied for the same reason.

---

## 6. Why is the family split so much harder than species/genus?

**Short answer.** It tests **cross-clade generalization** — predicting traits for
*entire bacterial families the model has never seen*.

- **Species split**: held-out organisms can have close relatives in training → easy, the
  model can interpolate within a clade.
- **Genus split**: held-out genera, relatives one step further away → harder.
- **Family split**: hold out *whole families* → the model must extrapolate phenotype from
  protein content alone, with **no near-neighbor to lean on**. This is the realistic
  setting for characterizing a newly discovered, deeply divergent microbe.

**Evidence (every pooler degrades monotonically):**

| Pooling | Species | Genus | Family |
|---|---:|---:|---:|
| Attention | 0.648 | 0.639 | 0.621 |
| Mean | 0.641 | 0.629 | 0.608 |
| Set Transformer | 0.642 | 0.630 | 0.539 |

The species→family drop *is* the generalization gap — the central scientific result.

---

## 7. What are you comparing against (baselines)?

- **Within this study**: the three pooling architectures against each other, with
  **mean pooling as the zero-parameter baseline**. A more complex pooler must beat mean to
  justify itself.
- **Prior / external**: phylogeny- and orthology-based predictors such as **eggNOG**-style
  approaches (annotation transfer via orthologous groups). The point of the family split
  is precisely where annotation-transfer methods struggle — no close ortholog reference —
  so a learned pooler that holds up there is the contribution.
- **Naive baselines**: majority-class / trait-prevalence priors (why we don't use
  accuracy — see Q5).

---

## 8. What is the "PU-learning / missing-not-negative" point?

**Short answer.** Absent labels are **unknown, not negative** — this is a
**Positive-Unlabeled (PU)** setting.

In BacDive-style data, if a trait isn't recorded for an organism it usually means
*nobody measured it*, not that the organism lacks it. Treating missing-as-negative
injects systematic false negatives and biases the model. We therefore **mask unlabeled
entries out of the loss** (only supervise on observed labels) rather than assuming a
negative. The empty-label-batch bug we fixed is a direct consequence of this masking:
under family-balanced resampling a batch can occasionally contain *zero* observed labels
for every head, producing a loss with no gradient — now skipped safely.

---

## 9. Headline results — what should I lead with?

1. **Cross-clade generalization gap is real and orderly**: species > genus > family for
   every architecture (the core finding).
2. **Attention pooling is the best and most robust pooler** — highest at every split and
   the clear winner on the hardest family split (0.621 vs 0.608 mean vs 0.539 set
   transformer).
3. **Capacity ≠ robustness**: the most expressive pooler (Set Transformer) is the *least*
   robust under distribution shift — it **collapses on the raw family split (0.539)** and
   one of three seeds destabilizes (seed CI ±0.25 vs ±0.01 elsewhere).
4. **Family-balanced resampling rescues the fragile model** (+0.072 → 0.611 for Set
   Transformer) but barely moves the robust one (attention 0.621 → 0.619), showing the
   instability was overfitting to dominant families, not a data limit.

---

## 10. Likely follow-up "gotcha" questions

- **"Why only 3 seeds?"** — Tight seed CIs (±0.003–0.011 for stable configs) show 3 is
  enough to separate architectures; the one wide CI (set_transformer/family) is itself the
  finding, not noise we need to average away.
- **"Isn't attention just mean pooling with extra steps?"** — Empirically it adds ~+0.01
  consistently and, crucially, *more* on the hard family split, i.e. it helps exactly when
  selective weighting matters most.
- **"Could the gap be label leakage across splits?"** — Splits are by taxonomy
  (`family_split`), so held-out families share no organisms with train; the monotone
  degradation is consistent with genuine extrapolation, not leakage.
- **"Why not fine-tune end-to-end for the headline number?"** — See Q3/Q4: cost,
  data size, and clean attribution; LoRA is the principled next step.
- **"What's the real-world use?"** — Predicting phenotype for newly sequenced, deeply
  divergent microbes where no close annotated relative exists — exactly the family-split
  regime.
