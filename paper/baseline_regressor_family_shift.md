---
title: "Simple Readouts on Frozen Genome Embeddings Are Strong Baselines for Cross-Family Microbial Trait Prediction"
author:
  - |
    Miyu Horiuchi  
    m@replicater.xyz
date: 2026-07-01
abstract: |
  Recent microbial phenotype predictors often place architectural emphasis on genome-level pooling or encoder adaptation. A simpler alternative is to treat a pretrained protein language model as a frozen feature extractor and fit a regularized downstream model. We test this baseline family on frozen ESM-2 genome embeddings under the same family-held-out split used in our prior cross-clade studies. On three representative binary traits (catalase, motility, sporulation), simple readouts are already strong: a soft-vote ensemble reaches mean AUROC 0.824 and mean macro-F1 0.736; the strongest single-model baselines are regularized random forest (macro-F1 0.724, AUROC 0.803), histogram gradient boosting (macro-F1 0.717, AUROC 0.803), and L2 logistic regression at PCA rank 25 (macro-F1 0.713, AUROC 0.808). Polynomial logistic models are competitive but not best, while the SGD-trained logistic baseline is weaker. The PCA rank sweep is especially informative: rank 25 generally outperforms rank 6, arguing against a perturbation-style "six-dimensional" explanation for these microbial traits. The result reframes our earlier pooling and LoRA findings: frozen ESM-2 already contains substantial recoverable trait signal, and the remaining family-shift failure is more likely due to cross-clade coverage, label sparsity/noise, and trait biology than to a missing high-capacity encoder update.
geometry: margin=1in
fontsize: 11pt
---

# Introduction

Our previous experiments isolated two architectural levers in microbial genome-to-phenotype prediction. The first asked whether learned pooling over protein embeddings helps compared with mean pooling. The answer was trait-dependent in-distribution, but the advantage collapsed under family-held-out evaluation. The second asked whether adapting ESM-2 with LoRA recovers the lost family-transfer signal. The answer was again negative once imbalanced-label metric artifacts were removed.

A natural critique of both experiments is that they may have skipped an important baseline: **simple downstream models on frozen embeddings**. In many applied biological models, a heavy upstream representation model produces rich features, and a smaller downstream regressor or classifier extracts the task-specific signal. If the embedding already contains the relevant information, then fine-tuning the encoder may add cost and overfitting risk without improving generalization.

This note tests that critique directly. We keep ESM-2 frozen, compress the genome embeddings to several PCA ranks, and fit a family of simple readouts: L2 logistic regression, an SGD-trained logistic classifier, degree-2 polynomial logistic regression, regularized random forest, histogram gradient boosting, and a fixed soft-vote ensemble.

![**Experimental control.** The encoder is not fine-tuned. We test whether frozen ESM-2 genome embeddings already contain trait signal that simple downstream models can recover under a family-held-out split.](figures/baseline_regressor_schematic.png){width=100%}

# Methods

## Data and split

We reuse the existing `data/esm2_features.npz` genome embeddings, `data/traits.parquet` labels, and `data/splits.parquet` family-held-out split. This keeps the readout experiment aligned with the earlier pooling and encoder-adaptation studies. The smoke run reported here evaluates three binary traits:

- `catalase`, a well-supported enzymatic physiology trait.
- `motility`, a cross-clade trait with weaker transfer.
- `sporulation`, a localized machinery trait with strong recoverable signal.

The evaluated family-test sizes are 1,442 for catalase, 1,673 for motility, and 809 for sporulation.

## Model family

All models receive the same frozen ESM-2 genome representation. To test whether the signal is low-dimensional, we apply a PCA bottleneck before the downstream model at ranks 6, 10, and 25. Rank 6 is included because low-rank perturbation benchmarks in single-cell modeling have been criticized for containing only a handful of effective dimensions. Rank 25 tests whether microbial trait signal requires a broader subspace.

The downstream models are:

- **L2 logistic regression:** a regularized linear classifier.
- **SGD logistic:** an elastic-net logistic classifier trained by stochastic gradient descent.
- **Polynomial logistic:** degree-2 feature interactions after PCA.
- **Regularized random forest:** shallow, leaf-regularized tree ensemble.
- **Histogram gradient boosting:** boosted trees with learning-rate and L2 regularization.
- **Soft-vote ensemble:** average probability over representative strong baselines.

We report AUROC and macro-F1. AUROC measures ranking quality independent of a threshold. Macro-F1 measures thresholded classification quality while weighting both classes equally, which is important for imbalanced biological labels.

# Results

## Best model per trait

Simple readouts are not weak controls; they are competitive models.

: Best downstream model on each target.

| Trait | Test n | Test positive rate | Best model | Rank | AUROC | Macro-F1 | AUPRC |
|---|---:|---:|---|---:|---:|---:|---:|
| `catalase` | 1,442 | 0.818 | L2 logistic regression | 25 | 0.805 | 0.700 | 0.933 |
| `motility` | 1,673 | 0.363 | soft-vote ensemble | fixed | 0.731 | 0.641 | 0.592 |
| `sporulation` | 809 | 0.253 | soft-vote ensemble | fixed | 0.936 | 0.848 | 0.861 |

The most striking result is sporulation: a simple ensemble over frozen embeddings reaches AUROC 0.936 and macro-F1 0.848 on held-out families. This means the ESM-2 genome embedding already exposes a strong sporulation signal to lightweight readouts. Motility is weaker but still above chance. Catalase is strongly ranked by logistic regression, though its high positive rate makes macro-F1 the more cautious measure.

![**Best readout per trait.** AUROC and macro-F1 for the strongest simple model on each held-out-family trait. Sporulation is highly recoverable, catalase is well ranked, and motility is the hardest of the three but still clears chance.](figures/baseline_regressor_per_trait.png){width=85%}

## Model-family comparison

: Mean performance across catalase, motility, and sporulation.

| Model | Rank | Mean macro-F1 | Mean AUROC | Mean AUPRC |
|---|---:|---:|---:|---:|
| soft-vote ensemble | fixed | **0.736** | **0.824** | **0.795** |
| regularized random forest | 25 | 0.724 | 0.803 | 0.765 |
| histogram gradient boosting | 25 | 0.717 | 0.803 | 0.768 |
| L2 logistic regression | 25 | 0.713 | 0.808 | 0.759 |
| polynomial logistic | 25 | 0.698 | 0.776 | 0.726 |
| SGD logistic | 25 | 0.671 | 0.761 | 0.720 |

Three single-model baselines are close: L2 logistic regression, regularized random forest, and histogram gradient boosting. This pattern is informative. The logistic result says much of the signal is almost linearly recoverable from frozen ESM-2. The tree results say there are useful nonlinear interactions, but they do not require a deep downstream architecture. Polynomial logistic models are reasonable but not best, suggesting that "all pairwise interactions" is not the right inductive bias by itself. The SGD classifier is the weakest of the main families, indicating that optimizer choice and regularization details matter even for simple readouts.

![**Rank sweep.** Rank 25 generally improves over rank 6, so these microbial traits do not look like a six-dimensional perturbation benchmark. The best single-model baselines cluster around AUROC 0.80, and the soft-vote ensemble reaches 0.824.](figures/baseline_regressor_rank_sweep.png){width=100%}

## Rank 25 beats rank 6

The PCA sweep is the key connection to the low-rank-data critique. If these traits behaved like a six-dimensional perturbation dataset, rank 6 would be expected to saturate the simple-model performance. It does not. For L2 logistic regression, mean AUROC rises from 0.751 at rank 6 to 0.808 at rank 25. For regularized random forest, mean AUROC rises from 0.770 to 0.803. For histogram gradient boosting, it rises from 0.781 to 0.803.

This does not prove that the full 640-dimensional embedding is necessary, because the smoke run did not include the full-rank setting. But it does show that the useful microbial trait signal is broader than six principal components, unlike the single-cell perturbation example that motivated the critique.

# Discussion

## What this changes relative to the pooling and LoRA papers

The result sharpens, rather than contradicts, our earlier findings. The pooling study showed that adaptive aggregation helps some traits in-distribution but not under family shift. The LoRA study showed that encoder adaptation does not rescue family transfer and can create metric artifacts on imbalanced heads. This baseline experiment adds a third point:

> Frozen ESM-2 embeddings already contain substantial microbial trait signal, and simple regularized readouts can recover much of it.

Therefore, the failure mode is unlikely to be simply "the model is not big enough" or "the encoder was not fine-tuned." The more plausible bottlenecks are:

- Family-level distribution shift: held-out families differ biologically from training families.
- Label sparsity and noise: BacDive labels are incomplete, biased toward cultured/model organisms, and often not true negatives.
- Trait-specific biology: some traits are localized and recoverable, while others are diffuse, confounded, or under-observed.
- Metric choice: accuracy can reward majority-class collapse on imbalanced labels, so F1/AUPRC must be reported.

## Practical implication

For this project, the next experimental baseline should not be another expensive encoder-adaptation run by default. The more defensible pipeline is:

1. Freeze the genome encoder.
2. Sweep PCA rank and simple regularized readouts.
3. Identify which traits plateau under simple models.
4. Only then escalate to attention pooling, LoRA, or a small transformer head for the traits where simple models fail.

This also makes the paper stronger. It shows that the project is not comparing foundation models against straw-man baselines. It tests the "Pfizer-style" pattern directly: heavy upstream embedding, simple downstream model.

# Limitations

This is a smoke run, not the final benchmark. It covers three binary traits, one encoder (`data/esm2_features.npz`), ranks 6/10/25, and one family-held-out split. The full run should add all binary traits, top cultivation-medium one-vs-rest labels, Bacformer embeddings, full-rank readouts, and multiple seeds where runtime permits. The current result is enough to establish that the baseline family is strong and scientifically necessary; it is not yet enough to claim a universal winner.

# Conclusion

The baseline-regressor experiment validates the critique that simple models should be tried before claiming that encoder adaptation is needed. On frozen ESM-2 genome embeddings, regularized random forest, histogram gradient boosting, and L2 logistic regression at PCA rank 25 are all strong. Polynomial logistic regression is competitive but not best, and the SGD-trained logistic baseline is weaker. Most importantly, rank 25 improves over rank 6, so these microbial trait labels do not appear to be a trivially six-dimensional perturbation-style dataset. The right interpretation is that frozen foundation embeddings already carry useful biological signal; the unsolved problem is making that signal transfer reliably across clades and sparse phenotypic labels.

# Reproducibility

The experiment is implemented in `paper/baseline_regressors.py`. The generated outputs are:

- `paper/tables/32_baseline_regressors.md`
- `paper/tables/32_baseline_regressors.csv`
- `paper/tables/32_baseline_regressors.json`

The smoke-run command was:

```bash
python paper/baseline_regressors.py --targets motility sporulation catalase --ranks 6,10,25 --top-media 0
```
