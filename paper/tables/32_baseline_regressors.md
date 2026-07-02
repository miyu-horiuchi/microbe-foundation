# Table 32 -- Baseline regressor family on frozen genome embeddings

Features: `data/esm2_features.npz`. Family-held-out split. Rank sweep: 6, 10, 25. Polynomial models use PCA first, so degree-2 expansion tests low-rank interactions without exploding the original embedding dimension.

Primary metric = AUPRC when the family-test positive rate is <20%, otherwise AUROC. This makes rare pathogenicity/media targets less misleading than accuracy.

## Best simple downstream model per target

| Target | Group | Test n | Pos rate | Best model | Rank | Primary | Score | Lift vs chance | Macro-F1 | AUROC | AUPRC |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| `catalase` | binary_trait | 1442 | 0.818 | logistic_l2 | 25 | auroc | 0.805 | +0.305 | 0.700 | 0.805 | 0.933 |
| `motility` | binary_trait | 1673 | 0.363 | soft_vote_ensemble | fixed | auroc | 0.731 | +0.231 | 0.641 | 0.731 | 0.592 |
| `sporulation` | binary_trait | 809 | 0.253 | soft_vote_ensemble | fixed | auroc | 0.936 | +0.436 | 0.848 | 0.936 | 0.861 |

## Model-family means

| Model | Rank | Targets | Mean macro-F1 | Mean AUROC | Mean AUPRC |
|---|---:|---:|---:|---:|---:|
| hist_gd | 10 | 3 | 0.708 | 0.791 | 0.743 |
| hist_gd | 25 | 3 | 0.717 | 0.803 | 0.768 |
| hist_gd | 6 | 3 | 0.688 | 0.781 | 0.724 |
| logistic_l2 | 10 | 3 | 0.671 | 0.764 | 0.646 |
| logistic_l2 | 25 | 3 | 0.713 | 0.808 | 0.759 |
| logistic_l2 | 6 | 3 | 0.640 | 0.751 | 0.648 |
| poly2_logistic | 10 | 3 | 0.688 | 0.785 | 0.715 |
| poly2_logistic | 25 | 3 | 0.698 | 0.776 | 0.726 |
| poly2_logistic | 6 | 3 | 0.660 | 0.764 | 0.696 |
| regularized_rf | 10 | 3 | 0.717 | 0.789 | 0.738 |
| regularized_rf | 25 | 3 | 0.724 | 0.803 | 0.765 |
| regularized_rf | 6 | 3 | 0.689 | 0.770 | 0.705 |
| sgd_logistic | 10 | 3 | 0.660 | 0.733 | 0.621 |
| sgd_logistic | 25 | 3 | 0.671 | 0.761 | 0.720 |
| sgd_logistic | 6 | 3 | 0.613 | 0.702 | 0.619 |
| soft_vote_ensemble | fixed | 3 | 0.736 | 0.824 | 0.795 |
