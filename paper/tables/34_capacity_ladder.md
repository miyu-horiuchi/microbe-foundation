# Table 34 -- Capacity ladder: plateau and escalation verdict

Ladder: L2 logistic at PCA rank 6/10/25/50/100/full, then random forest and hist gradient boosting at full rank. Plateau EPS=0.005; escalation MARGIN=0.02; ceiling = cosine-kNN AUROC from Table 33.

| Target | Primary | Lower bound | Plateau | Plateaued | Ceiling (kNN) | Verdict |
|---|---|---:|---:|:--:|---:|---|
| `motility` | auroc | 0.677 | 0.721 | False | 0.695 | keep-scaling-simple |
| `sporulation` | auroc | 0.824 | 0.925 | True | 0.924 | NO — coverage-limited |
| `catalase` | auroc | 0.751 | 0.793 | True | 0.764 | NO — coverage-limited |
