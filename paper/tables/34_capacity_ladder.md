# Table 34 -- Capacity ladder: plateau and escalation verdict

Ladder: L2 logistic at PCA rank 6/10/25/50/100/full, then random forest and hist gradient boosting at full rank. Plateau EPS=0.005; escalation MARGIN=0.02; ceiling = cosine-kNN AUROC from Table 33.

| Target | Primary | Lower bound | Plateau | Plateaued | Ceiling (kNN) | Verdict |
|---|---|---:|---:|:--:|---:|---|
| `catalase` | auroc | 0.751 | 0.816 | True | 0.764 | NO — coverage-limited |
| `cytochrome_oxidase` | auroc | 0.815 | 0.847 | True | nan | unknown |
| `sporulation` | auroc | 0.824 | 0.933 | True | 0.924 | NO — coverage-limited |
| `pigmentation` | auroc | 0.559 | 0.559 | True | nan | unknown |
| `pathogenicity_human` | auprc | 0.094 | 0.156 | True | nan | unknown |
| `pathogenicity_animal` | auprc | 0.195 | 0.240 | True | nan | unknown |
| `motility` | auroc | 0.677 | 0.738 | False | 0.695 | keep-scaling-simple |
| `cultivation_medium:65` | auprc | 0.684 | 0.802 | True | nan | unknown |
| `cultivation_medium:514` | auprc | 0.291 | 0.783 | True | nan | unknown |
| `cultivation_medium:693` | auprc | 0.100 | 0.249 | True | nan | unknown |
| `cultivation_medium:830` | auprc | 0.200 | 0.416 | True | nan | unknown |
| `cultivation_medium:92` | auprc | 0.146 | 0.213 | True | nan | unknown |
