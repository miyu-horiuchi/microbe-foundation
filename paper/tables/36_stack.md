# Table 36 -- Learned stack vs soft-vote

Base learners = Table-32 ensemble set. Out-of-fold predictions use GroupKFold on `family` (no family leakage); meta-learner = balanced logistic regression. The `best_base` column selects the strongest single base learner on the test set (an optimistic oracle), shown for reference only.

| Target | best_base | soft_vote | learned_stack |
|---|---:|---:|---:|
| `catalase` | 0.805 | 0.805 | 0.809 |
| `cultivation_medium:514` | 0.783 | 0.756 | 0.754 |
| `cultivation_medium:65` | 0.818 | 0.807 | 0.797 |
| `cultivation_medium:693` | 0.261 | 0.177 | 0.196 |
| `cultivation_medium:830` | 0.415 | 0.353 | 0.389 |
| `cultivation_medium:92` | 0.352 | 0.341 | 0.348 |
| `cytochrome_oxidase` | 0.871 | 0.860 | 0.867 |
| `motility` | 0.717 | 0.731 | 0.716 |
| `pathogenicity_animal` | 0.207 | 0.167 | 0.112 |
| `pathogenicity_human` | 0.142 | 0.155 | 0.143 |
| `pigmentation` | 0.598 | 0.506 | 0.579 |
| `sporulation` | 0.923 | 0.936 | 0.930 |
