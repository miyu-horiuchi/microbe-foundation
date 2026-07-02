# Table 36 -- Learned stack vs soft-vote

Base learners = Table-32 ensemble set. Out-of-fold predictions use GroupKFold on `family` (no family leakage); meta-learner = balanced logistic regression.

| Target | best_base | soft_vote | learned_stack |
|---|---:|---:|---:|
| `catalase` | 0.805 | 0.805 | 0.809 |
| `motility` | 0.717 | 0.731 | 0.716 |
| `sporulation` | 0.923 | 0.936 | 0.930 |
