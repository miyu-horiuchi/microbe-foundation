# Table 15 — Cross-clade collapse diagnostic

Why family-level transfer collapses: training-clade *coverage* vs a frozen-representation *wall*. Diagnostic A = test macro-F1 vs #training families; B = cross-clade k-NN label transfer vs the probe and a chance baseline.

## B. Cross-clade k-NN transfer

| Trait | Test n | Pos rate | k-NN F1 | k-NN AUROC | Probe F1 | Probe AUROC | Chance F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sporulation` | 809 | 0.253 | 0.820 | 0.915 | 0.828 | 0.925 | 0.427 |
| `motility` | 1673 | 0.363 | 0.627 | 0.693 | 0.635 | 0.696 | 0.389 |
| `catalase` | 1442 | 0.818 | 0.690 | 0.746 | 0.686 | 0.793 | 0.450 |

## A. Family-diversity curve (mean test macro-F1 over seeds)

| Trait | k=min | k=max | rising? | verdict |
|---|---:|---:|:--:|:--:|
| `sporulation` | 0.586 | 0.828 | yes | coverage-limited |
| `motility` | 0.515 | 0.635 | yes | coverage-limited |
| `catalase` | 0.546 | 0.686 | yes | coverage-limited |
