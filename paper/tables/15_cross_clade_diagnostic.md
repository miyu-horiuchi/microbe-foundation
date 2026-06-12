# Table 15 — Cross-clade collapse diagnostic

Why family-level transfer collapses: training-clade *coverage* vs a frozen-representation *wall*. Diagnostic A = test macro-F1 vs #training families; B = cross-clade k-NN label transfer vs the probe and a chance baseline.

## B. Cross-clade k-NN transfer

| Trait | Test n | Pos rate | k-NN F1 | k-NN AUROC | Probe F1 | Probe AUROC | Chance F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sporulation` | 809 | 0.253 | 0.820 | 0.915 | 0.828 | 0.925 | 0.427 |
| `motility` | 1673 | 0.363 | 0.627 | 0.693 | 0.635 | 0.696 | 0.389 |
| `catalase` | 1442 | 0.818 | 0.690 | 0.746 | 0.686 | 0.793 | 0.450 |

## A. Family-diversity curve (mean test macro-F1 over seeds)

The *natural* curve confounds #families with #training-genomes (fewer families = fewer genomes); the *fixed-N* curve holds training size constant so only family count varies, isolating clade diversity from data volume.

| Trait | nat k=min | nat k=max | nat rising? | fixed-N | fixedN k=min | fixedN k=max | fixedN rising? | verdict |
|---|---:|---:|:--:|---:|---:|---:|:--:|:--:|
| `sporulation` | 0.586 | 0.828 | yes | 15 | 0.616 | 0.707 | yes | coverage-limited (clade diversity) |
| `motility` | 0.515 | 0.635 | yes | 33 | 0.533 | 0.556 | yes | coverage-limited (clade diversity) |
| `catalase` | 0.546 | 0.686 | yes | 20 | 0.498 | 0.621 | yes | coverage-limited (clade diversity) |
