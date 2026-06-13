# Trait coverage analysis — `pathogenicity_human` (family-held-out)

- Train / test genomes: 6159 / 2158  (pos rate 0.130 / 0.061)
- **Signal vs threshold:** probe AUROC = **0.761** but macro-F1 = **0.581** (gap +0.179). The embedding ranks novel-family positives well above chance; imbalance + a 0.5 threshold flatten F1.
- **Positive concentration:** 49 of 195 training families contain any positive; Gini = 0.902; top-5 families hold 42.6% of all positive labels.

## Recall on true positives by neighbour-positive-rate (k=10)

Fraction of each novel-family test genome's k nearest *training* neighbours that are positive. Recall climbs with neighbour positive rate: the model flags a novel pathogen only when embedding-space neighbours were known pathogens — the coverage mechanism.

| Neighbour pos-rate | Genomes | Positives | Recall on positives |
|---|---:|---:|---:|
| [0, 0.001) | 1378 | 37 | 0.108 |
| [0.001, 0.1) | 0 | 0 | — |
| [0.1, 0.3) | 523 | 70 | 0.514 |
| [0.3, 1.01) | 257 | 25 | 0.720 |
