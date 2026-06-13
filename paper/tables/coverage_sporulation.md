# Trait coverage analysis — `sporulation` (family-held-out)

- Train / test genomes: 2473 / 809  (pos rate 0.410 / 0.253)
- **Signal vs threshold:** probe AUROC = **0.925** but macro-F1 = **0.827** (gap +0.098). The embedding ranks novel-family positives well above chance; imbalance + a 0.5 threshold flatten F1.
- **Positive concentration:** 32 of 139 training families contain any positive; Gini = 0.961; top-5 families hold 87.2% of all positive labels.

## Recall on true positives by neighbour-positive-rate (k=10)

Fraction of each novel-family test genome's k nearest *training* neighbours that are positive. Recall climbs with neighbour positive rate: the model flags a novel pathogen only when embedding-space neighbours were known pathogens — the coverage mechanism.

| Neighbour pos-rate | Genomes | Positives | Recall on positives |
|---|---:|---:|---:|
| [0, 0.001) | 359 | 6 | 0.000 |
| [0.001, 0.1) | 0 | 0 | — |
| [0.1, 0.3) | 116 | 13 | 0.077 |
| [0.3, 1.01) | 334 | 186 | 0.812 |
