# Table 33 -- Cosine geometry diagnostic for family-collapse

Frozen ESM-2 embeddings are standardized using family-train genomes, L2-normalized, and evaluated by cosine similarity. Negative Spearman correlations mean lower cosine support is associated with larger prediction error.

## Trait-level geometry

| Trait | Test n | Test families | Pos rate | nearest family cos | top-1 genome cos | kNN F1 | kNN AUROC | Probe F1 | Probe AUROC | rho(cos,error) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `motility` | 1673 | 148 | 0.363 | 0.846 | 0.854 | 0.625 | 0.695 | 0.643 | 0.728 | +0.007 | mixed/weak cosine-error link |
| `sporulation` | 809 | 112 | 0.253 | 0.850 | 0.850 | 0.835 | 0.924 | 0.815 | 0.935 | +0.220 | mixed/weak cosine-error link |
| `catalase` | 1442 | 109 | 0.818 | 0.845 | 0.850 | 0.691 | 0.764 | 0.713 | 0.817 | -0.107 | coverage/geometry-limited |

## Performance by nearest-training-family cosine bin

| Trait | Cosine bin | n | mean centroid cos | Pos rate | kNN F1 | kNN AUROC | Probe F1 | Probe AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `motility` | low | 558 | 0.743 | 0.346 | 0.580 | 0.663 | 0.639 | 0.733 |
| `motility` | mid | 557 | 0.867 | 0.400 | 0.604 | 0.665 | 0.606 | 0.708 |
| `motility` | high | 558 | 0.927 | 0.342 | 0.688 | 0.753 | 0.675 | 0.737 |
| `sporulation` | low | 270 | 0.741 | 0.230 | 0.808 | 0.927 | 0.860 | 0.946 |
| `sporulation` | mid | 269 | 0.875 | 0.186 | 0.787 | 0.904 | 0.792 | 0.921 |
| `sporulation` | high | 270 | 0.933 | 0.344 | 0.877 | 0.922 | 0.782 | 0.928 |
| `catalase` | low | 481 | 0.742 | 0.794 | 0.598 | 0.769 | 0.704 | 0.865 |
| `catalase` | mid | 480 | 0.866 | 0.821 | 0.685 | 0.744 | 0.683 | 0.781 |
| `catalase` | high | 481 | 0.929 | 0.838 | 0.786 | 0.755 | 0.750 | 0.785 |
