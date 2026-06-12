# Table 16 — Retrieval-augmented head (cross-clade)

Convex blend of the linear probe and cross-clade k-NN, `alpha * probe + (1 - alpha) * knn`, with `alpha` tuned on family-val and evaluated on family-test. `alpha* = 1` means the probe alone wins (retrieval adds nothing); `alpha* = 0` means k-NN alone wins. `delta F1` is blend minus probe-alone on family-test.

| Trait | Test n | Pos rate | alpha* | Probe F1 | k-NN F1 | Blend F1 | delta F1 | Probe AUROC | k-NN AUROC | Blend AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sporulation` | 809 | 0.253 | 0.4 | 0.827 | 0.820 | 0.844 | +0.018 | 0.925 | 0.915 | 0.935 |
| `motility` | 1673 | 0.363 | 0.4 | 0.636 | 0.627 | 0.656 | +0.020 | 0.696 | 0.693 | 0.721 |
| `catalase` | 1442 | 0.818 | 0.4 | 0.686 | 0.690 | 0.726 | +0.039 | 0.793 | 0.746 | 0.797 |
