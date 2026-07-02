# Table 23 — Encoder comparison: bacformer vs esm2_150M (family-held-out)

Same 12062 shared genomes, same family split, same balanced linear probe; only the frozen genome representation changes. Δ = bacformer − esm2_150M. A positive Δ means the stronger encoder recovers cross-clade signal that the pooling and retrieval levers could not.

| Trait | Mode | Test n | Pos rate | esm2_150M F1 | bacformer F1 | ΔF1 | esm2_150M AUROC | bacformer AUROC | ΔAUROC |
|---|:--|---:|---:|---:|---:|---:|---:|---:|---:|
| `catalase` | solved | 904 | 0.796 | 0.679 | 0.662 | -0.018 | 0.787 | 0.774 | -0.013 |
| `cytochrome_oxidase` | solved | 834 | 0.664 | 0.782 | 0.787 | +0.005 | 0.844 | 0.857 | +0.013 |
| `sporulation` | solved | 525 | 0.293 | 0.847 | 0.872 | +0.025 | 0.927 | 0.948 | +0.021 |
| `pigmentation` | solved | 475 | 0.741 | 0.500 | 0.521 | +0.021 | 0.517 | 0.569 | +0.053 |
| `pathogenicity_human` | coverage-limited | 1350 | 0.074 | 0.583 | 0.639 | +0.056 | 0.753 | 0.742 | -0.011 |
| `pathogenicity_animal` | coverage-limited | 1361 | 0.057 | 0.596 | 0.593 | -0.002 | 0.745 | 0.757 | +0.013 |
| `motility` | coverage-limited | 1029 | 0.342 | 0.628 | 0.629 | +0.001 | 0.671 | 0.677 | +0.005 |

Mean Δ across 7 traits: ΔF1 +0.013, ΔAUROC +0.012.
