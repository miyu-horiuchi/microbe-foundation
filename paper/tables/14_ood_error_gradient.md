# Table 14 — Family-split OOD score vs per-genome error

Linear-probe per-trait test of whether the embedding OOD score predicts per-genome error on novel families. Positive Spearman = more novel implies more error. The relationship is trait-specific: positive only for sporulation, null for most traits, negative for imbalanced pathogenicity. The OOD score is a novelty detector, not a per-genome error oracle.

| Trait | Group | Test n | Pos rate | AUROC | Spearman(OOD,err) | p |
|---|---|---:|---:|---:|---:|---:|
| `motility` | machinery | 1673 | 0.363 | 0.696 | -0.0081 | 7.4e-01 |
| `sporulation` | machinery | 809 | 0.253 | 0.925 | +0.1425 | 4.8e-05 |
| `catalase` | compositional | 1442 | 0.818 | 0.792 | -0.0176 | 5.0e-01 |
| `cytochrome_oxidase` | compositional | 1352 | 0.676 | 0.837 | +0.0562 | 3.9e-02 |
| `pigmentation` | compositional | 758 | 0.736 | 0.500 | -0.0406 | 2.6e-01 |
| `pathogenicity_animal` | other | 2172 | 0.044 | 0.774 | -0.1999 | 5.3e-21 |
| `pathogenicity_human` | other | 2158 | 0.061 | 0.761 | -0.1192 | 2.8e-08 |
