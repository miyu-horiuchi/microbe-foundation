# Table 21 — Coverage mechanism across coverage-limited traits (family-held-out)

For each trait: a linear probe's ranking quality (AUROC) far exceeds its macro-F1, positives are concentrated in a few training families (Gini, top-5 share), and recall on novel-family positives jumps from the no-neighbour-coverage bin to the high-coverage bin. Same mechanism in every case: cross-clade recall tracks neighbour coverage, not model capacity.

| Trait | Test pos rate | AUROC | Macro-F1 | AUROC−F1 | Pos Gini | Top-5 share | Recall (no cov) | Recall (high cov) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pathogenicity_human` | 0.061 | 0.761 | 0.581 | +0.179 | 0.902 | 42.6% | 0.108 (n=37) | 0.720 (n=25) |
| `pathogenicity_animal` | 0.044 | 0.773 | 0.569 | +0.205 | 0.903 | 40.3% | 0.176 (n=34) | 0.607 (n=28) |
| `motility` | 0.363 | 0.696 | 0.636 | +0.060 | 0.845 | 44.4% | 0.000 (n=9) | 0.631 (n=523) |
