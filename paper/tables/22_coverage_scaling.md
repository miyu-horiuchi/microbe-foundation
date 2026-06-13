# Table 22 — Training-clade coverage scaling (family-held-out)

Family-test performance vs the number of distinct training families. The *natural* curve grows families and genomes together (the real-world lever: collect more diverse data); the slope is the expected metric gain per doubling of training-clade coverage. The size-controlled column holds the genome budget fixed at the smallest-k pool to ask whether diversity helps *beyond* raw volume; for these rare-positive traits that budget is tiny (n shown), so it is underpowered and reported only as a directional check.

| Trait | #fam avail | n_train (k=min→max) | F1 (k=min→max) | F1 / 2x | AUROC (k=min→max) | AUROC / 2x | size-ctrl F1/2x (budget n) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pathogenicity_human` | 195 | 171→6159 | 0.481→0.580 (+0.099) | +0.017 | 0.617→0.761 (+0.144) | +0.026 | +0.001 (n=9) |
| `pathogenicity_animal` | 198 | 65→6300 | 0.484→0.569 (+0.085) | +0.013 | 0.588→0.773 (+0.185) | +0.021 | +0.008 (n=32) |
| `motility` | 175 | 48→5196 | 0.498→0.636 (+0.139) | +0.029 | 0.529→0.696 (+0.167) | +0.035 | +0.014 (n=15) |
