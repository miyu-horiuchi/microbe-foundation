# Table 13 — Within-clade matched pathogenicity control

This diagnostic restricts pathogenicity evaluation to test genomes whose genus or family is represented in training with both pathogenic and non-pathogenic labeled examples. It asks how much matched evaluation coverage exists after removing pure-clade shortcuts, and how strong a same-clade majority baseline remains on that matched subset.

Benchmark alignment: `data/esm2_features.npz` feature IDs when available.

| Trait | Split | Matched level | Test labels | Matched test | Coverage | Mixed train clades | Matched positive rate | Same-clade acc | Same-clade macro-F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pathogenicity_animal` | family | family | 2,172 | 0 | 0.000 | 47 | — | — | — |
| `pathogenicity_animal` | family | genus | 2,172 | 1 | 0.000 | 69 | 0.000 | 0.000 | — |
| `pathogenicity_animal` | genus | family | 2,567 | 1,000 | 0.390 | 56 | 0.039 | 0.886 | 0.621 |
| `pathogenicity_animal` | genus | genus | 2,567 | 0 | 0.000 | 60 | — | — | — |
| `pathogenicity_animal` | species | family | 3,144 | 1,543 | 0.491 | 68 | 0.093 | 0.890 | 0.716 |
| `pathogenicity_animal` | species | genus | 3,144 | 699 | 0.222 | 69 | 0.143 | 0.863 | 0.750 |
| `pathogenicity_human` | family | family | 2,158 | 0 | 0.000 | 47 | — | — | — |
| `pathogenicity_human` | family | genus | 2,158 | 1 | 0.000 | 91 | 0.000 | 1.000 | — |
| `pathogenicity_human` | genus | family | 2,511 | 1,020 | 0.406 | 58 | 0.066 | 0.873 | 0.495 |
| `pathogenicity_human` | genus | genus | 2,511 | 0 | 0.000 | 75 | — | — | — |
| `pathogenicity_human` | species | family | 3,084 | 1,823 | 0.591 | 78 | 0.070 | 0.886 | 0.625 |
| `pathogenicity_human` | species | genus | 3,084 | 850 | 0.276 | 101 | 0.108 | 0.804 | 0.642 |
