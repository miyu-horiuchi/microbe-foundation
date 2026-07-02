# Table 30 -- Coverage-efficient long-tail via semi-supervised self-training (lever #2)

conf=0.85  pseudo_weight=0.3. Pseudo-labels drawn only from unlabelled genomes in TRAINING clades (leak-safe). `delta` = self-train minus supervised on {level}-test.

## Per-trait, per-split

| Level | Trait | Train n | Pool | Pseudo | Test n | Sup F1 | ST F1 | dF1 | Sup AUC | ST AUC | dAUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| species | `motility` | 3620 | 5255 | 2904 | 2575 | 0.734 | 0.733 | -0.001 | 0.805 | 0.805 | -0.001 |
| species | `sporulation` | 1533 | 7342 | 6513 | 1329 | 0.901 | 0.900 | -0.001 | 0.955 | 0.955 | -0.000 |
| species | `pathogenicity_human` | 4406 | 4469 | 2478 | 3084 | 0.549 | 0.554 | +0.005 | 0.783 | 0.786 | +0.002 |
| species | `pathogenicity_animal` | 4469 | 4406 | 2762 | 3144 | 0.592 | 0.597 | +0.005 | 0.825 | 0.827 | +0.002 |
| genus | `motility` | 4888 | 6354 | 3738 | 1968 | 0.687 | 0.687 | +0.001 | 0.755 | 0.756 | +0.000 |
| genus | `sporulation` | 2271 | 8971 | 8055 | 979 | 0.859 | 0.858 | -0.001 | 0.935 | 0.934 | -0.002 |
| genus | `pathogenicity_human` | 5837 | 5405 | 3210 | 2511 | 0.547 | 0.555 | +0.008 | 0.758 | 0.761 | +0.003 |
| genus | `pathogenicity_animal` | 5922 | 5320 | 3350 | 2567 | 0.564 | 0.568 | +0.005 | 0.789 | 0.791 | +0.002 |
| family | `motility` | 5196 | 6595 | 3667 | 1673 | 0.636 | 0.633 | -0.003 | 0.696 | 0.696 | -0.000 |
| family | `sporulation` | 2473 | 9318 | 8223 | 809 | 0.827 | 0.826 | -0.001 | 0.925 | 0.926 | +0.001 |
| family | `pathogenicity_human` | 6159 | 5632 | 3237 | 2158 | 0.581 | 0.585 | +0.003 | 0.761 | 0.762 | +0.001 |
| family | `pathogenicity_animal` | 6300 | 5491 | 3521 | 2172 | 0.569 | 0.569 | +0.001 | 0.773 | 0.771 | -0.002 |

## Mean delta over traits, by split

| Level | mean dF1 | mean dAUROC | mean pseudo n |
|---|---:|---:|---:|
| species | +0.002 | +0.001 | 3664 |
| genus | +0.003 | +0.001 | 4588 |
| family | +0.000 | +0.000 | 4662 |
