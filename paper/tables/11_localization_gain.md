# Table 11 — Sparse gene-family localization audit

Audit feature matrix: `data/eggnog_features_6738.npz` (6,738 genomes × 24,854 eggNOG orthologous groups).

Localization proxy: for each scalar trait, fit an L1-regularized gene-family classifier on the species-train split, evaluate it on the species-test split, then measure how concentrated the absolute coefficient mass is. `localization_score` is the top-10 coefficient-mass share when the sparse fit is stable; otherwise it falls back to the univariate class-conditional association share. Multilabel and regression-vector heads are excluded from this scalar audit.

Spearman correlation between `localization_score` and species-level attention gain: rho = **0.494**, p = **0.052** (n=16 traits).

| Trait | Class | Type | Source | Audit labels | Sparse train/test | Sparse macro-F1 | Localization | Sparse nonzero | n80 | Species Δ | Genus Δ | Family Δ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `catalase` | compositional | binary | sparse_linear | 3,178 | 1,426/893 | 0.842 | 0.013 | 6609 | 2932 | 0.004 | 0.000 | -0.001 |
| `biosafety_level` | machinery | multiclass | sparse_linear | 6,223 | 2,666/1,798 | 0.792 | 0.012 | 6379 | 2800 | 0.037 | 0.050 | -0.005 |
| `cytochrome_oxidase` | compositional | binary | sparse_linear | 3,129 | 1,442/854 | 0.822 | 0.012 | 5990 | 2611 | 0.032 | -0.002 | -0.007 |
| `motility` | compositional | binary | sparse_linear | 3,617 | 1,535/1,059 | 0.785 | 0.011 | 5943 | 2641 | 0.064 | 0.034 | 0.019 |
| `pathogenicity_human` | machinery | binary | sparse_linear | 5,409 | 2,198/1,620 | 0.679 | 0.011 | 7229 | 3127 | 0.207 | 0.128 | 0.028 |
| `pathogenicity_animal` | machinery | binary | sparse_linear | 5,511 | 2,224/1,659 | 0.773 | 0.009 | 8561 | 3737 | 0.338 | 0.298 | 0.051 |
| `sporulation` | compositional | binary | sparse_linear | 1,705 | 620/558 | 0.941 | 0.006 | 10046 | 4451 | 0.021 | 0.027 | 0.023 |
| `pigmentation` | compositional | binary | sparse_linear | 1,651 | 602/543 | 0.532 | 0.006 | 12582 | 5475 | 0.009 | 0.007 | -0.001 |
| `halophily` | compositional | multiclass | sparse_linear | 2,251 | 864/717 | 0.475 | 0.005 | 19156 | 8207 | 0.040 | 0.050 | 0.000 |
| `ph_class` | compositional | multiclass | sparse_linear | 1,854 | 624/610 | 0.574 | 0.005 | 20606 | 8975 | 0.021 | 0.012 | 0.089 |
| `gram_stain` | compositional | multiclass | sparse_linear | 3,820 | 1,612/1,126 | 0.658 | 0.004 | 21412 | 8544 | 0.002 | 0.001 | 0.008 |
| `temperature_class` | compositional | multiclass | sparse_linear | 6,693 | 2,956/1,888 | 0.530 | 0.004 | 23512 | 10701 | 0.014 | -0.002 | -0.007 |
| `cell_shape` | compositional | multiclass | univariate | 3,723 | — | — | 0.002 | — | 12736 | 0.015 | 0.017 | 0.002 |
| `oxygen_tolerance` | compositional | multiclass | univariate | 4,736 | — | — | 0.002 | — | 14131 | 0.004 | 0.028 | -0.032 |
| `country` | metadata | multiclass | univariate | 5,822 | — | — | 0.001 | — | 16825 | 0.001 | -0.001 | 0.001 |
| `isolation_source` | metadata | multiclass | univariate | 6,512 | — | — | 0.001 | — | 14875 | 0.019 | -0.003 | 0.018 |
