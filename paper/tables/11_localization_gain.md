# Table 11 — Gene-family localization audit

Audit feature matrix: `data/eggnog_features_6738.npz` (6,738 genomes × 24,854 eggNOG orthologous groups).

Localization proxy: for each scalar trait, compute a supervised gene-family association vector from class-conditional feature-rate differences, then measure how concentrated that vector is. `top10_share` is the share of association mass carried by the ten strongest gene families; `n80` is the number of gene families needed to reach 80% of association mass. Multilabel and regression-vector heads are excluded from this lightweight audit and should be handled by a heavier per-output analysis in the main-track version.

Spearman correlation between `top10_share` and species-level attention gain: rho = **0.447**, p = **0.083** (n=16 traits).

| Trait | Class | Type | Audit labeled genomes | top10 share | Gini | n80 | Species Δ | Genus Δ | Family Δ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `motility` | compositional | binary | 3,617 | 0.003 | 0.410 | 13115 | 0.064 | 0.034 | 0.019 |
| `biosafety_level` | machinery | multiclass | 6,223 | 0.003 | 0.449 | 11833 | 0.037 | 0.050 | -0.005 |
| `ph_class` | compositional | multiclass | 1,854 | 0.003 | 0.369 | 13512 | 0.021 | 0.012 | 0.089 |
| `pathogenicity_animal` | machinery | binary | 5,511 | 0.002 | 0.424 | 12518 | 0.338 | 0.298 | 0.051 |
| `catalase` | compositional | binary | 3,178 | 0.002 | 0.425 | 12891 | 0.004 | 0.000 | -0.001 |
| `pathogenicity_human` | machinery | binary | 5,409 | 0.002 | 0.469 | 11471 | 0.207 | 0.128 | 0.028 |
| `temperature_class` | compositional | multiclass | 6,693 | 0.002 | 0.349 | 14314 | 0.014 | -0.002 | -0.007 |
| `cell_shape` | compositional | multiclass | 3,723 | 0.002 | 0.415 | 12736 | 0.015 | 0.017 | 0.002 |
| `cytochrome_oxidase` | compositional | binary | 3,129 | 0.002 | 0.415 | 12688 | 0.032 | -0.002 | -0.007 |
| `pigmentation` | compositional | binary | 1,651 | 0.002 | 0.466 | 11494 | 0.009 | 0.007 | -0.001 |
| `oxygen_tolerance` | compositional | multiclass | 4,736 | 0.002 | 0.365 | 14131 | 0.004 | 0.028 | -0.032 |
| `gram_stain` | compositional | multiclass | 3,820 | 0.002 | 0.348 | 14369 | 0.002 | 0.001 | 0.008 |
| `sporulation` | compositional | binary | 1,705 | 0.002 | 0.409 | 13132 | 0.021 | 0.027 | 0.023 |
| `halophily` | compositional | multiclass | 2,251 | 0.002 | 0.324 | 14558 | 0.040 | 0.050 | 0.000 |
| `country` | metadata | multiclass | 5,822 | 0.001 | 0.215 | 16825 | 0.001 | -0.001 | 0.001 |
| `isolation_source` | metadata | multiclass | 6,512 | 0.001 | 0.318 | 14875 | 0.019 | -0.003 | 0.018 |
