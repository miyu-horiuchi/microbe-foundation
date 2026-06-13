# Table 20 — Per-trait failure-mode classification

Each trait's in-distribution ceiling (species-split F1) vs cross-clade generalization (family-split F1). `decay = species - family`. Modes: **solved** (family F1 ≥ 0.70), **label-ceiling** (species F1 < 0.30 — poor even in-distribution; a label/task problem, not the model), **coverage-limited** (learnable but collapses off-clade, decay ≥ 0.10 — needs training-clade coverage), **moderate-flat** (partial signal, no collapse, no ceiling).

Summary: 4 solved, 4 coverage-limited, 6 moderate-flat, 6 label-ceiling

| Trait | Species F1 | Genus F1 | Family F1 | Decay | Failure mode |
|---|---:|---:|---:|---:|:--|
| `pathogenicity_animal` | 0.510 | 0.356 | 0.190 | 0.319 | coverage-limited |
| `pathogenicity_human` | 0.333 | 0.209 | 0.063 | 0.270 | coverage-limited |
| `temperature_class` | 0.502 | 0.506 | 0.321 | 0.181 | coverage-limited |
| `motility` | 0.766 | 0.702 | 0.589 | 0.177 | coverage-limited |
| `ph_class` | 0.545 | 0.520 | 0.455 | 0.090 | moderate-flat |
| `carbon_utilization` | 0.545 | 0.539 | 0.487 | 0.058 | moderate-flat |
| `amr_phenotype` | 0.380 | 0.368 | 0.347 | 0.033 | moderate-flat |
| `halophily` | 0.465 | 0.454 | 0.440 | 0.025 | moderate-flat |
| `gram_stain` | 0.643 | 0.641 | 0.637 | 0.006 | moderate-flat |
| `biosafety_level` | 0.489 | 0.479 | 0.628 | -0.139 | moderate-flat |
| `cultivation_medium` | 0.278 | 0.247 | 0.133 | 0.145 | label-ceiling |
| `oxygen_tolerance` | 0.274 | 0.296 | 0.180 | 0.093 | label-ceiling |
| `cell_shape` | 0.190 | 0.174 | 0.141 | 0.048 | label-ceiling |
| `metabolite_production` | 0.102 | 0.095 | 0.076 | 0.027 | label-ceiling |
| `country` | 0.020 | 0.016 | 0.017 | 0.003 | label-ceiling |
| `isolation_source` | 0.193 | 0.176 | 0.232 | -0.039 | label-ceiling |
| `sporulation` | 0.920 | 0.846 | 0.819 | 0.101 | solved |
| `catalase` | 0.948 | 0.943 | 0.921 | 0.028 | solved |
| `cytochrome_oxidase` | 0.871 | 0.874 | 0.881 | -0.010 | solved |
| `pigmentation` | 0.836 | 0.837 | 0.846 | -0.010 | solved |
