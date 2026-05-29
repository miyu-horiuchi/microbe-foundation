# Table 8 — Benchmark: chance / random / ESM-2 / prior

Family-held-out split. F1 is the honest metric on imbalanced heads; accuracy can be misleading when one class dominates (see `pathogenicity_*`). All comparators reported here are directly comparable in metric and evaluation protocol (see `paper/tables/06_vs_prior.md` for the full list).

Legend:  **bold** = best non-chance row;  ⬆ = our model beats prior;  ⬇ = our model loses to prior;  🆕 = no published prior comparator

| Trait | Metric | Chance | Random-640 | **ESM-2 150M (ours)** | Prior best | Verdict |
|---|---|---:|---:|---:|---:|:---:|
| `amr_phenotype` | f1_macro | 0.000 | 0.379 | **0.492** | — | 🆕 |
| `biosafety_level` | acc | 0.835 | 0.821 | **0.827** | — | 🆕 |
| `carbon_utilization` | f1 | 0.000 | 0.412 | **0.498** | — | 🆕 |
| `catalase` | acc | 0.786 | 0.758 | **0.872** | — | 🆕 |
| `cell_shape` | f1 | 0.131 | 0.141 | **0.230** | — | 🆕 |
| `country` | acc | 0.149 | 0.096 | **0.013** | — | 🆕 |
| `cultivation_medium` | f1 | 0.000 | 0.001 | **0.332** | — | 🆕 |
| `cytochrome_oxidase` | acc | 0.319 | 0.523 | **0.837** | — | 🆕 |
| `gram_stain` | f1 | 0.268 | 0.324 | **0.630** | 0.960 | ⬇ (Traitar 2016) |
| `halophily` | acc | 0.594 | 0.519 | **0.389** | — | 🆕 |
| `isolation_source` | acc | 0.016 | 0.278 | **0.363** | — | 🆕 |
| `metabolite_production` | f1_macro | 0.000 | 0.636 | **0.692** | — | 🆕 |
| `motility` | f1 | 0.000 | 0.362 | **0.556** | 0.860 | ⬇ (Traitar 2016) |
| `oxygen_tolerance` | f1 | 0.151 | 0.162 | **0.282** | — | 🆕 |
| `pathogenicity_animal` | f1 | 0.000 | 0.052 | **0.235** | — | 🆕 |
| `pathogenicity_human` | acc | 0.932 | 0.889 | **0.813** | 0.880 | ⬇ (PathogenFinder 2013) |
| `ph_class` | acc | 0.573 | 0.513 | **0.539** | — | 🆕 |
| `pigmentation` | acc | 0.735 | 0.625 | **0.732** | — | 🆕 |
| `sporulation` | f1 | 0.000 | 0.280 | **0.777** | 0.930 | ⬇ (Traitar 2016) |
| `temperature_class` | f1 | 0.191 | 0.190 | **0.356** | — | 🆕 |

## Headline counters

- Traits evaluated: **20**
- Where a directly-comparable prior exists: **4**
  - We **beat** prior: **0**
  - We **lose** to prior: **4**
- Traits where we set the **first published baseline** ('white-space'): **16** (80% of evaluated traits)

## Aggregate improvement (mean across heads, where the same metric exists across rows)

| Metric | Chance | Random-640 | **ESM-2 150M (ours)** | Δ vs chance | Δ vs random |
|---|---:|---:|---:|---:|---:|
| acc (mean across heads) | 0.659 | 0.626 | **0.655** | -0.004 | +0.029 |
| f1 (mean across heads) | 0.160 | 0.288 | **0.437** | +0.277 | +0.149 |
| f1_macro (mean across heads) | 0.000 | 0.324 | **0.404** | +0.404 | +0.080 |

## Interpretation

- **Chance** = always predict the train-set majority class on test. Sets the no-signal floor.
- **Random-640** = our same downstream head trained on 640-dim Gaussian noise instead of ESM-2 embeddings. Sets the 'head capacity alone' floor.
- **ESM-2 150M (ours)** = `compute_esm2_features_mp.py` features + multi-task head with selective class weighting (`--class-weights --imbalance-threshold 5`).
- F1 — not accuracy — is the metric to read on imbalanced heads (`pathogenicity_*`, `sporulation`, `temperature_class`). Random-640 can match ESM-2 on accuracy there *just by predicting the majority class*. F1 separates real signal from class-imbalance gaming.
