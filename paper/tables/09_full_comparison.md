# Table 9 — Full per-trait comparison

Best score across all 14 model variants in `runs/` for each metric the head emits, vs every directly-comparable prior published number. RMSE is lower-is-better; everything else is higher-is-better.

Legend: **⬆** = we beat the listed prior · **⬇** = we lose to the listed prior · **~** = prior method exists but no score published · **🆕** = no directly-comparable prior exists

| Trait | Metric | Best ours | (from run) | Prior method | Prior score | Verdict |
|---|---|---:|---|---|---:|:---:|
| `amr_phenotype` | f1 | **0.403** | `esm2-150M-species-bal` | — | — | 🆕 |
| `amr_phenotype` | f1_macro | **0.561** | `esm2-150M-species-bal` | — | — | 🆕 |
| `biosafety_level` | acc | **0.911** | `esm2-150M-genus` | — | — | 🆕 |
| `biosafety_level` | f1 | **0.606** | `esm2-150M-family` | — | — | 🆕 |
| `carbon_utilization` | f1 | **0.577** | `esm2-150M-species-bal` | — | — | 🆕 |
| `carbon_utilization` | f1_macro | **0.534** | `esm2-150M-species-bal` | — | — | 🆕 |
| `catalase` | acc | **0.906** | `esm2-150M-species` | — | — | 🆕 |
| `catalase` | f1 | **0.947** | `esm2-150M-species` | — | — | 🆕 |
| `cell_shape` | acc | **0.842** | `chance-majority-family` | — | — | 🆕 |
| `cell_shape` | f1 | **0.276** | `esm2-150M-species-h1024` | — | — | 🆕 |
| `country` | acc | **0.227** | `esm2-150M-genus` | — | — | 🆕 |
| `country` | f1 | **0.024** | `esm2-150M-species-h1024` | — | — | 🆕 |
| `cultivation_medium` | f1 | **0.427** | `esm2-150M-species-h1024` | microbe-model v0 (this team) | — | ~ |
| `cytochrome_oxidase` | acc | **0.843** | `esm2-150M-family-h1024` | — | — | 🆕 |
| `cytochrome_oxidase` | f1 | **0.889** | `esm2-150M-family-bal` | — | — | 🆕 |
| `fatty_acid_profile` | rmse | **0.092** | `esm2-150M-genus-h1024` | — | — | 🆕 |
| `gram_stain` | f1 | **0.656** | `esm2-150M-species-sel` | Koblitz 2025 | — | ~ |
| `gram_stain` | f1 | **0.656** | `esm2-150M-species-sel` | Traitar 2016 | 0.960 | ⬇ |
| `gram_stain` | auroc | — | _no run_ | Brbic 2016 | 0.990 | (metric not tracked) |
| `halophily` | acc | **0.680** | `esm2-150M-genus` | — | — | 🆕 |
| `halophily` | f1 | **0.464** | `esm2-150M-species-h1024` | — | — | 🆕 |
| `isolation_source` | acc | **0.600** | `esm2-150M-family` | — | — | 🆕 |
| `isolation_source` | f1 | **0.239** | `esm2-150M-family` | — | — | 🆕 |
| `metabolite_production` | f1 | **0.167** | `esm2-150M-species-bal` | — | — | 🆕 |
| `metabolite_production` | f1_macro | **0.717** | `esm2-150M-family-bal` | — | — | 🆕 |
| `motility` | f1 | **0.724** | `esm2-150M-species-bal` | Koblitz 2025 | — | ~ |
| `motility` | f1 | **0.724** | `esm2-150M-species-bal` | Traitar 2016 | 0.860 | ⬇ |
| `oxygen_tolerance` | acc | **0.865** | `esm2-150M-family` | Wan 2025 Genomics | — | ~ |
| `oxygen_tolerance` | f1 | **0.321** | `esm2-150M-species-sel` | Koblitz 2025 (Commun Biol) | — | ~ |
| `pathogenicity_animal` | acc | **0.962** | `chance-majority-family` | — | — | 🆕 |
| `pathogenicity_animal` | f1 | **0.341** | `esm2-150M-species-h1024` | — | — | 🆕 |
| `pathogenicity_human` | acc | **0.936** | `esm2-150M-genus` | PathogenFinder 2013 | 0.880 | ⬆ |
| `ph_class` | acc | **0.650** | `esm2-150M-genus` | — | — | 🆕 |
| `ph_class` | f1 | **0.564** | `esm2-150M-species` | — | — | 🆕 |
| `pigmentation` | acc | **0.735** | `chance-majority-family` | — | — | 🆕 |
| `pigmentation` | f1 | **0.847** | `chance-majority-family` | — | — | 🆕 |
| `sporulation` | f1 | **0.907** | `esm2-150M-species` | Koblitz 2025 | — | ~ |
| `sporulation` | f1 | **0.907** | `esm2-150M-species` | Traitar 2016 | 0.930 | ⬇ |
| `temperature_class` | acc | **0.960** | `esm2-150M-species` | — | — | 🆕 |
| `temperature_class` | f1 | **0.509** | `esm2-150M-genus` | — | — | 🆕 |

## Summary counters

- Total comparison rows: **39**
- Rows where we beat the prior (⬆): **1** (1 distinct traits)
- Rows where we lose to the prior (⬇): **3**
- Rows where prior exists but no published score (~): **6**
- Rows where no directly-comparable prior exists (🆕): **29**

- Distinct traits in benchmark: **21**
  - with at least one directly-comparable prior: **6** (29%)
  - **white-space** (no direct prior published): **15** (71%)

## Notes

- 'Best ours' picks the strongest score across all configurations in `runs/` for that trait+metric. Different traits' best runs may come from different model variants (selective class weights, h1024 head, different split level, etc.) — see the run name column.
- Multilabel heads can report both `f1` (sample-averaged) and `f1_macro` (label-averaged). Both rows are emitted when a prior reports the corresponding metric.
- Numbers marked '~' indicate the prior method exists in the literature but the published paper does not report the metric we need for a head-to-head comparison.
