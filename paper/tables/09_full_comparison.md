# Table 9 — Full per-trait comparison

Best score across all 26 model variants in `runs/` for each metric the head emits, vs every directly-comparable prior published number. RMSE is lower-is-better; everything else is higher-is-better.

Legend: **⬆** = we beat the listed prior · **⬇** = we lose to the listed prior · **~** = prior method exists but no score published · **🆕** = no directly-comparable prior exists

| Trait | Metric | Best ours | (from run) | Prior method | Prior score | Verdict |
|---|---|---:|---|---|---:|:---:|
| `amr_phenotype` | f1 | **0.448** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `amr_phenotype` | f1_macro | **0.561** | `esm2-150M-species-bal` | — | — | 🆕 |
| `biosafety_level` | acc | **0.911** | `esm2-150M-genus` | — | — | 🆕 |
| `biosafety_level` | f1 | **0.722** | `hybrid-bacformer-family-sel` | — | — | 🆕 |
| `carbon_utilization` | f1 | **0.577** | `esm2-150M-species-bal` | — | — | 🆕 |
| `carbon_utilization` | f1_macro | **0.534** | `esm2-150M-species-bal` | — | — | 🆕 |
| `catalase` | acc | **0.910** | `esm2-150M-200p-species-sel` | — | — | 🆕 |
| `catalase` | f1 | **0.949** | `esm2-150M-200p-species-sel` | — | — | 🆕 |
| `cell_shape` | acc | **0.842** | `chance-majority-family` | — | — | 🆕 |
| `cell_shape` | f1 | **0.351** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `country` | acc | **0.227** | `esm2-150M-genus` | — | — | 🆕 |
| `country` | f1 | **0.047** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `cultivation_medium` | f1 | **0.472** | `hybrid-v2-species-sel` | microbe-model v0 (this team) | — | ~ |
| `cytochrome_oxidase` | acc | **0.857** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `cytochrome_oxidase` | f1 | **0.890** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `fatty_acid_profile` | rmse | **0.092** | `esm2-150M-genus-h1024` | — | — | 🆕 |
| `gram_stain` | f1 | **0.679** | `esm2-150M-200p-species-sel` | Koblitz 2025 | — | ~ |
| `gram_stain` | f1 | **0.679** | `esm2-150M-200p-species-sel` | Traitar 2016 | 0.960 | ⬇ |
| `gram_stain` | auroc | — | _no run_ | Brbic 2016 | 0.990 | (metric not tracked) |
| `halophily` | acc | **0.680** | `esm2-150M-genus` | — | — | 🆕 |
| `halophily` | f1 | **0.517** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `isolation_source` | acc | **0.600** | `esm2-150M-family` | — | — | 🆕 |
| `isolation_source` | f1 | **0.287** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `metabolite_production` | f1 | **0.167** | `esm2-150M-species-bal` | — | — | 🆕 |
| `metabolite_production` | f1_macro | **0.717** | `esm2-150M-family-bal` | — | — | 🆕 |
| `motility` | f1 | **0.747** | `hybrid-v2-species-sel` | Koblitz 2025 | — | ~ |
| `motility` | f1 | **0.747** | `hybrid-v2-species-sel` | Traitar 2016 | 0.860 | ⬇ |
| `oxygen_tolerance` | acc | **0.865** | `esm2-150M-family` | Wan 2025 Genomics | — | ~ |
| `oxygen_tolerance` | f1 | **0.405** | `hybrid-bacformer-species-sel` | Koblitz 2025 (Commun Biol) | — | ~ |
| `pathogenicity_animal` | acc | **0.962** | `chance-majority-family` | — | — | 🆕 |
| `pathogenicity_animal` | f1 | **0.560** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `pathogenicity_human` | acc | **0.936** | `esm2-150M-genus` | PathogenFinder 2013 | 0.880 | ⬆ |
| `ph_class` | acc | **0.650** | `esm2-150M-genus` | — | — | 🆕 |
| `ph_class` | f1 | **0.579** | `hybrid-v2-genus-sel` | — | — | 🆕 |
| `pigmentation` | acc | **0.742** | `hybrid-v2-species-sel` | — | — | 🆕 |
| `pigmentation` | f1 | **0.849** | `esm2-150M-200p-family-sel` | — | — | 🆕 |
| `sporulation` | f1 | **0.944** | `hybrid-bacformer-species-sel` | Koblitz 2025 | — | ~ |
| `sporulation` | f1 | **0.944** | `hybrid-bacformer-species-sel` | Traitar 2016 | 0.930 | ⬆ |
| `temperature_class` | acc | **0.960** | `esm2-150M-species` | — | — | 🆕 |
| `temperature_class` | f1 | **0.563** | `hybrid-v2-species-sel` | — | — | 🆕 |

## Summary counters

- Total comparison rows: **39**
- Rows where we beat the prior (⬆): **2** (2 distinct traits)
- Rows where we lose to the prior (⬇): **2**
- Rows where prior exists but no published score (~): **6**
- Rows where no directly-comparable prior exists (🆕): **29**

- Distinct traits in benchmark: **21**
  - with at least one directly-comparable prior: **6** (29%)
  - **white-space** (no direct prior published): **15** (71%)

## Notes

- 'Best ours' picks the strongest score across all configurations in `runs/` for that trait+metric. Different traits' best runs may come from different model variants (selective class weights, h1024 head, different split level, etc.) — see the run name column.
- Multilabel heads can report both `f1` (sample-averaged) and `f1_macro` (label-averaged). Both rows are emitted when a prior reports the corresponding metric.
- Numbers marked '~' indicate the prior method exists in the literature but the published paper does not report the metric we need for a head-to-head comparison.
