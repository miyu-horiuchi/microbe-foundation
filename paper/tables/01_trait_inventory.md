# Table 1 — Trait Inventory

Schema version: 0.1.0

| Block | Trait | Head | Size | Est. labeled strains |
|---|---|---|---:|---:|
| morphology | `gram_stain` | multiclass | 3 | 15,900 |
| morphology | `cell_shape` | multiclass | 7 | 15,900 |
| morphology | `motility` | binary | 1 | 15,300 |
| morphology | `sporulation` | binary | 1 | 5,600 |
| morphology | `pigmentation` | binary | 1 | 6,800 |
| physiology | `oxygen_tolerance` | multiclass | 6 | 23,700 |
| physiology | `catalase` | binary | 1 | 14,600 |
| physiology | `cytochrome_oxidase` | binary | 1 | 13,100 |
| physiology | `halophily` | multiclass | 4 | 13,100 |
| growth | `temperature_class` | multiclass | 5 | 48,900 |
| growth | `ph_class` | multiclass | 3 | 7,800 |
| cultivation | `cultivation_medium` | multilabel | 200 | 28,000 |
| cultivation | `carbon_utilization` | multilabel | 80 | 29,600 |
| cultivation | `metabolite_production` | multilabel | 50 | 24,600 |
| cultivation | `amr_phenotype` | multilabel | 20 | 10,000 |
| safety | `biosafety_level` | multiclass | 4 | 26,200 |
| safety | `pathogenicity_human` | binary | 1 | 2,800 |
| safety | `pathogenicity_animal` | binary | 1 | 2,800 |
| ecology | `isolation_source` | multiclass | 20 | 60,400 |
| ecology | `country` | multiclass | 200 | 53,900 |
| chemotaxonomy | `fatty_acid_profile` | regression_vector | 30 | 7,200 |

_Total: 21 prediction heads across 7 blocks._
