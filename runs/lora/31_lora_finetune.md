# Table 31 -- Encoder adaptation: frozen vs LoRA vs full (the decisive lever)

split=family. Per-run metric JSONs from `finetune_lora.py`, aggregated as mean +/- std over seeds. `avg_primary` is the across-head headline metric (per-head primary: acc for binary/multiclass, F1 for multilabel, RMSE for regression). The LoRA-minus-frozen delta is the encoder-adaptation lever.

## Overall (avg_primary across heads)

| Mode | avg_primary | seeds |
|---|---|---:|
| frozen | 0.5680 +/- 0.0000 | 1 |

## Per-trait (primary metric, mean over seeds)

| Trait | frozen | LoRA-frozen |
|---|---|---:|
| `amr_phenotype` | 0.417 | n/a |
| `biosafety_level` | 0.886 | n/a |
| `carbon_utilization` | 0.487 | n/a |
| `catalase` | 0.818 | n/a |
| `cell_shape` | 0.820 | n/a |
| `country` | 0.067 | n/a |
| `cultivation_medium` | 0.110 | n/a |
| `cytochrome_oxidase` | 0.675 | n/a |
| `fatty_acid_profile` | 0.131 | n/a |
| `gram_stain` | 0.696 | n/a |
| `halophily` | 0.563 | n/a |
| `isolation_source` | 0.368 | n/a |
| `metabolite_production` | 0.120 | n/a |
| `motility` | 0.455 | n/a |
| `oxygen_tolerance` | 0.854 | n/a |
| `pathogenicity_animal` | 0.753 | n/a |
| `pathogenicity_human` | 0.699 | n/a |
| `ph_class` | 0.575 | n/a |
| `pigmentation` | 0.734 | n/a |
| `sporulation` | 0.747 | n/a |
| `temperature_class` | 0.951 | n/a |
