# Per-layer adaptation-rank profile

split=species  pooling=None  batch=256  cov_batches=8  ft_steps=300

`kron_estimate` (cheap, one backward) predicts `grad_reff`; `deltaW_reff` is the empirical adaptation rank after fine-tuning; `suggested_lora_rank = ceil(deltaW_reff)`.

| layer | shape | act_reff | err_reff | kron_est | grad_reff | dW_reff | LoRA r | dW frac |
|-------|-------|---------:|---------:|---------:|----------:|--------:|-------:|--------:|
| encoder.0 | 512x640 | 1.15 | 99.86 | 1.15 | 1.01 | 7.95 | 8 | 0.016 |
| encoder.3 | 256x512 | 4.87 | 70.01 | 4.87 | 1.03 | 9.84 | 10 | 0.038 |
| heads.gram_stain | 3x256 | 7.66 | 1.74 | 1.74 | 1.02 | 1.85 | 2 | 0.617 |
| heads.cell_shape | 7x256 | 7.66 | 1.97 | 1.97 | 1.0 | 3.16 | 4 | 0.452 |
| heads.motility | 1x256 | 7.66 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| heads.sporulation | 1x256 | 7.66 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| heads.pigmentation | 1x256 | 7.66 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| heads.oxygen_tolerance | 6x256 | 7.66 | 2.11 | 2.11 | 1.0 | 2.93 | 3 | 0.488 |
| heads.catalase | 1x256 | 7.66 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| heads.cytochrome_oxidase | 1x256 | 7.66 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| heads.halophily | 4x256 | 7.66 | 2.26 | 2.26 | 1.01 | 2.48 | 3 | 0.621 |
| heads.temperature_class | 5x256 | 7.66 | 1.28 | 1.28 | 1.0 | 2.55 | 3 | 0.511 |
| heads.ph_class | 3x256 | 7.66 | 1.76 | 1.76 | 1.01 | 1.9 | 2 | 0.635 |
| heads.cultivation_medium | 200x256 | 7.66 | 1.22 | 1.22 | 1.0 | 1.49 | 2 | 0.007 |
| heads.carbon_utilization | 80x256 | 7.66 | 32.35 | 7.66 | 1.05 | 8.43 | 9 | 0.105 |
| heads.metabolite_production | 50x256 | 7.66 | 3.39 | 3.39 | 1.01 | 2.67 | 3 | 0.053 |
| heads.amr_phenotype | 30x256 | 7.66 | 14.78 | 7.66 | 1.37 | 7.52 | 8 | 0.251 |
| heads.biosafety_level | 4x256 | 7.66 | 1.75 | 1.75 | 1.0 | 1.65 | 2 | 0.412 |
| heads.pathogenicity_human | 1x256 | 7.66 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| heads.pathogenicity_animal | 1x256 | 7.66 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| heads.isolation_source | 20x256 | 7.66 | 7.75 | 7.66 | 1.06 | 4.19 | 5 | 0.21 |
| heads.country | 100x256 | 7.66 | 25.65 | 7.66 | 1.06 | 5.18 | 6 | 0.052 |
| heads.fatty_acid_profile | 30x256 | 7.66 | 8.73 | 7.66 | 1.06 | 24.64 | 25 | 0.821 |
