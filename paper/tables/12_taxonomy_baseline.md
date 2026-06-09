# Table 12 — Taxonomy-majority baseline

Baseline: for each test genome, predict using the most specific taxonomy level observed in training (`species`, then `genus`, `family`, `order`, `class`, `phylum`, `domain`), falling back to the train-set majority. This is a confounding diagnostic, not a proposed model.

| Split | Trait | Class | Test labels | Taxonomy acc | Taxonomy macro-F1 | Attention metric | Attention score |
|---|---|---|---:|---:|---:|---|---:|
| family | `biosafety_level` | machinery | 4,168 | 0.839 | 0.402 | f1 | 0.628 |
| family | `catalase` | compositional | 2,236 | 0.841 | 0.705 | f1 | 0.921 |
| family | `cell_shape` | compositional | 2,455 | 0.837 | 0.220 | f1 | 0.141 |
| family | `country` | metadata | 7,042 | 0.183 | 0.020 | f1 | 0.017 |
| family | `cytochrome_oxidase` | compositional | 2,164 | 0.808 | 0.776 | f1 | 0.881 |
| family | `gram_stain` | compositional | 2,584 | 0.968 | 0.645 | f1 | 0.637 |
| family | `halophily` | compositional | 1,495 | 0.592 | 0.302 | f1 | 0.440 |
| family | `isolation_source` | metadata | 7,894 | 0.034 | 0.000 | f1 | 0.232 |
| family | `motility` | compositional | 2,387 | 0.681 | 0.666 | f1 | 0.589 |
| family | `oxygen_tolerance` | compositional | 3,953 | 0.841 | 0.236 | f1 | 0.180 |
| family | `pathogenicity_animal` | machinery | 3,595 | 0.956 | 0.495 | f1 | 0.190 |
| family | `pathogenicity_human` | machinery | 3,570 | 0.930 | 0.482 | f1 | 0.063 |
| family | `ph_class` | compositional | 1,498 | 0.596 | 0.305 | f1 | 0.455 |
| family | `pigmentation` | compositional | 917 | 0.730 | 0.544 | f1 | 0.846 |
| family | `sporulation` | compositional | 1,028 | 0.800 | 0.782 | f1 | 0.819 |
| family | `temperature_class` | compositional | 7,590 | 0.927 | 0.309 | f1 | 0.321 |
| genus | `biosafety_level` | machinery | 4,418 | 0.869 | 0.445 | f1 | 0.479 |
| genus | `catalase` | compositional | 2,401 | 0.871 | 0.804 | f1 | 0.943 |
| genus | `cell_shape` | compositional | 2,712 | 0.795 | 0.241 | f1 | 0.174 |
| genus | `country` | metadata | 6,981 | 0.183 | 0.017 | f1 | 0.016 |
| genus | `cytochrome_oxidase` | compositional | 2,373 | 0.794 | 0.770 | f1 | 0.874 |
| genus | `gram_stain` | compositional | 2,857 | 0.963 | 0.643 | f1 | 0.641 |
| genus | `halophily` | compositional | 1,966 | 0.545 | 0.347 | f1 | 0.454 |
| genus | `isolation_source` | metadata | 7,888 | 0.043 | 0.001 | f1 | 0.176 |
| genus | `motility` | compositional | 2,603 | 0.751 | 0.742 | f1 | 0.702 |
| genus | `oxygen_tolerance` | compositional | 3,820 | 0.793 | 0.313 | f1 | 0.296 |
| genus | `pathogenicity_animal` | machinery | 4,206 | 0.942 | 0.650 | f1 | 0.356 |
| genus | `pathogenicity_human` | machinery | 4,126 | 0.914 | 0.562 | f1 | 0.209 |
| genus | `ph_class` | compositional | 1,808 | 0.562 | 0.485 | f1 | 0.520 |
| genus | `pigmentation` | compositional | 1,139 | 0.718 | 0.532 | f1 | 0.837 |
| genus | `sporulation` | compositional | 1,208 | 0.865 | 0.843 | f1 | 0.846 |
| genus | `temperature_class` | compositional | 7,664 | 0.941 | 0.474 | f1 | 0.506 |
| species | `biosafety_level` | machinery | 4,933 | 0.883 | 0.503 | f1 | 0.489 |
| species | `catalase` | compositional | 2,958 | 0.904 | 0.814 | f1 | 0.948 |
| species | `cell_shape` | compositional | 3,337 | 0.856 | 0.400 | f1 | 0.190 |
| species | `country` | metadata | 7,125 | 0.186 | 0.037 | f1 | 0.020 |
| species | `cytochrome_oxidase` | compositional | 2,844 | 0.832 | 0.816 | f1 | 0.871 |
| species | `gram_stain` | compositional | 3,466 | 0.960 | 0.643 | f1 | 0.643 |
| species | `halophily` | compositional | 2,493 | 0.676 | 0.527 | f1 | 0.465 |
| species | `isolation_source` | metadata | 7,871 | 0.052 | 0.004 | f1 | 0.193 |
| species | `motility` | compositional | 3,226 | 0.799 | 0.794 | f1 | 0.766 |
| species | `oxygen_tolerance` | compositional | 4,337 | 0.826 | 0.301 | f1 | 0.274 |
| species | `pathogenicity_animal` | machinery | 4,571 | 0.929 | 0.730 | f1 | 0.510 |
| species | `pathogenicity_human` | machinery | 4,484 | 0.914 | 0.647 | f1 | 0.333 |
| species | `ph_class` | compositional | 2,141 | 0.584 | 0.507 | f1 | 0.545 |
| species | `pigmentation` | compositional | 1,419 | 0.686 | 0.575 | f1 | 0.836 |
| species | `sporulation` | compositional | 1,620 | 0.951 | 0.945 | f1 | 0.920 |
| species | `temperature_class` | compositional | 7,931 | 0.952 | 0.575 | f1 | 0.502 |
