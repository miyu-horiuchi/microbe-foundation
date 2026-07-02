# Table 29 -- Retrieval-augmented prediction head (lever #1)

k=10. Distance-weighted (phylogeny-aware) k-NN blended with the linear probe, `alpha*probe + (1-alpha)*knn_phylo`, alpha tuned on {level}-val and scored once on {level}-test. `delta` columns are blend minus probe-alone.

## Per-trait, per-split

| Level | Trait | Test n | Pos | a* | Probe F1 | kNN-uni F1 | kNN-phylo F1 | Blend F1 | dF1 | Probe AUC | Blend AUC | dAUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| species | `motility` | 2575 | 0.42 | 0.2 | 0.734 | 0.751 | 0.757 | 0.764 | +0.030 | 0.805 | 0.837 | +0.032 |
| species | `sporulation` | 1329 | 0.33 | 0.4 | 0.901 | 0.928 | 0.922 | 0.934 | +0.033 | 0.955 | 0.978 | +0.023 |
| species | `catalase` | 2244 | 0.85 | 0.2 | 0.721 | 0.791 | 0.784 | 0.795 | +0.075 | 0.802 | 0.852 | +0.050 |
| species | `cytochrome_oxidase` | 2130 | 0.64 | 0.3 | 0.749 | 0.807 | 0.808 | 0.813 | +0.064 | 0.818 | 0.862 | +0.044 |
| species | `pigmentation` | 1166 | 0.72 | 0.0 | 0.499 | 0.576 | 0.552 | 0.552 | +0.053 | 0.520 | 0.599 | +0.079 |
| species | `pathogenicity_human` | 3084 | 0.05 | 0.2 | 0.549 | 0.648 | 0.643 | 0.646 | +0.097 | 0.783 | 0.843 | +0.060 |
| species | `pathogenicity_animal` | 3144 | 0.06 | 0.2 | 0.592 | 0.638 | 0.674 | 0.682 | +0.090 | 0.825 | 0.894 | +0.069 |
| genus | `motility` | 1968 | 0.43 | 0.0 | 0.687 | 0.695 | 0.707 | 0.707 | +0.020 | 0.755 | 0.754 | -0.001 |
| genus | `sporulation` | 979 | 0.27 | 0.4 | 0.859 | 0.854 | 0.851 | 0.873 | +0.014 | 0.935 | 0.953 | +0.018 |
| genus | `catalase` | 1656 | 0.83 | 0.5 | 0.700 | 0.772 | 0.771 | 0.786 | +0.086 | 0.800 | 0.814 | +0.014 |
| genus | `cytochrome_oxidase` | 1573 | 0.69 | 0.4 | 0.730 | 0.773 | 0.777 | 0.778 | +0.048 | 0.790 | 0.835 | +0.044 |
| genus | `pigmentation` | 930 | 0.74 | 1.0 | 0.497 | 0.525 | 0.532 | 0.497 | +0.000 | 0.508 | 0.508 | +0.000 |
| genus | `pathogenicity_human` | 2511 | 0.05 | 0.0 | 0.547 | 0.522 | 0.535 | 0.535 | -0.012 | 0.758 | 0.676 | -0.082 |
| genus | `pathogenicity_animal` | 2567 | 0.04 | 0.4 | 0.564 | 0.538 | 0.561 | 0.577 | +0.014 | 0.789 | 0.789 | -0.001 |
| family | `motility` | 1673 | 0.36 | 0.5 | 0.636 | 0.627 | 0.624 | 0.656 | +0.020 | 0.696 | 0.721 | +0.025 |
| family | `sporulation` | 809 | 0.25 | 0.4 | 0.827 | 0.820 | 0.821 | 0.848 | +0.022 | 0.925 | 0.935 | +0.010 |
| family | `catalase` | 1442 | 0.82 | 0.4 | 0.686 | 0.690 | 0.677 | 0.722 | +0.036 | 0.793 | 0.797 | +0.005 |
| family | `cytochrome_oxidase` | 1352 | 0.68 | 0.6 | 0.758 | 0.809 | 0.805 | 0.800 | +0.042 | 0.837 | 0.865 | +0.028 |
| family | `pigmentation` | 758 | 0.74 | 0.0 | 0.487 | 0.487 | 0.488 | 0.488 | +0.001 | 0.500 | 0.574 | +0.074 |
| family | `pathogenicity_human` | 2158 | 0.06 | 0.5 | 0.581 | 0.507 | 0.514 | 0.575 | -0.006 | 0.761 | 0.760 | -0.001 |
| family | `pathogenicity_animal` | 2172 | 0.04 | 0.4 | 0.569 | 0.524 | 0.522 | 0.556 | -0.013 | 0.773 | 0.764 | -0.009 |

## Mean delta over traits, by split (does retrieval help more OOD?)

| Level | mean dF1 | mean dAUROC | mean a* | n traits |
|---|---:|---:|---:|---:|
| species | +0.063 | +0.051 | 0.21 | 7 |
| genus | +0.024 | -0.001 | 0.39 | 7 |
| family | +0.014 | +0.019 | 0.40 | 7 |
