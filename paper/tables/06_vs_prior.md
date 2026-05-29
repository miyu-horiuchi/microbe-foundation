# Table 6 — microbe-foundation vs. Prior Work

_Comparators curated in `prior_numbers.json`; comparability flagged per row._

## A. Directly comparable

| Trait | Metric | esm2-150M-family | Prior method | Prior score | Cite |
|---|---|---|---|---|---|
| `cultivation_medium` | f1 | 0.1363 | microbe-model v0 (this team) | — | — |
| `gram_stain` | f1 | 0.6326 | Koblitz 2025 | — | [@koblitz2025bacdiveml] |
| `gram_stain` | f1 | 0.6326 | Traitar 2016 | 0.9600 | [@weimann2016traitar] |
| `gram_stain` | auroc | — | Brbic 2016 | 0.9900 | [@brbic2016landscape] |
| `motility` | f1 | 0.5379 | Koblitz 2025 | — | [@koblitz2025bacdiveml] |
| `motility` | f1 | 0.5379 | Traitar 2016 | 0.8600 | [@weimann2016traitar] |
| `oxygen_tolerance` | acc | 0.8654 | Wan 2025 Genomics | — | [@wan2025oxygen] |
| `oxygen_tolerance` | f1 | 0.2398 | Koblitz 2025 (Commun Biol) | — | [@koblitz2025bacdiveml] |
| `pathogenicity_human` | acc | 0.9212 | PathogenFinder 2013 | 0.8800 | [@cosentino2013pathogenfinder] |
| `sporulation` | f1 | 0.8019 | Koblitz 2025 | — | [@koblitz2025bacdiveml] |
| `sporulation` | f1 | 0.8019 | Traitar 2016 | 0.9300 | [@weimann2016traitar] |

## B. Context (different metric / evaluation — not apples-to-apples)

| Trait | Prior method | Metric | Score | Cite | Notes |
|---|---|---|---|---|---|
| `amr_phenotype` | CARD/RGI 2023 | other | — | [@alcock2023card] | Reference baseline. Gene-presence != MIC; we predict per-class binary phenotype. |
| `amr_phenotype` | AMRFinderPlus 2021 | f1 | — | [@feldgarden2021amrfinderplus] |  |
| `amr_phenotype` | DeepARG 2018 | f1 | 0.8000 | [@arangoargoty2018deeparg] | Reads-based, not genome-pheno; cite as ML peer. |
| `carbon_utilization` | GapMind 2022 | other | — | [@price2022gapmindcarbon] | Mechanistic, not ML; reports per-substrate yes/no confidence with high precision. |
| `cultivation_medium` | KOMODO 2015 | other | — | [@oberhardt2015komodo] | Only published predictor in this space; not multi-label F1. |
| `fatty_acid_profile` | (none — literature white-space) | rmse | — | — | Live-search confirmed: zero papers predict FAME composition from genome. microbe-foundation establishes the baseline. |
| `ph_class` | Ramoneda 2023 | auroc | — | [@ramoneda2023ph] | Reports AUROC ~0.85-0.9 per their abstract; binned-class accuracy not directly reported. |
| `sporulation` | Galperin et al. 2012 | other | — | [@galperin2012sporulation] | Rule-based; reports recall on known sporeformers ~95%. |
| `temperature_class` | Engqvist 2018 | rmse | 5.400 | [@engqvist2018temperature] | RMSE on continuous OGT; we use 5-class accuracy. Convert via class bin width. |
| `temperature_class` | Tome (Li & Engqvist 2019) | rmse | 6.000 | [@li2019tome] | Protein-level model; organism-level numbers approximate. |
| `temperature_class` | Liu 2025 BMC Genomics | rmse | — | [@liu2025ogt] | Score TBD — paper accessible but exact RMSE not extracted by audit. |

## C. Literature white-space (no prior comparator)

| Trait | Status |
|---|---|
| `biosafety_level` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `catalase` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `cell_shape` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `country` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `cytochrome_oxidase` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `fatty_acid_profile` | Live-search confirmed: zero papers predict FAME composition from genome. microbe-foundation establishes the baseline. |
| `halophily` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `isolation_source` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `metabolite_production` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `pathogenicity_animal` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |
| `pigmentation` | No prior entry in `prior_numbers.json` (likely white-space — add explicit entry) |

_1 of our runs compared against 11 prior-trait entries._
