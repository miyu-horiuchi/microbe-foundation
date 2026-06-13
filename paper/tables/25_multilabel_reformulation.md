# Table 25 — Multi-label reformulation of the structured ceiling traits

Scored as a single collapsed label these traits look unsolvable (Table 20); posed correctly as multi-label they are not. Per trait: balanced one-vs-rest linear probes on frozen ESM-2, species split, over the well-supported labels. The single-label collapse baseline is the species-split macro-F1 from Table 20.

| Trait | Formulation | #labels | Macro-AUROC | Macro-F1 | Micro-F1 | Collapse F1 (Table 20) |
|---|:--|---:|---:|---:|---:|---:|
| `metabolite_production` | masked binary / compound | 6 | 0.603 | 0.405 | 0.359 | 0.102 |
| `cultivation_medium` | presence / medium | 54 | 0.891 | 0.289 | 0.424 | 0.278 |

## Best-predicted labels — `metabolite_production`

| Label | AUROC | F1 | n pos (train/test) |
|---|---:|---:|---:|
| `nitrite` | 0.788 | 0.545 | 35/5 |
| `indole` | 0.672 | 0.237 | 415/179 |
| `dinitrogen` | 0.621 | 0.000 | 8/5 |
| `hydrogen sulfide` | 0.559 | 0.275 | 232/126 |
| `acetoin` | 0.550 | 0.518 | 584/239 |
| `lactate` | 0.429 | 0.857 | 10/7 |

## Best-predicted labels — `cultivation_medium`

| Label | AUROC | F1 | n pos (train/test) |
|---|---:|---:|---:|
| `1076` | 1.000 | 0.878 | 26/18 |
| `585` | 1.000 | 0.829 | 40/19 |
| `9` | 0.998 | 0.300 | 30/5 |
| `645` | 0.996 | 0.779 | 85/61 |
| `28` | 0.992 | 0.222 | 6/5 |
| `1560` | 0.990 | 0.578 | 17/18 |
| `11` | 0.988 | 0.768 | 166/67 |
| `372` | 0.982 | 0.473 | 29/21 |
