# Table 26 — cultivation_medium: turning rank quality into usable predictions

Per-medium F1 on the species test split as we (1) tune the decision threshold on validation and (2) apply an Elkan--Noto positive-unlabeled correction. AUROC is unchanged by thresholding and shown for reference; `c` is the estimated labeling propensity P(listed | grows), so low `c` means many true-but-unlisted media.

Single-label collapse macro-F1 (Table 20): 0.278. Naive multilabel @0.5 macro-F1 (Table 25): 0.289.

| Metric (macro over media) | Value |
|---|---:|
| #media | 23 |
| AUROC | 0.887 |
| F1 @0.5 (naive) | 0.382 |
| F1 + threshold tuning | 0.378 |
| F1 + tuning + PU correction | 0.640 |
| mean labeling propensity c | 0.533 |

## Best-recovered media (by PU-adjusted F1)

| Medium | AUROC | F1@0.5 | F1 tuned | F1 PU | c | n pos (test) |
|---|---:|---:|---:|---:|---:|---:|
| `65` | 0.975 | 0.796 | 0.800 | 0.899 | 0.838 | 478 |
| `553` | 0.932 | 0.314 | 0.289 | 0.869 | 0.389 | 84 |
| `78` | 0.946 | 0.170 | 0.150 | 0.868 | 0.437 | 34 |
| `84` | 0.967 | 0.575 | 0.571 | 0.868 | 0.737 | 166 |
| `110` | 0.942 | 0.238 | 0.248 | 0.858 | 0.476 | 56 |
| `252` | 0.937 | 0.369 | 0.355 | 0.834 | 0.488 | 88 |
| `11` | 0.988 | 0.763 | 0.760 | 0.829 | 0.870 | 67 |
| `104` | 0.928 | 0.453 | 0.456 | 0.804 | 0.483 | 141 |
| `645` | 0.996 | 0.768 | 0.789 | 0.765 | 0.897 | 61 |
| `514` | 0.927 | 0.648 | 0.666 | 0.748 | 0.759 | 457 |
