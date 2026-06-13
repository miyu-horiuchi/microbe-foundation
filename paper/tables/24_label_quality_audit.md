# Table 24 — Label-quality audit of the ceiling traits

## A. Why each label-ceiling trait is hard

| Trait | n labeled | n classes | Top-1 share | Singleton frac | Norm. entropy | Dominant pathology |
|---|---:|---:|---:|---:|---:|:--|
| `country` | 53065 | 246 | 0.196 | 0.142 | 0.618 | many classes, no genomic basis (e.g. geography) |
| `isolation_source` | 57783 | 24004 | 0.055 | 0.795 | 0.831 | free-text explosion (sparse classes) |
| `cell_shape` | 15181 | 7 | 0.798 | 0.000 | 0.363 | few classes, weak genomic signal |
| `oxygen_tolerance` | 23253 | 6 | 0.667 | 0.000 | 0.512 | few classes, weak genomic signal |
| `metabolite_production` | 23631 | structured | — | — | — | structured/multilabel (collapsed to one label) |
| `cultivation_medium` | 28637 | structured | — | — | — | structured/multilabel (collapsed to one label) |

## B. Fixing the fixable: consolidate the label schema, re-probe in-distribution

Same balanced multiclass linear probe on frozen ESM-2, species-split macro-F1, before vs after schema consolidation. A rise means the ceiling was partly a label artifact, not a representational wall.

| Trait | Before: classes → F1 | After: classes → F1 | ΔF1 |
|---|---:|---:|---:|
| `oxygen_tolerance` | 5 → 0.343 | 3 → 0.420 | +0.078 |
| `isolation_source` | 13 → 0.270 | 8 → 0.273 | +0.003 |
