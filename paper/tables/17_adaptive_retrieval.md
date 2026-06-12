# Table 17 — Novelty-adaptive retrieval head (cross-clade)

Per-genome blend weight `alpha_i = clip(alpha_lo + (alpha_hi - alpha_lo) * t_i, 0, 1)`, where `t_i` is family-val-normalized novelty (Euclidean OOD vs family-train). Ramp endpoints tuned on family-val, scored once on family-test. `alpha_hi > alpha_lo` means the probe is trusted more as novelty rises. Compared against the global-alpha blend (Table 16) and probe-alone on the same test set.

| Trait | Test n | alpha_lo | alpha_hi | alpha_global | Probe F1 | Global F1 | Adaptive F1 | delta vs global | delta vs probe | Probe AUROC | Global AUROC | Adaptive AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sporulation` | 809 | 0.4 | 0.4 | 0.4 | 0.827 | 0.844 | 0.844 | +0.000 | +0.018 | 0.925 | 0.935 | 0.935 |
| `motility` | 1673 | 0.4 | 0.6 | 0.4 | 0.636 | 0.656 | 0.660 | +0.004 | +0.024 | 0.696 | 0.721 | 0.723 |
| `catalase` | 1442 | 0.0 | 0.4 | 0.4 | 0.686 | 0.726 | 0.709 | -0.016 | +0.023 | 0.793 | 0.797 | 0.790 |

**Verdict:** novelty-conditioning does not reliably beat the global blend (mean delta-F1 vs global = -0.004; 0/3 traits improve by >0.01, 1 regress). Tuning the ramp on family-val can overfit and transfer negatively (catalase), so the simpler global alpha (Table 16) is the robust choice.
