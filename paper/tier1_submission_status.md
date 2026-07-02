---
title: "Microbe Foundation --- Research Status and Tier-1 Submission Roadmap"
author:
  - Miyu Horiuchi
date: 2026-06-12
geometry: margin=1in
fontsize: 10pt
linkcolor: blue
header-includes:
  - \usepackage{microtype}
---

# Executive summary

This project asks **when protein-set pooling matters for bacterial trait prediction**, validates the mechanism on pathogenicity, and diagnoses **why performance collapses on novel taxonomic families**. The core scientific arc is largely complete in code and manuscript form. What remains for **ICML / NeurIPS / AAAI main track** is to demonstrate that cross-clade improvements hold on the **trained full model** (not only linear probes), add **one strong novelty pillar** (Set Transformer or family-balanced training), and close **reviewer-facing gaps** (pathogenicity confounds, measured localization, figures, more seeds).

**Repository:** microbe-foundation (main branch, through commit f62607b).

---

# 1. Problem and hypothesis

## Setting

- **Data:** 19,592 bacterial genomes, 21 BacDive trait labels, family/genus/species holdout splits.
- **Encoder:** Frozen ESM-2 (esm2\_t30\_150M, 640-d per protein); ~82M protein embeddings.
- **Variable:** Pooling operator only (mean vs attention vs others in model.py).

## Predictability gradient hypothesis

> Attention-pooling helps **machinery** traits (localized gene signal) more than **compositional** traits (diffuse genome-wide signal).

## Binding constraint (from paper)

> Cross-clade (family-level) generalization dominates; pooling architecture advantage **vanishes** under family holdout.

---

# 2. Methods (what was built)

## 2.1 Core model pipeline

- **model.py** --- Multi-task MLP on pooled genome vectors; per-protein path with attention/mean/max/top-k/gated pooling.
- **compute\_esm2\_\*.py, embed\_from\_cache.py** --- GPU embedding pipeline (multi-GPU, resumable).
- **Splits** --- data/splits.parquet; three-way family split: train / val / test (disjoint families).

## 2.2 Analysis and monitoring scripts (shipped)

- **microbe\_model/monitoring** --- Euclidean OOD score vs reference manifold.
- **ood\_error\_analysis.py** --- Table 14 (OOD vs per-genome error).
- **cross\_clade\_diagnostic.py** --- Table 15 + diversity-curve figure.
- **retrieval\_head.py** --- Table 16 (global probe + k-NN blend).
- **adaptive\_retrieval.py** --- Table 17 (novelty-ramped blend; negative result).

**Leakage control:** Probe, k-NN reference, and novelty manifold fit on family-train; blend weights tuned on family-val; family-test scored once.

## 2.3 Manuscript

Four variants updated with cross-clade section: main, academic, ip\_safe, publishable (PDFs via pandoc + tectonic).

## 2.4 Tests

79 passed on analysis/monitoring suite. Torch tests blocked by local anaconda numpy hang (environment issue, not a code regression).

---

# 3. Results so far

## 3.1 Main paper --- predictability gradient

| Split | Compositional $\Delta$F1 | Machinery $\Delta$F1 | Gap |
|:--|:--:|:--:|:--:|
| Species | +0.021 $\pm$ 0.002 | +0.083 $\pm$ 0.012 | +0.062 |
| Genus | +0.016 $\pm$ 0.004 | +0.067 $\pm$ 0.010 | +0.052 |
| Family | +0.009 $\pm$ 0.002 | +0.010 $\pm$ 0.003 | +0.001 |

**Finding:** Attention helps machinery traits ~4x more within distribution; advantage collapses at family level.

## 3.2 Mechanism --- pathogenicity (VFDB)

- Top-5 proteins carry ~81% of attention mass.
- ~5x VFDB enrichment vs random proteins in same genome (Wilcoxon $p = 2.5 \times 10^{-7}$).
- Masking top-5 flips ~23% of pathogenic predictions.
- **Caveat:** Taxonomy-majority baseline is strong; matched-clade controls needed for tier-1.

## 3.3 OOD monitor (Table 14)

- Euclidean k-NN distance: AUROC 0.76 for novel-family detection.
- Diffusion-map variant does not beat Euclidean.
- OOD score is a distribution-shift detector, not a per-genome error oracle.

## 3.4 Cross-clade diagnostic (Table 15)

**Question:** Why does family-level transfer collapse?

- **k-NN label transfer:** Matches trained linear probe; AUROC 0.69--0.92; well above chance. Conclusion: **not a representation wall**.
- **Size-controlled diversity curve:** Transfer rises with more training families at fixed genome count. Conclusion: **clade-coverage-limited**.

Traits tested: sporulation, motility, catalase.

## 3.5 Retrieval-augmented head (Table 16)

**Method:** $p = \alpha\, p_{\text{probe}} + (1-\alpha)\, p_{\text{knn}}$, $\alpha$ tuned on family-val.

| Trait | $\alpha^*$ | Probe F1 | Blend F1 | $\Delta$F1 | Probe AUROC | Blend AUROC |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Sporulation | 0.4 | 0.827 | 0.844 | +0.018 | 0.925 | 0.935 |
| Motility | 0.4 | 0.636 | 0.656 | +0.020 | 0.696 | 0.721 |
| Catalase | 0.4 | 0.686 | 0.726 | +0.039 | 0.793 | 0.797 |

**Finding:** Global blend improves cross-clade transfer without encoder fine-tuning.

**Scope:** Linear probe on mean-pooled 640-d features --- **not yet the full model.py system**.

## 3.6 Novelty-adaptive blend (Table 17) --- negative

| Trait | $\alpha_{\text{lo}} \to \alpha_{\text{hi}}$ | $\Delta$ vs global |
|:--|:--:|:--:|
| Sporulation | 0.4 $\to$ 0.4 | +0.000 |
| Motility | 0.4 $\to$ 0.6 | +0.004 |
| Catalase | 0.0 $\to$ 0.4 | -0.016 |

**Finding:** Novelty-conditioning does not beat constant $\alpha = 0.4$ (mean $\Delta$F1 vs global = -0.004).

---

# 4. One-paragraph story (current)

Pooling should be chosen by trait localization: attention helps machinery traits in-distribution and is mechanistically grounded for pathogenicity (VFDB). The pooling advantage disappears at family holdout, where the bottleneck is training-clade coverage, not a frozen-embedding wall. Cross-clade signal exists in embedding geometry; a probe + k-NN blend recovers a modest but consistent slice of family-level performance without fine-tuning. Per-genome novelty weighting does not help.

---

# 5. What tier-1 reviewers will ask

| Question | Status today |
|:--|:--|
| What is new beyond comparing known poolers? | Partial --- Set Transformer has no results yet |
| Does the fix work on the real model? | No --- Tables 16--17 use linear probes |
| Is pathogenicity real or taxonomy? | Incomplete --- matched controls uncommitted |
| Is compositional/machinery split measured? | Weak --- eggNOG audit $p = 0.083$ |
| Enough seeds, traits, figures? | Thin --- 3 traits; no cross-clade figures in manuscript |
| Single encoder scale? | Yes --- only 150M ESM-2 |

---

# 6. What needs to be done (tier-1 main track)

## P0 --- Submission blockers

**6.1 New method or training recipe (at least one; ideally both)**

- **Set Transformer pooler (paper section 6):** Implement ISAB + PMA; train 3--5 seeds; all splits including family.
- **Family-balanced training:** Cap genomes per family on family-train; evaluate on family-test.

**6.2 Cross-clade on the real system**

- Retrieval blend on model.py checkpoint outputs (not LogisticRegression).
- All binary traits; 5+ seeds; compare model vs k-NN vs blend.

**6.3 Pathogenicity confound controls**

- Matched within-family pathogenic / non-pathogenic controls.
- VFDB enrichment and ablation under controls.

**6.4 Measured localization**

- Sparse gene-family predictors; correlation with attention gain ($p < 0.05$).

**6.5 Figures and statistics**

- Cross-clade figure panel; Set Transformer comparison; 5+ seeds with CIs.

## P1 --- Rebuttal support

- ESM-2 650M re-embed + re-run diagnostics.
- Reproducibility package; fix environment; freeze one manuscript.

## P2 --- Defer

- Adaptive blend variants (done).
- LoRA fine-tuning before P0.
- Product deploy (not tier-1 main lever).

---

# 7. Go / no-go criteria

Submit ICML / NeurIPS / AAAI main when most are true:

- Set Transformer or family-balanced training shows family-level lift on real model (3+ traits, seeded).
- Retrieval blend beats model-alone on family-test (5+ traits, error bars).
- Pathogenicity survives matched-clade controls.
- Localization measured, not hand-labeled only.
- Cross-clade and method figures included.
- 650M encoder ablation done or bounded.

**Fallback if P0 fails:** NeurIPS Datasets and Benchmarks, ML4Science workshop, or ISMB / RECOMB.

---

# 8. Suggested execution order

1. Set Transformer implement + train + eval (weeks 1--2)
2. Family-balanced training + retrieval on real model (weeks 2--3)
3. Pathogenicity controls + localization upgrade (week 3)
4. Figures, seeds, 650M, manuscript freeze (week 4)

---

# 9. Venue map

| Venue | Fit now | After P0 |
|:--|:--|:--|
| NeurIPS / ICML main | Weak--borderline | Competitive if method or training lands |
| AAAI main | Borderline | Competitive with confound controls |
| NeurIPS D&B | Moderate | Strong as benchmark paper |
| ISMB / RECOMB | Strong | Ready with polish |

---

# 10. Artifacts index

- Main manuscript: paper/predictability\_gradient.md / .pdf
- Table 14: paper/tables/14\_ood\_error\_gradient
- Table 15: paper/tables/15\_cross\_clade\_diagnostic
- Table 16: paper/tables/16\_retrieval\_head
- Table 17: paper/tables/17\_adaptive\_retrieval
- Figure: paper/figures/cross\_clade\_diversity\_curve.png
- Code: cross\_clade\_diagnostic.py, retrieval\_head.py, adaptive\_retrieval.py

---

*Generated 2026-06-12. Summarizes work through f62607b.*
