---
title: predictability-gradient
emoji: 🧬
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Interactive companion for the predictability-gradient paper
---

# predictability-gradient

Research showcase for the paper:

> When Does Attention Help? A Predictability Gradient for Genomic Trait Prediction,
> Validated by Virulence-Factor Attribution

This Space is a paper companion, not a production predictor. It presents the main
scientific result as an interactive exhibit:

- attention-pooling helps machinery traits more than compositional traits;
- the advantage collapses at family-level holdout;
- pathogenicity attention concentrates on a few proteins;
- top-attended proteins are enriched for VFDB virulence factors;
- top-protein ablation flips pathogenic predictions.

## Refreshing assets

From the repository root:

```bash
python3 spaces/research_showcase/scripts/build_assets.py
```

The script reads `runs/*.json` and writes:

```text
spaces/research_showcase/assets/predictability_gradient.json
```

That JSON is intentionally small so the Space can run on CPU-basic without access to
the full 105 GB per-protein embedding corpus.
