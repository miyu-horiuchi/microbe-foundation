---
title: "When Does Attention Help? A Predictability Gradient for Genomic Trait Prediction, Validated by Virulence-Factor Attribution"
author:
  - |
    Miyu Horiuchi  
    m@replicater.xyz
date: 2026-06-05
abstract: |
  A bacterial genome is a *set* of proteins, and predicting an organism's phenotype from that set requires pooling thousands of protein representations into a fixed-size genome vector. The default choice, mean-pooling, treats every protein equally; learned attention-pooling can instead concentrate on a few. We ask a question usually answered by default rather than by evidence: **when does the pooling choice matter, and why?** We propose the *predictability gradient* hypothesis: pooling architecture matters in proportion to how *localized* a trait's genomic signal is. Compositional traits (e.g. oxygen tolerance, cell shape) reflect diffuse genome-wide signal and should be insensitive to pooling; machinery traits (e.g. pathogenicity, nutrient requirements) are determined by a handful of decisive genes and should benefit from attention. We test this on 19,592 bacterial genomes across 21 traits and three taxonomic generalization regimes. With three seeds per configuration, attention-pooling improves machinery traits ~4x more than compositional traits (species-level gap in F1 gain $+0.062 \pm 0.011$ vs $+0.001$ at family level), confirming the gradient and revealing that the advantage *collapses entirely under family-level covariate shift*. We then show the mechanism is genuine: for pathogenicity, the model concentrates 81% of its attention on five proteins, and those proteins are 5x enriched for known virulence factors versus random proteins in the same genome (Wilcoxon $p=2.5\times10^{-7}$) and 3.2x versus non-pathogenic genomes ($p=6.8\times10^{-14}$); ablating them flips 23% of pathogenic predictions. The attention does not merely correlate with virulence; it locates adhesins, fimbrial ushers, and invasion loci, and the prediction depends on them. Our results give a predictive, mechanistically-grounded rule for when un-pooled protein representations are worth their cost, and identify cross-clade generalization, not pooling, as the binding constraint for genomic foundation models.
geometry: margin=1in
fontsize: 10pt
linkcolor: blue
header-includes:
  - \usepackage{microtype}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{Horiuchi, Predictability Gradient}
  - \fancyhead[R]{IP-conscious manuscript draft}
  - \fancyfoot[C]{\thepage}
  - \fancyfoot[L]{Copyright \textcopyright{} 2026 Miyu Horiuchi. All rights reserved.}
---

> **IP notice.** Copyright © 2026 Miyu Horiuchi. All rights reserved. This manuscript is made available for scholarly review and publication consideration only. No license is granted to reproduce, commercialize, deploy, or create derivative software systems from any implementation, dataset, checkpoint, production workflow, or future product design referenced here. Patentable subject matter, if any, is expressly reserved. This notice does not restrict fair citation of the scientific findings.

# Introduction

Foundation models for bacterial genomes have advanced rapidly: protein language models such as ESM-2 [@lin2023esm2] produce rich per-protein representations, and recent genome-level models [@wiatrak2025bacformer; @microgenomer2025; @bacpt2026] aggregate them to predict phenotype. A genome, however, is not a sequence but an unordered *set* of $P$ proteins ($P \approx 10^3$--$10^4$), so every such model must **pool** a variable-size set of protein vectors into one genome representation before prediction. This pooling step is almost always chosen by convention, typically a mean, and almost never studied.

This is a missed opportunity, because pooling encodes a strong inductive bias about *where a trait's signal lives in the genome*. Mean-pooling assumes the signal is diffuse: every protein contributes equally. Attention-pooling [@ilse2018attention] assumes the signal is localized: the model learns to up-weight a few decisive proteins. Which bias is correct is not a universal fact; it depends on the trait. Whether a bacterium is Gram-positive is a property of its entire cell envelope; whether it is pathogenic can hinge on a single secretion system.

We formalize this as the **predictability gradient** hypothesis:

> The benefit of attention-pooling over mean-pooling for a trait is proportional to how *localized* that trait's genomic determinants are. Diffuse "compositional" traits gain little; gene-specific "machinery" traits gain much.

This hypothesis is attractive because it is *falsifiable inside a single architecture*: hold the encoder fixed, swap only the pooling, and measure the gain as a function of trait class. We do exactly this, and add two things a benchmark number cannot provide. First, we vary the **generalization regime** (species-, genus-, family-held-out splits), turning the experiment into a test of whether the gradient survives distribution shift. Second, for the trait with the largest gain, pathogenicity, we ask whether the attention is *mechanistically real*: does it concentrate on the actual virulence machinery, and does the prediction causally depend on it? This converts a performance claim ("attention helps") into a scientific one ("attention helps because it finds the responsible genes").

**Contributions.**

1. **The predictability-gradient hypothesis and its validation.** A testable account of when set-pooling architecture matters for genomic prediction, confirmed on 19,592 genomes / 21 traits with three seeds: attention helps machinery traits ~4x more than compositional traits, with non-overlapping error bars (§4).
2. **A covariate-shift result.** The gradient is strong within taxonomic distribution and vanishes at the family level (gain gap $+0.062 \to +0.001$). The binding constraint for genomic foundation models is cross-clade generalization, not pooling (§4.2).
3. **Mechanistic attribution against a virulence-factor database.** For pathogenicity, attention concentrates (81% mass on 5 of ~3,800 proteins) and is enriched for VFDB virulence factors under two controls; ablation shows the prediction depends on them; the hits are coherent adherence/invasion machinery (§5).
4. **An evaluation protocol for trait-localization hypotheses.** The paper defines a compact, reusable experimental pattern for testing whether architectural changes help because the relevant phenotype is diffuse, localized, or shifted out of distribution. Production systems, future recipe-generation workflows, and nonessential implementation assets are outside the scope of this public manuscript.

We are explicit about what this is *not*: not a new encoder (we freeze ESM-2), not a state-of-the-art benchmark sweep, not a release of a production cultivation-recommendation system, and not a solution to cross-clade generalization, which our own results show is the open problem.

# Related Work

**Genomic and protein foundation models.** ESM-2 [@lin2023esm2] provides the frozen per-protein embeddings we pool. Genome-level models, Bacformer [@wiatrak2025bacformer], MicroGenomer [@microgenomer2025], and BacPT [@bacpt2026], aggregate protein or gene representations for phenotype prediction but report benchmark accuracy without analyzing the pooling step or attributing predictions to specific genes. Our contribution is orthogonal and complementary: we hold the encoder fixed and study the aggregation choice and its mechanism.

**Microbial trait prediction.** Classical predictors (Traitar [@weimann2016traitar], Genome Properties [@richardson2019genomeprops], PathogenFinder [@cosentino2013pathogenfinder]) use hand-engineered gene-content features and per-trait classifiers; curated random-forest baselines on BacDive traits remain strong [@koblitz2025traits]. These methods implicitly assume localized, gene-presence signal, consistent with our "machinery" pole, but do not contrast it against a diffuse-signal baseline or quantify when the assumption pays off.

**Attention as explanation.** A literature cautions that attention weights are not, on their own, faithful explanations [@jain2019attention; @wiegreffe2019attention]. We take this seriously: our mechanistic claim does not rest on attention magnitudes but on (i) *external* validation against a curated virulence-factor database [@liu2022vfdb] under matched controls and (ii) *causal* ablation. We flag where ablation is partly mechanical (§5.3, §6).

**Set learning.** Attention-pooling over instances is standard in multiple-instance learning [@ilse2018attention], and permutation-invariant set attention is formalized by the Set Transformer [@lee2019settransformer]. Our novelty is not the existence of attention over sets but tying its benefit to a measurable property of the label (trait localization) and validating the learned attention against ground-truth biology.

# Setup

## Data

Labels are drawn from BacDive [@schober2025bacdive], yielding **21 prediction heads** across seven biological blocks (morphology, physiology, growth conditions, cultivation, safety, ecology, chemotaxonomy). For each strain with an NCBI genome we predict open reading frames with pyrodigal [@larralde2022pyrodigal] and embed each protein independently with a frozen ESM-2 encoder, producing a ragged protein-representation set per genome. A mean-pooled baseline feature is computed from the same protein representations, isolating pooling as the only experimental variable. The training corpus contains 19,592 genomes and approximately 82M protein representations.

This public manuscript describes the scientific evaluation. Complete deployment artifacts, production inference workflows, model checkpoints, and large feature stores are intentionally not included in this version pending separate IP and release review.

## Trait taxonomy: compositional vs machinery

We pre-register a partition of the 21 heads into **compositional** (signal expected diffuse) and **machinery** (signal expected gene-localized), from biological first principles and *before* seeing pooling results:

- **Compositional (11):** gram stain, cell shape, motility, sporulation, oxygen tolerance, catalase, cytochrome oxidase, temperature class, pH class, halophily, pigmentation.
- **Machinery (8):** pathogenicity (human), pathogenicity (animal), cultivation medium, carbon utilization, metabolite production, antimicrobial-resistance phenotype, biosafety level, fatty-acid (FAME) profile.

Two metadata heads (isolation source, country) are excluded from the gradient analysis as they are not biological traits.

## Model

A shared MLP encoder feeds 21 linear heads under a **masked multi-task loss** that contributes zero gradient for missing labels (BacDive coverage ranges 5--95% per head). The only architectural variable in this experiment is the pooling:

- **Mean-pool:** genome vector $\bar{x} = \frac{1}{P}\sum_i x_i$.
- **Attention-pool:** a learned scalar scorer produces a masked softmax over real proteins and a weighted-sum genome vector.

Both share encoder, heads, and loss; only pooling differs. This deliberately narrow comparison is the reason the result can be interpreted as a pooling effect rather than an encoder effect.

## Evaluation regimes

We use **species-, genus-, and family-held-out** splits: no species (resp. genus, family) appears in more than one fold. Family-held-out approximates prediction for clades unlike anything seen in training, the regime relevant to uncultured "microbial dark matter." We report macro-F1 per head and, for the gradient, $\Delta\mathrm{F1} = \mathrm{F1}_{\text{attn}} - \mathrm{F1}_{\text{mean}}$, aggregated within trait class, mean $\pm$ std over **3 seeds**.

# The Predictability Gradient

## Attention helps machinery traits, not compositional traits

: Gain from attention-pooling over mean-pooling ($\Delta$F1, mean $\pm$ std over 3 seeds).

| Split | Compositional (n=11) | Machinery (n=8) | Gap (mach $-$ comp) |
|:--|:--:|:--:|:--:|
| species | $+0.021 \pm 0.002$ | $\mathbf{+0.083 \pm 0.012}$ | $\mathbf{+0.062}$ |
| genus | $+0.016 \pm 0.004$ | $\mathbf{+0.067 \pm 0.010}$ | $\mathbf{+0.052}$ |
| family | $+0.009 \pm 0.002$ | $+0.010 \pm 0.003$ | $+0.001$ |

At the species and genus levels the machinery gain exceeds the compositional gain by ~4x, with **non-overlapping error bars** ($0.083\pm0.012$ vs $0.021\pm0.002$). The compositional gain is small, positive, and tight; attention does not *hurt* diffuse traits, it simply adds little, exactly as the hypothesis predicts: when signal is genome-wide, a weighted average and a flat average converge. The single largest per-head effects are pathogenicity (animal F1 $0.26\to0.50$, human $0.16\to0.32$ at species), the most gene-localized traits in the set.

## The gradient collapses under covariate shift

The most consequential result is the family row. As the test distribution moves from same-species to same-family to *novel-family* organisms, the machinery advantage decays monotonically ($+0.083 \to +0.067 \to +0.010$) until the gradient is effectively gone (gap $+0.001$). Mean- and attention-pooling become indistinguishable precisely in the regime that matters for uncultured organisms. This localizes the bottleneck: the limiting factor is not how we pool proteins but whether *any* protein-set representation transfers across evolutionary distance.

## Detecting the covariate-shift regime

If the pooling advantage vanishes precisely for novel-family organisms, a practitioner needs to *know* when a query genome falls in that regime. This is feasible without labels: a monitor over the genome embeddings flags novel-family organisms. Using mean $k$-nearest-neighbour distance from a candidate's 640-d ESM-2 vector to the training-family genomes, held-out *novel families* separate from held-out strains of *seen families* at AUROC 0.76. A curvature-aware diffusion-map variant (a heat-kernel embedding of the reference manifold) does **not** improve on plain Euclidean distance (AUROC 0.69--0.73 across 10--100 diffusion components); the simplest detector is the best one.

Detecting the regime is not the same as triaging individual genomes. We tested whether the same novelty score predicts *per-genome* error, training family-split classifiers per trait and correlating embedding distance with $|y-\hat p|$ on novel families. The relationship is trait-specific and weak: positive only for the most localized, highly-learnable machinery trait (sporulation, Spearman $+0.09$ with the production model; $+0.14$ with a linear probe), null for compositional traits, and *negative* for imbalanced pathogenicity, where genomes far from known pathogens are confidently---and correctly---predicted non-pathogenic. The embedding monitor is therefore a **distribution-shift detector, not a per-genome confidence estimate**: it signals that the model is operating off-distribution, not which individual predictions to distrust.

# Does the Attention Find the Right Genes?

A performance gain does not establish that attention is mechanistically meaningful; attention weights need not be faithful [@jain2019attention]. We validate the largest-gain trait, **pathogenicity**, against external ground truth (VFDB [@liu2022vfdb], 4,663 experimentally-verified virulence factors) with two matched controls and a causal ablation. We train single-task attention-pool models per pathogenicity head (so the shared pool specializes); both discriminate well on held-out test genomes (AUROC 0.88 animal, 0.85 human).

## Attention concentrates

On held-out genomes the attention distribution is sharp: median normalized entropy 0.26 (animal), with the **top-5 of ~3,800 proteins carrying 81% of the attention mass** (top-1 alone $\approx 40\%$). Concentration is a precondition for an interpretable spotlight and is label-independent (the model always selects a few proteins); the question is *which*.

## Top-attended proteins are enriched for virulence factors

We map each protein's attention rank to its amino-acid sequence and call a protein a virulence factor if it matches VFDB by diamond blastp [@buchfink2021diamond] at $E<10^{-5}$. Two controls:

: VFDB enrichment of top-5 attended proteins (animal head; species split).

| Test | Top-attended | Control | Statistic |
|:--|:--:|:--:|:--|
| Within-genome (paired, vs random in same genome) | 28.1% | 5.9% | Wilcoxon $p=2.5\times10^{-7}$, $n{=}74$ |
| Between-class (vs top-attended in non-path. genomes) | 28.1% | 10.9% | Fisher OR$=3.2$, $p=6.8\times10^{-14}$ |

The proteins attention selects are ~5x more likely to be known virulence factors than random proteins in the same genome, and the enrichment is 3.2x stronger in pathogenic than non-pathogenic genomes. The human head replicates (between-class OR 3.1, $p=3.8\times10^{-5}$; within-genome $p=0.034$ at $n{=}16$). The hits are not database noise: the most frequently selected VFs are coherent **adherence/invasion machinery**, including fimbrial ushers `papC`/`mrkC`, filamentous hemagglutinin `fhaB`, attachment-invasion locus `ail`, type-IV pilus `pilQ`, and flagellar `fliR`.

## The prediction causally depends on them

Masking the top-5 attended proteins (of ~3,800) and re-predicting flips **22.7% of pathogenic calls** to non-pathogenic (animal; Wilcoxon vs random-protein masking $p=1.2\times10^{-4}$); the human head replicates (12.5%, $p=9\times10^{-3}$). Masking proteins ranked 6--10 produces a 27.5x smaller drop; the dependence is specific to the very top proteins. We note honestly that a top-vs-random ablation gap is *partly mechanical*: attention-pooling is a weighted sum, so removing high-weight elements changes the output more by construction. The non-trivial facts are therefore the **absolute** effect (removing 5 of ~3,800 proteins overturns a quarter of predictions) and its conjunction with §5.2 (those 5 proteins are the virulence machinery). Together they support a causal-mechanistic reading that neither establishes alone.

# Discussion and Limitations

**What the results say.** Set-pooling architecture for genomic prediction should be chosen by trait localization, not convention: attention is worth its cost for gene-determined traits and not for diffuse ones, and for pathogenicity it works by locating the responsible genes. This gives practitioners a predictive rule and gives the field a mechanistic check (external attribution + ablation) that any genome model can adopt.

**What this public manuscript does not disclose.** This version intentionally omits nonessential deployment details, checkpoint distribution, large feature-store locations, production inference workflows, and future cultivation-recipe system designs. It also avoids specifying any unreleased next-generation architecture beyond the high-level observation that future work should model protein interactions rather than only independent protein weights.

**Limitations, stated plainly.**

- *Covariate shift is unsolved and dominates.* The benefit evaporates at family-level holdout (§4.2). For the uncultured-organism application that motivates genomic trait models, this is the result that matters most, and it is negative for the pooling lever. It is at least *detectable*: a label-free embedding monitor flags novel-family genomes at AUROC 0.76, though that signal does not transfer to per-genome confidence.
- *Weighted sum, not combinations.* The validated model up-weights individual proteins; it does not model protein interactions.
- *Enrichment, not coverage.* 68% of top-attended proteins are not VFDB hits; VFDB catalogs only *known* factors, so this is expected and our claim is strictly about enrichment.
- *Single encoder scale.* All results use one ESM-2 size; whether a larger encoder sharpens the gradient is untested.
- *Statistical scope.* The gradient uses 3 seeds; the human-head within-genome control is underpowered ($n{=}16$). The animal head and between-class tests are well-powered ($p<10^{-6}$); the small compositional effects ($\le0.02$ $\Delta$F1) are within their own noise and we do not over-interpret them.

**Why this is a useful result.** The positive half (a validated, mechanistically-explained predictability gradient) is a clean architectural insight; the negative half (it does not survive clade shift) redirects effort from the pooling question, which we consider resolved, to the generalization question, which we do not.

# Intellectual Property and Availability

This manuscript is intended for scholarly publication while preserving optionality around implementation-specific intellectual property. The text, tables, and figures are copyright © 2026 Miyu Horiuchi, all rights reserved unless a later publication agreement states otherwise. No patent rights, software license, model license, dataset license, or commercial deployment rights are granted by this manuscript.

For academic review, the trait-class definitions, split definitions, evaluation protocol, and summary statistics are described here. Supporting materials sufficient for independent audit can be made available to editors, reviewers, or collaborators under an appropriate confidentiality or material-transfer arrangement where needed. Public release of full code, trained checkpoints, large feature stores, production inference assets, and cultivation-recipe tooling should occur only after separate IP review.

# Conclusion

Pooling a genome's proteins is an inductive-bias choice, and the right choice is a measurable function of the trait: attention for localized "machinery" traits, mean for diffuse "compositional" traits. We validated this gradient with seeds, showed for pathogenicity that the attention concentrates on bona fide virulence factors and the prediction depends on them, and found the entire advantage contingent on staying within taxonomic distribution. Reading the right genes is achievable; reading them for organisms unlike anything previously characterized is the open problem.

# References {-}
