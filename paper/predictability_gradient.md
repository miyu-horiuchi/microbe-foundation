# When Does Attention Help? A Predictability Gradient for Genomic Trait Prediction, Validated by Virulence-Factor Attribution

**Miyu Horiuchi**
*Correspondence: miyuhpenn@gmail.com*

---

## Abstract

A bacterial genome is a *set* of proteins, and predicting an organism's phenotype from that set requires pooling thousands of protein representations into a fixed-size genome vector. The default choice — mean-pooling — treats every protein equally; learned attention-pooling can instead concentrate on a few. We ask a question that is usually answered by default rather than by evidence: **when does the pooling choice matter, and why?** We propose the **predictability gradient** hypothesis: pooling architecture matters in proportion to how *localized* a trait's genomic signal is. *Compositional* traits (e.g. oxygen tolerance, cell shape) reflect diffuse genome-wide signal and should be insensitive to pooling; *machinery* traits (e.g. pathogenicity, nutrient requirements) are determined by a handful of decisive genes and should benefit from attention. We test this on 19,592 bacterial genomes (82M per-protein ESM-2 embeddings) across 21 traits and three taxonomic generalization regimes. With three seeds per configuration, attention-pooling improves machinery traits **~4× more** than compositional traits (species-level gap in F1 gain +0.062 ± 0.011 vs +0.001 at family-level), confirming the gradient — and revealing that the advantage **collapses entirely under family-level covariate shift**. We then show the mechanism is genuine: for pathogenicity, the model concentrates 81% of its attention on five proteins, and those proteins are **5× enriched for known virulence factors** versus random proteins in the same genome (Wilcoxon $p=2.5\times10^{-7}$) and 3.2× versus non-pathogenic genomes ($p=6.8\times10^{-14}$); ablating them flips 23% of pathogenic predictions. The attention does not merely correlate with virulence — it locates adhesins, fimbrial ushers, and invasion loci, and the prediction depends on them. Our results give a predictive, mechanistically-grounded rule for when un-pooled protein representations are worth their cost, and identify cross-clade generalization — not pooling — as the binding constraint for genomic foundation models.

---

## 1. Introduction

Foundation models for bacterial genomes have advanced rapidly: protein language models such as ESM-2 [esm2] produce rich per-protein representations, and recent genome-level models [bacformer; microgenomer; bacpt] aggregate them to predict phenotype. A genome, however, is not a sequence but an unordered *set* of $P$ proteins ($P \approx 10^3$–$10^4$), so every such model must **pool** a variable-size set of protein vectors into one genome representation before prediction. This pooling step is almost always chosen by convention — typically a mean — and almost never studied.

This is a missed opportunity, because pooling encodes a strong inductive bias about *where a trait's signal lives in the genome*. Mean-pooling assumes the signal is diffuse: every protein contributes equally. Attention-pooling assumes the signal is localized: the model learns to up-weight a few decisive proteins. Which bias is correct is not a universal fact — it depends on the trait. Whether a bacterium is Gram-positive is a property of its entire cell envelope; whether it is pathogenic can hinge on a single secretion system.

We formalize this as the **predictability gradient** hypothesis:

> The benefit of attention-pooling over mean-pooling for a trait is proportional to how *localized* that trait's genomic determinants are. Diffuse "compositional" traits gain little; gene-specific "machinery" traits gain much.

This hypothesis is attractive because it is *falsifiable inside a single architecture*: hold the encoder fixed, swap only the pooling, and measure the gain as a function of trait class. We do exactly this, and add two things a benchmark number cannot provide. First, we vary the **generalization regime** (species-, genus-, family-held-out splits), turning the experiment into a test of whether the gradient survives distribution shift. Second, for the trait with the largest gain — pathogenicity — we ask whether the attention is *mechanistically real*: does it concentrate on the actual virulence machinery, and does the prediction causally depend on it? This converts a performance claim ("attention helps") into a scientific one ("attention helps because it finds the responsible genes").

**Contributions.**

1. **The predictability-gradient hypothesis and its validation.** We give a testable account of when set-pooling architecture matters for genomic prediction, and confirm it on 19,592 genomes / 21 traits with three seeds: attention helps machinery traits ~4× more than compositional traits, with non-overlapping error bars (§4).
2. **A covariate-shift result.** The gradient is strong within taxonomic distribution (species/genus) and vanishes at the family level (gain gap +0.062 → +0.001). The binding constraint for genomic foundation models is cross-clade generalization, not pooling (§4.2).
3. **Mechanistic attribution against a virulence-factor database.** For pathogenicity, attention concentrates (81% mass on 5 of ~3,800 proteins) and is enriched for VFDB virulence factors under two controls; ablation shows the prediction depends on those proteins; the hits are coherent adherence/invasion machinery (§5).
4. **A reproducible pipeline** (data, splits, checkpoints, three analysis scripts) releasing the per-protein feature set and the attribution tooling (§6, Reproducibility).

We are explicit about what this is *not*: not a new encoder (we freeze ESM-2), not a state-of-the-art benchmark sweep, and not a solution to cross-clade generalization — which our own results show is the open problem.

## 2. Related Work

**Genomic and protein foundation models.** ESM-2 [esm2] provides the frozen per-protein embeddings we pool. Genome-level models — Bacformer [bacformer], MicroGenomer [microgenomer], and BacPT [bacpt] — aggregate protein or gene representations for phenotype prediction but report benchmark accuracy without analyzing the pooling step or attributing predictions to specific genes. Our contribution is orthogonal and complementary: we hold the encoder fixed and study the aggregation choice and its mechanism.

**Microbial trait prediction.** Classical predictors (Traitar [traitar], Genome Properties [genomeprops], and KO/pathway-based methods) use hand-engineered gene-content features and per-trait classifiers. These implicitly assume localized, gene-presence signal — consistent with our "machinery" pole — but do not contrast it against a diffuse-signal baseline or quantify when the assumption pays off.

**Attention as explanation.** A literature cautions that attention weights are not, on their own, faithful explanations [jain2019attention; wiegreffe2019attention]. We take this seriously: our mechanistic claim does not rest on attention magnitudes but on (i) *external* validation against a curated virulence-factor database under matched controls and (ii) *causal* ablation. We also flag where ablation is partly mechanical (§5.3, §6).

**Multiple-instance and set learning.** Attention-pooling over instances is standard in multiple-instance learning [ilse2018attention]; a genome-as-bag-of-proteins is a natural instance. Our novelty is not the mechanism but tying its benefit to a measurable property of the label (trait localization) and validating the learned attention against ground-truth biology.

## 3. Setup

### 3.1 Data

Labels are drawn from BacDive, the largest curated bacterial phenotype resource, yielding **21 prediction heads** across seven biological blocks (morphology, physiology, growth conditions, cultivation, safety, ecology, chemotaxonomy). For each strain with an NCBI genome we predict open reading frames with pyrodigal and embed **every** protein independently with ESM-2 (`esm2_t30_150M`, 640-d), producing a ragged $[P_i, 640]$ matrix per genome with no pooling. The resulting per-protein corpus is **19,592 genomes / ~82M proteins (~105 GB)**, released on cloud storage (§Reproducibility). A mean-pooled $[640]$ baseline feature is computed from the same embeddings.

### 3.2 Trait taxonomy: compositional vs machinery

We pre-register a partition of the 21 heads into **compositional** (signal expected diffuse) and **machinery** (signal expected gene-localized), from biological first principles and *before* seeing pooling results:

- **Compositional (11):** gram stain, cell shape, motility, sporulation, oxygen tolerance, catalase, cytochrome oxidase, temperature class, pH class, halophily, pigmentation.
- **Machinery (8):** pathogenicity (human), pathogenicity (animal), cultivation medium, carbon utilization, metabolite production, antimicrobial-resistance phenotype, biosafety level, fatty-acid (FAME) profile.

(Two ecological/metadata heads — isolation source, country — are excluded from the gradient analysis as they are not biological traits.)

### 3.3 Model

A shared MLP encoder feeds 21 linear heads under a **masked multi-task loss** that contributes zero gradient for missing labels (BacDive coverage ranges 5–95% per head). The only architectural variable is the pooling:

- **Mean-pool:** genome vector $= \frac{1}{P}\sum_i x_i$.
- **Attention-pool:** a two-layer scorer $s_i = w^\top \tanh(W x_i)$, masked softmax $a = \mathrm{softmax}(s)$ over real proteins, genome vector $=\sum_i a_i x_i$. The weights $a_i$ are exposed for analysis.

Both share encoder/heads/loss; only pooling differs, isolating the variable of interest.

### 3.4 Evaluation regimes

We use **species-, genus-, and family-held-out** splits: train/val/test partitions in which no species (resp. genus, family) appears in more than one fold. Family-held-out is the hardest, approximating prediction for clades unlike anything seen in training — the regime relevant to uncultured "microbial dark matter." We report macro-F1 per head and, for the gradient, $\Delta\mathrm{F1} = \mathrm{F1}_{\text{attn}} - \mathrm{F1}_{\text{mean}}$, aggregated within trait class, mean ± std over **3 seeds** per configuration (20 epochs, batch 256, ≤4,096 proteins/genome).

## 4. The Predictability Gradient

### 4.1 Attention helps machinery traits, not compositional traits

Table 1 reports the per-class mean $\Delta\mathrm{F1}$ across seeds.

**Table 1. Gain from attention-pooling over mean-pooling ($\Delta$F1, mean ± std over 3 seeds).**

| Split | Compositional (n=11) | Machinery (n=8) | Gap (mach − comp) |
|---|---|---|---|
| species | +0.021 ± 0.002 | **+0.083 ± 0.012** | **+0.062** |
| genus | +0.016 ± 0.004 | **+0.067 ± 0.010** | **+0.052** |
| family | +0.009 ± 0.002 | +0.010 ± 0.003 | +0.001 |

At the species and genus levels the machinery gain exceeds the compositional gain by ~4×, with **non-overlapping** error bars (species: $0.083\pm0.012$ vs $0.021\pm0.002$). The compositional gain is small but positive and tight — attention does not *hurt* diffuse traits, it simply adds little, exactly as the hypothesis predicts: when signal is genome-wide, a weighted average and a flat average converge. The single largest per-head effects are pathogenicity (animal F1 0.26→0.50, human 0.16→0.32 at species) — the most gene-localized traits in the set.

### 4.2 The gradient collapses under covariate shift

The most consequential result is the family row. As the test distribution moves from same-species to same-family to *novel-family* organisms, the machinery advantage decays monotonically (+0.083 → +0.067 → +0.010) until the gradient is effectively gone (gap +0.001). Mean- and attention-pooling become indistinguishable precisely in the regime that matters for uncultured organisms. This localizes the field's bottleneck: the limiting factor is not how we pool proteins but whether *any* protein-set representation transfers across evolutionary distance. We return to this in §6.

## 5. Does the Attention Find the Right Genes?

A performance gain does not establish that attention is mechanistically meaningful — attention weights need not be faithful [jain2019attention]. We therefore validate the largest-gain trait, **pathogenicity**, against external ground truth (VFDB, 4,663 experimentally-verified virulence factors) with two matched controls and a causal ablation. We train single-task attention-pool models per pathogenicity head (so the shared pool specializes to the trait); both discriminate well on held-out test genomes (AUROC 0.88 animal, 0.85 human).

### 5.1 Attention concentrates

On held-out genomes the attention distribution is sharp: median normalized entropy 0.26 (animal), with the **top-5 of ~3,800 proteins carrying 81% of the attention mass** (top-1 alone ≈40%). Concentration is a precondition for an interpretable spotlight and is label-independent (the model always selects a few proteins); the question is *which*.

### 5.2 Top-attended proteins are enriched for virulence factors

We map each protein's attention rank to its amino-acid sequence (preserved in ORF order alongside the embeddings) and call a protein a virulence factor if it matches VFDB by diamond blastp ($E<10^{-5}$). Two controls (Table 2):

**Table 2. VFDB enrichment of top-5 attended proteins (animal head; species split).**

| Test | Top-attended | Control | Statistic |
|---|---|---|---|
| **Within-genome** (paired, vs random proteins from the *same* genome) | 28.1% VF | 5.9% | Wilcoxon $p=2.5\times10^{-7}$, $n{=}74$ |
| **Between-class** (vs top-attended in non-pathogenic genomes) | 28.1% | 10.9% | Fisher OR$=3.2$, $p=6.8\times10^{-14}$ |

The proteins attention selects are ~5× more likely to be known virulence factors than random proteins in the same genome, and the enrichment is 3.2× stronger in pathogenic than non-pathogenic genomes. The human head replicates (between-class OR 3.1, $p=3.8\times10^{-5}$; within-genome $p=0.034$ at $n{=}16$). The hits are not database noise: the most frequently selected VFs are coherent **adherence/invasion machinery** — fimbrial ushers `papC`/`mrkC`, filamentous hemagglutinin `fhaB`, attachment-invasion locus `ail`, type-IV pilus `pilQ`, flagellar `fliR`.

### 5.3 The prediction causally depends on them

Masking the top-5 attended proteins (of ~3,800) and re-predicting flips **22.7% of pathogenic calls** to non-pathogenic (animal; Wilcoxon vs random-protein masking $p=1.2\times10^{-4}$); the human head replicates (12.5%, $p=9\times10^{-3}$). Masking proteins ranked 6–10 produces a 27.5× smaller drop — the dependence is specific to the very top proteins. We note honestly that a top-vs-random ablation gap is *partly mechanical*: attention-pooling is a weighted sum, so removing high-weight elements changes the output more by construction. The non-trivial facts are therefore the **absolute** effect (removing 5 of ~3,800 proteins overturns a quarter of predictions) and its conjunction with §5.2 (those 5 proteins are the virulence machinery). Together they support a causal-mechanistic reading that neither establishes alone.

## 6. Discussion and Limitations

**What the results say.** Set-pooling architecture for genomic prediction should be chosen by trait localization, not convention: attention is worth its cost for gene-determined traits and not for diffuse ones, and — for pathogenicity — it works by locating the responsible genes. This gives practitioners a predictive rule and gives the field a mechanistic check (external attribution + ablation) that any genome model can adopt.

**Limitations**, stated plainly:

- **Covariate shift is unsolved and dominates.** The benefit evaporates at family-level holdout (§4.2). For the uncultured-organism application that motivates this work, this is the result that matters most, and it is negative for the pooling lever.
- **Weighted sum, not combinations.** Attention up-weights individual proteins; it cannot represent interactions ("gene A *with* gene B"). Pathogenicity is combinatorial in reality; a set-transformer that models protein interactions is the natural next architecture and may both raise accuracy and change the attribution story.
- **Enrichment, not coverage.** 68% of top-attended proteins are not VFDB hits; VFDB catalogs only *known* factors, so this is expected and our claim is strictly about enrichment, never that all attended proteins are virulence factors.
- **Single encoder scale.** All results use one ESM-2 size (150M, 640-d). Whether a larger encoder sharpens or flattens the gradient is untested.
- **Statistical scope.** The gradient uses 3 seeds; the human-head within-genome control is underpowered ($n{=}16$ pathogenic test genomes). The animal head and between-class tests are well-powered ($p<10^{-6}$); the small effects ($\le0.02$ $\Delta$F1) in the compositional column are within their own noise and we do not over-interpret them.

**Why this is a useful negative-and-positive result.** The positive half (a validated, mechanistically-explained predictability gradient) is a clean architectural insight; the negative half (it does not survive clade shift) redirects effort from the pooling question, which we consider resolved, to the generalization question, which we do not.

## 7. Conclusion

Pooling a genome's proteins is an inductive-bias choice, and the right choice is a measurable function of the trait: attention for localized "machinery" traits, mean for diffuse "compositional" traits. We validated this gradient with seeds, showed for pathogenicity that the attention concentrates on — and the prediction depends on — bona fide virulence factors, and found that the entire advantage is contingent on staying within taxonomic distribution. Reading the right genes is achievable; reading them for organisms unlike anything previously characterized is the open problem.

---

## Reproducibility

Code is public (`model.py --per-protein`; `extract_attention.py`, `vfdb_enrichment.py`, `ablate_attention.py`). The per-protein feature set (19,592 genomes, 105 GB), labels, splits, vocabularies, trained checkpoints, and per-genome attention tables are released; VFDB and diamond are public. All numbers in Tables 1–2 regenerate from the released checkpoints and scripts. Trait-class assignment, split definitions, and the $E<10^{-5}$ VF threshold are pre-specified.

## Data Availability

Per-protein embeddings: object storage (`s3://microbe-foundation-esm2-perprotein/`). Protein sequences and annotation references: persistent volumes. Labels/splits derive from BacDive (CC-BY 4.0) and are regenerable from public sources. See repository README, "Data & artifacts."

## References

*(BibTeX keys above resolve in the repository bibliography; abbreviated here.)*
esm2 (Lin et al., 2023), bacformer (Wiatrak et al., 2025), microgenomer (2025), bacpt (2026), jain2019attention (Jain & Wallace, 2019), wiegreffe2019attention (Wiegreffe & Pinter, 2019), ilse2018attention (Ilse et al., 2018), traitar (Weimann et al., 2016), genomeprops (Richardson et al., 2019), VFDB (Liu et al.), diamond (Buchfink et al., 2021), pyrodigal (Larralde, 2022), BacDive (Schober et al., 2025).
