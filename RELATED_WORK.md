# Related Work

> Draft scaffold for the microbe-foundation paper / README.
> Last updated 2026-05-28 (post-citation-audit, post-BacBench discovery, post-live-search).
> Citations are in `references.bib`.

## 1. Classical trait prediction from microbial genomes

Predicting organism-level phenotypes from microbial genome sequence has a long history. **Traitar** [@weimann2016traitar] established the canonical pipeline — Pfam-domain features fed into per-trait SVMs across 67 phenotypes — and remains the standard baseline. **Brbić et al.** [@brbic2016landscape] extended this to 424 traits across ~3,000 prokaryotes and introduced phylogeny-aware evaluation splits that prevent label leakage between closely related strains; their splits remain a useful reference for any new method. **MicroPheno** [@asgari2018micropheno] explored k-mer-based features as a lighter-weight alternative, and **Genome Properties** [@richardson2019genome] codifies rule-based inference from InterPro annotations. The **Madin et al.** trait synthesis [@madin2020synthesis] consolidated phenotypic measurements for ~170k bacteria and archaea and remains a primary label resource. A more recent workflow tool, **MiGenPro** [@loomans2025migenpro], reproduces this paradigm with semantic-web-style data integration and classical ML on four traits.

What unites this prior work is a per-trait, feature-engineered formulation: one model per phenotype, hand-designed inputs, no shared representation across tasks.

## 2. Cultivation medium prediction

A narrower line of work addresses cultivation medium recommendation. **KOMODO** [@oberhardt2015komodo] established the Known Media Database and a first ML formulation linking 16S identity to medium composition. **MediaDive** [@koblitz2023mediadive; @schober2025bacdive] is the modern curated medium database, formally linked to BacDive strain records. Despite a decade since KOMODO, no comprehensive successor model has appeared in the literature; adjacent metabolic-reconstruction tools such as **GapMind** [@price2022gapmindcarbon] cover carbon-source utilization but not medium composition. microbe-foundation subsumes cultivation medium prediction as one head in a broader multi-task setting, exploiting BacDive's MediaDive-linked records to predict the *set* of supporting media as a multilabel task.

## 3. Modern microbial foundation models (2024–2026)

Three preprints released in the past twelve months establish "foundation model for microbial genomes" as a live research direction:

- **MicroGenomer** [@wang2025microgenomer] (Tsinghua, Dec 2025) — a 470M-parameter DNA-LM trained on 234.5 Gbp, with task-specific heads for ecophysiological trait prediction (optimal pH, temperature, probiotic classification) and wet-lab validation.
- **Bacformer** [@wiatrak2025bacformer] (Jul 2025), co-authored by Brbić and Weimann — treats whole-bacterial genomes as ordered sequences of proteins, trained on ~1.3M genomes and ~3B proteins, with downstream tasks spanning operon prediction, protein-protein interactions, gene essentiality, and phenotypic traits. The companion **BacBench** repository [@bacbench2025] supplies a benchmark of 24,462 genomes × 139 traits with genus-disjoint splits.
- **BacPT** [@bacpt2026] (Mar 2026) — a transformer trained on mean-pooled ESM-2 embeddings of whole bacterial proteomes, with downstream evaluation on metabolic and ecological-interaction tasks.

These works overlap with microbe-foundation in approach but each occupies a slice of the trait space: MicroGenomer is ecophysiology-focused; Bacformer's phenotypic-trait head is one of many downstream uses; BacPT targets metabolic and ecological function. None cover the full BacDive/IJSEM species-description surface, none include chemotaxonomic prediction, none provide MediaDive-linked medium prediction, and none evaluate on family-or-higher-level held-out splits.

A complementary text-based approach by **Münch & McHardy** [@munch2025llmpheno] benchmarks 50+ general-purpose LLMs (Claude Sonnet 4, GPT-5) for phenotype annotation *from literature text* rather than genome sequence — orthogonal to microbe-foundation but a useful point of comparison for retrieval-augmented variants.

## 4. Per-trait single-task predictors as direct baselines

Several recent single-trait models provide head-level comparators:

- **Temperature optimum** — Engqvist 2018 [@engqvist2018temperature], Tome [@li2019tome], and Liu et al. 2025 [@liu2025ogt] on protein-domain signatures.
- **pH optimum** — Ramoneda et al. 2023 [@ramoneda2023ph], a 56-gene logistic model.
- **Oxygen tolerance** — Wan et al. 2025 [@wan2025oxygen].
- **Antibiotic resistance** — CARD/RGI [@alcock2023card], AMRFinderPlus [@feldgarden2021amrfinderplus], DeepARG [@arangoargoty2018deeparg], ResFinder [@bortolaia2020resfinder].
- **Pathogenicity** — PathogenFinder [@cosentino2013pathogenfinder].
- **Carbon-source utilization** — GapMind [@price2022gapmindcarbon].
- **Sporulation gene cascade** — Galperin et al. [@galperin2012sporulation].

The strongest contemporary multi-trait baseline is **Koblitz et al. 2025** [@koblitz2025bacdiveml], from the BacDive curators themselves: random-forest models on Pfam features for eight curated traits using the BacDive labels directly. This is the per-trait score microbe-foundation must beat to claim foundation-model gains.

## 5. Genome and protein language models

The encoder side of microbe-foundation rests on the rapid maturation of sequence language models. Protein LMs, exemplified by **ESM-2** [@lin2023esm2], produce per-residue embeddings that pool to strong organism-level representations when applied across a full predicted proteome. DNA LMs include **DNABERT-2** [@zhou2024dnabert2], **Nucleotide Transformer** [@dallatorre2025nt], **HyenaDNA** [@nguyen2023hyenadna], and the long-context **Evo** [@nguyen2024evo] and **Evo 2** [@brixi2026evo2] models from Arc Institute. **gLM** [@hwang2024glm] and its successor **gLM2** [@cornman2025glm2] occupy a useful middle ground by treating genomes as sequences of protein tokens. We adopt mean-pooled ESM-2 embeddings as the primary encoder for the v1 multi-task model — directly comparable to BacPT — with frozen Bacformer embeddings as a peer comparator, and DNA-LM ablations reserved for chemotaxonomic heads where non-coding regulatory signal may matter.

## 6. Foundation models in biology

The framing follows precedents in adjacent biological domains. **Geneformer** [@theodoris2023geneformer] and **scGPT** [@cui2024scgpt] established the pattern of a pretrained encoder with many fine-tuning heads in single-cell biology; **ESMFold** [@lin2023esm2] demonstrated transfer of self-supervised protein representations to structure prediction.

## 7. Concurrent and orthogonal work

Several adjacent efforts contextualize but do not overlap with microbe-foundation's core contribution:

- **WGRL** [@wgrl2025] — unsupervised whole-genome representation learning with k-NN evaluation on 25 phenotypes.
- **Hoffert & Fierer "periodic table of bacteria"** [@hoffert2025periodictable] — a visualization framework over predicted trait values; a potential downstream consumer of microbe-foundation predictions.
- **Gómez-Pérez & Keller 2025** [@gomezperez2025nlp] — NLP-from-literature combined with genome features.
- **HostPhinder** [@villarroel2016hostphinder] — phage host prediction (analogous task in a different domain).

## 8. Positioning

The contribution of microbe-foundation is **not** to be the first foundation model for microbial genomes — that ground is already held by MicroGenomer, Bacformer, and BacPT. The contribution is the **assembly of four currently-unfilled gaps** into a single artifact:

1. **Full BacDive species-description coverage.** The v1 schema spans 21 prediction heads across morphology, physiology, growth conditions, cultivation, safety, ecology, and chemotaxonomy — broader than any existing benchmark. MicroGenomer covers ecophysiology only; BacBench is genus-only with no chemotaxonomy, no MediaDive medium linkages, and no pathogenicity / BSL.
2. **Chemotaxonomic prediction from genome — confirmed literature white-space.** Live search (2026-05-28) found zero genome-to-FAME predictors in the literature. Fatty-acid composition prediction is the first head of a chemotaxonomy-from-sequence research line.
3. **MediaDive-linked cultivation medium prediction at scale.** The only prior model in this space is KOMODO 2015; modern multi-task formulations have not revisited it despite the MediaDive curation effort.
4. **Masked multi-task formulation against heterogeneous sparse labels and family-held-out splits.** BacDive's label sparsity per trait (5%–60% coverage) makes masked-loss multi-task training the natural formulation; existing benchmarks use random or genus-disjoint splits, which underestimate cross-family generalization. microbe-foundation reports family-held-out splits as the primary evaluation.

Each contribution is independently citeable. The combined artifact is a unified benchmark + multi-task model + checkpoint, positioned to anchor downstream work the way Brbić 2016 splits anchored the 2016–2022 generation.
