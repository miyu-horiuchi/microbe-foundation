# Related Work

> Draft scaffold for the microbe-foundation paper / README.
> Citations are in `references.bib`. Entries marked `VERIFY` in the .bib need confirmation before use.

## Trait prediction from microbial genomes

Predicting organism-level phenotypes from microbial genome sequence has a long history. Early work formalized the task as binary classification over hand-curated trait labels, using gene-family presence/absence as features. **Traitar** [@weimann2016traitar] established the canonical pipeline — Pfam-domain features fed into per-trait SVMs across 67 phenotypes — and remains the standard baseline. **Brbić et al.** [@brbic2016landscape] extended this to 424 traits across ~3,000 prokaryotes and, critically, introduced phylogeny-aware evaluation splits that prevent label leakage between closely related strains; their splits are still the right comparison for any new method. **MicroPheno** [@asgari2018micropheno] explored k-mer-based features as a lighter-weight alternative, and **Genome Properties** [@richardson2019genome] codifies rule-based inference from InterPro annotations. The **Madin et al.** trait synthesis [@madin2020synthesis] consolidated phenotypic measurements for ~170k bacteria and archaea and is the primary label resource for any modern effort.

What unites this prior work is a per-trait, feature-engineered formulation: one model per phenotype, hand-designed inputs, no shared representation across tasks. microbe-foundation departs from this by training a single shared genome encoder with masked multi-task heads, allowing learned representations to transfer across phenotypes and exploit the heavy missing-label structure of the underlying databases.

## Cultivation medium prediction

A narrower line of work addresses cultivation medium recommendation — predicting which growth medium will support a given uncultured microbe. **KOMODO** [@oberhardt2015komodo] established the Known Media Database and a first ML formulation linking 16S identity to medium composition. **MediaDive** [@koblitz2023mediadive] is the modern curated medium database underlying recent ML approaches, including our prior single-task model that motivated this work. Adjacent metabolic-reconstruction tools such as **GapMind** [@price2022gapmind] predict carbon-source utilization from genome content and provide a complementary, mechanistic view. microbe-foundation subsumes cultivation medium prediction as one head in a broader multi-task setting, hypothesizing that joint training with related traits (e.g., oxygen requirement, temperature optimum, autotrophy) improves data efficiency.

## Genome and protein language models

The encoder side of microbe-foundation rests on the rapid maturation of sequence language models. Protein LMs, exemplified by **ESM-2** [@lin2023esm2], produce per-residue embeddings that pool to strong organism-level representations when applied across a full predicted proteome. DNA LMs include **DNABERT-2** [@zhou2024dnabert2], **Nucleotide Transformer** [@dallatorre2024nt], **HyenaDNA** [@nguyen2023hyenadna], and the long-context **Evo** [@nguyen2024evo] and **Evo 2** [@brixi2025evo2] models from Arc Institute, the latter trained at genome scale on prokaryotes. **gLM2** [@hwang2024glm2] occupies a useful middle ground by treating genomes as sequences of protein tokens. We adopt mean-pooled ESM-2 embeddings as our primary encoder for the initial multi-task model, with DNA-LM and gLM2-style encoders reserved for ablation, on the working hypothesis that organism-level traits are dominated by coding-region signal.

## Foundation models in biology

The framing of microbe-foundation as a "foundation model" follows precedents in adjacent biological domains. **Geneformer** [@theodoris2023geneformer] and **scGPT** [@cui2024scgpt] established the pattern of a pretrained encoder with many fine-tuning heads in single-cell biology; **ESMFold** [@lin2023esm2] demonstrated transfer of self-supervised protein representations to structure prediction. To our knowledge, no comparable artifact exists at the level of whole-microbe trait prediction: prior microbial trait work is task-siloed and feature-engineered, and prior microbial genome LMs are pretrained but not evaluated on a unified multi-task trait benchmark. microbe-foundation aims to close this gap.

## Positioning

The contribution of microbe-foundation is therefore not a new encoder, nor a new trait database, but the assembly: a unified, phylogeny-aware benchmark spanning the trait categories established by [@weimann2016traitar; @brbic2016landscape; @madin2020synthesis], a multi-task model with a modern sequence encoder trained against masked labels, and a public leaderboard and checkpoint release to anchor follow-on work.
