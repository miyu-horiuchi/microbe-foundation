# Citation Audit — microbe-foundation

Audit performed against publisher pages, PubMed, OpenReview, and bioRxiv source pages. All Job-1 items resolved; Job-2 unearthed three direct foundation-model competitors that materially affect positioning. See **Threat assessment** at the end.

---

## Job 1 — Verification of existing `VERIFY` entries

| # | Item | Status | Notes / change |
|---|---|---|---|
| 1 | Brbić et al. 2016 — landscape of microbial phenotypic traits | ✅ confirmed | Cite is correct: **NAR 44(21):10074–10090**, doi `10.1093/nar/gkw964`. Sci Rep memory was wrong; the original entry was already right. Removed `VERIFY` note. |
| 2 | Koblitz et al. 2023 — MediaDive (NAR) | ⚠️ corrected | Title was wrong ("ecosystem for standardized cultivation media" → actual: **"the expert-curated cultivation media database"**). Author list was missing 5 of the 9 authors. Now full: Koblitz, Halama, Spring, Thiel, Baschien, Hahnke, Pester, Overmann, Reimer. Volume/issue/pages/DOI confirmed (51(D1):D1531–D1538; `10.1093/nar/gkac803`). |
| 3 | Price — GapMind | ⚠️ split into two entries | The original bib entry conflated two papers. Split into: (a) `price2020gapmind` = amino-acid biosynthesis, **mSystems 5(3):e00291-20**, doi `10.1128/mSystems.00291-20`; (b) `price2022gapmindcarbon` = carbon-source utilization, **PLOS Genetics 18(4):e1010156**, doi `10.1371/journal.pgen.1010156`. **For the "predicts carbon source utilization from genome" claim, cite (b), not (a).** Note: the 2022 carbon-source paper is in PLOS Genetics, not mSystems. |
| 4 | Zhou et al. — DNABERT-2 | ✅ confirmed | ICLR 2024 poster (OpenReview `oMLQB4EZE1`, accepted 16 Jan 2024). Title corrected: "Multi-Species Genome**s**" (plural). Added 6th author Davuluri's middle initial V. Removed `VERIFY`. |
| 5 | Dalla-Torre et al. — Nucleotide Transformer | ⚠️ corrected | Now in print: **Nature Methods 22:287–297 (2025)**; published online 28 Nov 2024. DOI `10.1038/s41592-024-02523-z` confirmed. Renamed key `dallatorre2024nt` → `dallatorre2025nt`. |
| 6 | Brixi et al. — Evo 2 | ⚠️ corrected — major | **No longer a preprint.** Published in **Nature (2026)**, doi `10.1038/s41586-026-10176-5`. Renamed key `brixi2025evo2` → `brixi2026evo2`; title's spelling now matches journal ("modelling" with double-l). |
| 7 | Hwang — gLM vs gLM2 | ⚠️ corrected — split into two | Original entry `hwang2024glm2` was actually the **gLM (v1)** paper (Hwang et al., Nat Commun 15:2880, 2024, doi `10.1038/s41467-024-46947-9`), not gLM2. Renamed to `hwang2024glm` and added a separate `cornman2025glm2` entry for gLM2 (Cornman, West-Roberts, ..., Hwang; ICLR 2025; bioRxiv doi `10.1101/2024.08.14.607850`). gLM2 is from Tatta Bio, lead author Andre Cornman, not Hwang. |
| 8 | Richardson et al. 2019 — Genome Properties | ✅ confirmed | Author order matches the publisher page exactly (Richardson, Rawlings, Salazar, Almeida, Haft, Ducq, Sutton, Finn). NAR 47(D1):D564–D572 confirmed. Removed `VERIFY`. |
| 9 | Asgari et al. 2018 — MicroPheno | ✅ confirmed | Bioinformatics 34(13):i32–i42 — the **i**-paginated range confirms it is in the ISMB 2018 supplement issue. DOI `10.1093/bioinformatics/bty296` confirmed. Removed `VERIFY`; added a clarifying note. |

**Summary:** 4 confirmed clean, 5 corrected. The Evo 2 → Nature and gLM/gLM2 → two papers corrections are the most consequential.

---

## Job 2 — Targeted searches

### 2a. Pinned-down citations

| Cite request | Status | Resolved citation |
|---|---|---|
| Engqvist 2018 BMC Microbiology (OGT) | ✅ found | Engqvist (sole author), *BMC Microbiology* 18(1):177, 2018, doi `10.1186/s12866-018-1320-7`. Note: the 2018 paper is enzyme-annotation-vs-OGT correlative; the ML predictor itself is `li2019tome` below. |
| Tome (Li, Engqvist et al. 2019) | ✅ found | Li G, Rabe KS, Nielsen J, Engqvist MKM. *ACS Synthetic Biology* 8(6):1411–1420, 2019, doi `10.1021/acssynbio.9b00099`. (Published in *ACS Synth Biol*, not Bioinformatics.) |
| PathogenFinder (Cosentino 2013 PLoS One) | ✅ found | Cosentino, Larsen, Aarestrup, Lund. *PLoS ONE* 8(10):e77302, 2013, doi `10.1371/journal.pone.0077302`. |
| CARD/RGI (Alcock 2023 NAR) | ✅ found | Alcock et al., *NAR* 51(D1):D690–D699, 2023, doi `10.1093/nar/gkac920`. |
| ResFinder (Bortolaia 2020 JAC) | ✅ found | Bortolaia et al., *JAC* 75(12):3491–3500, 2020, doi `10.1093/jac/dkaa345`. |
| AMRFinderPlus (Feldgarden 2021 AAC) | ⚠️ venue corrected | **Not in AAC.** Feldgarden et al., *Scientific Reports* 11:12728, 2021, doi `10.1038/s41598-021-91456-0`. (The 2019 AMRFinder validation paper is in AAC; the 2021 AMRFinderPlus paper is in Sci Rep.) |
| DeepARG (Arango-Argoty 2018 Microbiome) | ✅ found | Arango-Argoty, Garner, Pruden, Heath, Vikesland, Zhang. *Microbiome* 6(1):23, 2018, doi `10.1186/s40168-018-0401-z`. |
| KOMODO (Oberhardt 2015 Nat Commun) | ✅ confirmed | Existing entry verified — *Nat Commun* 6:8493, doi `10.1038/ncomms9493`. No change. |
| GTDB-Tk (Chaumeil 2020 Bioinformatics) | ✅ found | Chaumeil, Mussig, Hugenholtz, Parks. *Bioinformatics* 36(6):1925–1927, 2020, doi `10.1093/bioinformatics/btz848`. |
| Plata et al. 2015 PNAS — metabolic capability | ❌ does not exist as stated | The Plata–Henry–Vitkup 2015 paper is in **Nature 517(7534):369–372**, not PNAS, doi `10.1038/nature13827`. Title is "Long-term phenotypic evolution of bacteria." Filed as `plata2015evolution`; user should re-check whether this is the intended paper or a different one. No PNAS paper by Plata in 2015 on bacterial metabolic capability was found. |
| HostPhinder (Villarroel 2016) | ✅ found | Villarroel, Kleinheinz, Jurtz, Zschach, Lund, Nielsen, Larsen. *Viruses* 8(5):116, 2016, doi `10.3390/v8050116`. |
| Galperin sporulation-gene cascades | ✅ found | Best single cite: Galperin, Mekhedov, Puigbo, Smirnov, Wolf, Rigden. *Environmental Microbiology* 14(11):2870–2890, 2012, doi `10.1111/j.1462-2920.2012.02841.x` — "minimal set of sporulation-specific genes." A 2022 update (Galperin et al., *J Bacteriol* 204:e00079-22, doi `10.1128/jb.00079-22`) is also available if a more recent cite is wanted. |

### 2b. Open searches

| Search | Finding |
|---|---|
| **2023–2026 "microbial foundation model" / "genome foundation model for trait prediction"** | **Three direct hits — see Threat assessment.** (i) MicroGenomer (Tsinghua, bioRxiv Dec 2025, `10.64898/2025.12.28.696777`); (ii) Bacformer (Wiatrak/Brbić/Floto, bioRxiv Jul 2025, `10.1101/2025.07.20.665723`); (iii) BacPT / Bacterial Proteome Foundation Model (bioRxiv Mar 2026, `10.64898/2026.03.07.710335`, with earlier preprint `10.1101/2025.03.19.644232`). |
| **2023–2026 cultivation-medium ML beyond KOMODO** | The strongest signal is Koblitz et al. *Commun Biol* 8 (2025), doi `10.1038/s42003-025-08313-3` — from the BacDive curators themselves, RF baselines on 8 traits including oxygen tolerance, growth temperature range, sporulation, motility, Gram stain. No new dedicated cultivation-medium predictor was found beyond KOMODO. A "MediaMatch" XGBoost model (PMC12495271) predicts 16S → medium binary for 45 media — a narrower-scope successor to KOMODO. |
| **2023–2026 ESM-2 for organism-level prediction** | Bacformer (above) explicitly aggregates per-protein ESM-2 embeddings to the organism level. BacPT (above) is literally trained on ESM-2 embeddings of whole proteomes. No third independent line of work yet. |
| **Evo / gLM2 for downstream organism-level trait prediction** | No published paper specifically uses Evo or gLM2 embeddings as fixed features for organism-level trait prediction yet. Evo 2 (Nature 2026) discusses downstream uses but does not run trait benchmarks; gLM2 (ICLR 2025) targets per-protein function and operon prediction. **This is an open niche.** |
| **pH-optimum prediction from genome** | Best cite: Ramoneda et al. *Science Advances* 9(17):eadf8998 (2023), doi `10.1126/sciadv.adf8998`. 56-gene logistic model on ~1470 samples. |
| **MIC prediction with ML** | Best organism-specific: Nguyen et al., *J Clin Microbiol* 57(2):e01260-18 (2019), doi `10.1128/JCM.01260-18` — XGBoost on ~5,278 nontyphoidal *Salmonella* genomes, 15 antibiotics, 95% ±1-dilution accuracy. (Pataki et al. exists for *E. coli* but the *Salmonella* + ESKAPEE Nguyen line of work is the dominant cite.) |
| **FAME composition prediction from genome** | ❌ **No paper found.** All FAME-ML work in the literature uses measured FAME profiles as the **input** for taxonomic identification (e.g., Slabbinck et al. 2009; Kaminski et al. 2008), not as an **output** predicted from genome sequence. This is a clear white-space for microbe-foundation. |

---

## Threat assessment

Three 2025–2026 preprints occupy substantial territory that microbe-foundation has been positioning into. The good news: no single one of them covers the **full** BacDive/IJSEM trait surface or stakes the "automate the species description from genome" claim. The thesis remains defensible but needs sharper framing.

### 1. MicroGenomer (bioRxiv 10.64898/2025.12.28.696777, Dec 2025) — HIGHEST OVERLAP

- Self-describes as **"a foundation model for transferable microbial genome representations enabling multi-scale genomic understanding and ecophysiological trait prediction"** — almost word-for-word the microbe-foundation pitch.
- 470M-parameter transformer, three-stage training on 234.5 Gbp, GTDB marker-gene mid-training, task-specific post-training.
- Wet-lab validation specifically on **optimal growth pH and temperature** for newly isolated strains.
- Affiliation: Shenzhen International Graduate School, Tsinghua University.
- **Gap microbe-foundation can still claim:** (a) MicroGenomer's trait set is ecophysiological (oxygen, pH, temperature, probiotic) — it does **not** target the full BacDive description (medium recipe, morphology, chemotaxonomy, AMR, pathogenicity); (b) it is DNA-LM-based, not ESM-2/proteome-based; (c) no public BacDive-aligned multi-task leaderboard.
- **Action:** read this paper first; reframe microbe-foundation as "full species-description multi-task" vs MicroGenomer's "ecophysiology-focused".

### 2. Bacformer (Wiatrak, …, Brbić, Weimann, Floto; bioRxiv Jul 2025, doi `10.1101/2025.07.20.665723`) — HIGH OVERLAP

- Crucially, this is **co-authored by Brbić and Weimann** — the authors of the two flagship prior trait-prediction papers `brbic2016landscape` and `weimann2016traitar`. They have moved into the foundation-model space.
- Trained on ~1.3M bacterial genomes as ordered sequences of proteins; tutorial-confirmed "Bacformer for phenotypic traits prediction."
- Predicts: PPI, operons, gene essentiality, protein function, **phenotypic traits**, and can design synthetic genomes.
- **Gap microbe-foundation can still claim:** the headline tasks in Bacformer are protein-protein/operon/essentiality; phenotypic trait prediction appears as one of several downstream applications, not the central evaluation. The unified BacDive/IJSEM trait-prediction benchmark is still unclaimed.
- **Action:** carefully read the phenotypic-trait section of Bacformer; cite it as a peer foundation model and clearly differentiate (proteome-of-proteins vs whole-trait benchmark; their per-task heads vs your masked-label joint training).

### 3. BacPT / Bacterial Proteome Foundation Model (bioRxiv 10.64898/2026.03.07.710335, Mar 2026; earlier preprint 10.1101/2025.03.19.644232) — MODERATE OVERLAP

- "Bacteria Proteome Transformer" trained on ESM-2 embeddings of ~33,140 whole bacterial proteomes — architecturally **very close** to what microbe-foundation is doing with mean-pooled ESM-2 embeddings.
- Predicts enzyme activity, BGCs, **metabolic traits**, ecological interaction outcomes.
- Reports outperforming Traitar and plain ESM on per-metabolite prediction.
- **Gap microbe-foundation can still claim:** BacPT's trait targets are metabolic/ecological, not the full taxonomist's description set; ~33k genomes is two orders of magnitude smaller than the Bacformer/MicroGenomer pretraining corpora; the "masked-label multi-task joint training across heterogeneous trait categories" framing is not their framing.
- **Action:** treat as the most direct architectural comparator. The "ESM-2 embeddings → organism-level prediction" framing is no longer novel as of March 2026.

### Secondary related work (worth citing but not threat-level)

- **WGRL** (bioRxiv 2025, doi `10.1101/2025.04.01.646674`): k-NN evaluation across 25 phenotypes; smaller in ambition but shows the broader trend toward whole-genome representation learning for multi-phenotype evaluation.
- **Koblitz et al. 2025 Commun Biol** (`10.1038/s42003-025-08313-3`): from the BacDive team — RF baselines on 8 traits using Pfam features. **This is the most important baseline microbe-foundation must beat per-trait** to claim foundation-model gains.
- **Gómez-Pérez & Keller 2025 NAR Genomics Bioinf** (`10.1093/nargab/lqaf174`): NLP-from-literature + genome — complementary but methodologically orthogonal.

### Bottom line for positioning

The "first microbial foundation model" claim is no longer defensible — MicroGenomer, Bacformer, and BacPT have all staked it within the last 12 months. The remaining defensible claims for microbe-foundation are:

1. **Coverage breadth.** None of the three competitors covers the full BacDive/IJSEM trait surface (medium, morphology, chemotaxonomy including FAME, AMR phenotype, pathogenicity, ecology). MicroGenomer is ecophysiology-focused; Bacformer foregrounds protein-level tasks; BacPT focuses on metabolic/ecological function.
2. **Masked multi-task formulation against heterogeneous missing-label structure** — this is the right algorithmic framing given how sparse BacDive labels are per-trait; none of the three explicitly frames the problem this way.
3. **Unified, phylogeny-aware, public trait benchmark + leaderboard.** The space lacks one; Brbić-style splits are still the right comparison and no one has published a unified leaderboard.
4. **FAME / chemotaxonomy prediction from genome** is true white-space — no ML/pathway predictor exists in the literature.

---

## Top 3 papers to read first

1. **MicroGenomer** — bioRxiv `10.64898/2025.12.28.696777`. The most direct positioning threat; settle whether microbe-foundation has a meaningfully different trait surface and architecture before writing the framing.
2. **Bacformer** — bioRxiv `10.1101/2025.07.20.665723`. Co-authored by the people who wrote the two papers microbe-foundation already builds on. Read closely to understand exactly what phenotypic-trait results they report, because reviewers will compare against this.
3. **Koblitz et al. 2025 Commun Biol** — `10.1038/s42003-025-08313-3`. The strongest per-trait single-task baselines on curated BacDive data, from the BacDive curators themselves. This is the score-to-beat for any "foundation model wins" claim.
