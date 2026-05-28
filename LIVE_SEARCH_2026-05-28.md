# Live literature search — 2026-05-28

Follow-up to `CITATION_AUDIT.md`. Targeted search to verify no paper occupies the *reframed* microbe-foundation scope (BacDive-native multi-task + chemotaxonomy + MediaDive + family-held-out splits). Performed via `agent-browser` against Google Scholar, bioRxiv, and Semantic Scholar.

## Verdict

**No new direct competitors found.** The reframed positioning holds. Four new related-work papers should be cited but none cover the full project scope.

## New papers surfaced (not in prior audit)

### 1. MiGenPro (Loomans, Suarez-Diez, Schaap, Saccenti, Koehorst — Aug 2025)

- DOI: `10.1101/2025.08.21.671437` (bioRxiv preprint)
- Affiliation: Wageningen University
- Scope: Linked-data workflow for genome→trait ML. Predicts 4 traits: motility, Gram stain, optimal temperature range, sporulation. Classical ML with 5-fold CV.
- **Relation to microbe-foundation:** Related workflow tool, not a foundation model. Narrow trait set, no chemotaxonomy / medium / FAME, no phylogeny-aware splits. Cite as related workflow.

### 2. Münch, Safaei, ..., McHardy — Comparative Assessment of LLMs for Microbial Phenotype Annotation (Nov 2025)

- DOI: `10.1101/2025.11.24.690272`
- Affiliation: Helmholtz Centre for Infection Research + Harvard Chan + others
- Scope: Benchmarks 50+ text LLMs (incl. Claude Sonnet 4, GPT-5 family) for microbial phenotype annotation **from text**.
- **Relation to microbe-foundation:** Orthogonal — annotates from literature text, not genome sequence. Cite as alternative approach; could complement genome-based prediction.

### 3. Hoffert, Gorman, Lladser, Fierer — "A periodic table of bacteria" (Jul 2025)

- DOI: `10.1101/2025.07.11.664459`
- Affiliation: U Colorado Boulder
- Scope: Organizes 50,745 genomes into trait-space clades via Haar wavelet transformation. Uses **predicted trait values from existing tools** for 6 traits (oxygen tolerance, autotrophy, chlorophototrophy, max growth rate, GC, genome size).
- **Relation to microbe-foundation:** Visualization/organization framework. Not a predictor itself. Could be a **downstream consumer** of microbe-foundation's predictions — interesting positioning angle.

### 4. Liu, Zhu, Chen, Ye, Zhang, Han — ML prediction of bacterial OGT from protein domain signatures (2025)

- Venue: BMC Genomics
- Scope: Single-trait optimal growth temperature predictor using protein domain features. Extension of Engqvist 2018.
- **Relation to microbe-foundation:** Cite as a Tier-2 single-trait comparator on the temperature head.

## White-space confirmations

| Claim | Confirmed? | Evidence |
|---|---|---|
| No FAME prediction from genome | ✅ | All "fatty acid + ML + bacteria" Scholar hits are about disease biomarkers, FAS enzyme engineering, or omega-3 supplements — none predict an organism's FAME profile from its genome. |
| No comprehensive cultivation-medium predictor since KOMODO 2015 | ✅ | Medium-related ML hits are about optimizing media for specific known organisms, not predicting medium from unknown genome. |
| No paper covers BacDive-native + chemotaxonomy + MediaDive + family-held-out splits scope | ✅ | Closest are MicroGenomer (ecophys), Bacformer (protein-of-proteins), BacPT (metabolic/ecological), BacBench (no chemotax, no MediaDive, genus-only). |

## Full direct-comparator list (for the benchmark)

The papers microbe-foundation must report numbers against, by head:

| Head | Comparators |
|---|---|
| Temperature | Engqvist 2018, Tome (Li & Engqvist 2019), Liu et al. 2025 BMC Genomics, MicroGenomer |
| pH | Ramoneda et al. 2023 *Sci Adv*, MicroGenomer |
| Oxygen | Wan et al. 2025 *Genomics*, BacBench |
| Carbon sources | GapMind 2022, BacBench |
| AMR | CARD/RGI 2023, AMRFinderPlus 2021, DeepARG 2018 |
| Cultivation medium | KOMODO 2015, user's prior model |
| Morphology / sporulation / motility / Gram | Traitar 2016, Brbić 2016, MiGenPro 2025, Bacformer phenotypic-trait tutorial, BacBench, Koblitz 2025 *Commun Biol* |
| Pathogenicity | PathogenFinder 2013, PaPrBaG 2017 |
| FAME / polar lipids / quinones / peptidoglycan | **No comparators — white-space.** |
| Cross-task / multi-task representation | MicroGenomer, Bacformer, BacPT, WGRL (Dufault & Moses 2025), MiGenPro |

## Methodology

- Tool: `agent-browser navigate --output text` (Playwright Chromium)
- Sources: Google Scholar (4 query angles, 2024+ filter), bioRxiv search (2 broad queries), Semantic Scholar Graph API
- Queries:
  - `"BacDive" "multi-task" genome prediction` (Scholar, 2024+)
  - `"fatty acid" composition predict genome bacterial "machine learning"` (Scholar, 2023+)
  - `"cultivation medium" OR "growth medium" prediction genome "deep learning" OR "neural network"` (Scholar, 2024+)
  - `microbial foundation model trait prediction bacdive` (bioRxiv)
  - `microbe species description genome prediction` (bioRxiv)
  - Semantic Scholar: `BacDive phenotype multi-task genome`, year 2024–2026
- Scholar hits skimmed: ~50; bioRxiv hits skimmed: ~50; SS hits: 20
