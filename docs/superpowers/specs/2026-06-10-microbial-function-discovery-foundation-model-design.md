# Microbial Function Discovery Foundation Model

**Date:** 2026-06-10
**Status:** Draft for review
**Topic:** Build a microbial genome foundation model and discovery engine for useful-function
prediction in sequenced but uncultured microbes.

---

## Goal

Build the core model and benchmark for **useful-function discovery from microbial genomes**.
The target user has genome sequences, MAGs, SAGs, or protein FASTA from organisms that may never
have been cultured. They want to know what those organisms can do, which ones are worth pursuing,
and what evidence supports the prediction.

The model should answer:

- What biological functions is this microbe likely to perform?
- Which application areas does it look useful for?
- Which genes, proteins, domains, or pathways support that prediction?
- How risky is the organism from a biosafety/pathogenicity perspective?
- How much should we trust the prediction under taxonomic novelty?

The company-level positioning is:

> Microbe Foundation searches microbial dark matter for useful biology. Given genome sequence, it
> predicts function, application potential, biosafety risk, confidence, and the gene-level evidence
> behind each result.

## Non-goals

- **Not a single vertical-only model.** The long-term product covers environmental, therapeutic,
  industrial, biofuel, food, and agriculture applications. The implementation should use a shared
  foundation encoder with application-specific heads, not isolated one-off models.
- **Not a wet-lab replacement.** The output is a ranked, evidence-backed shortlist for validation,
  not a claim that a function has been experimentally proven.
- **Not organism-level causality.** Gene/pathway evidence supports model predictions. It does not
  prove that deleting or adding a gene will cause the organism-level phenotype without experiments.
- **Not only cultured organisms.** Cultured strains provide supervision, but the design is judged by
  transfer to taxonomically novel genomes and uncultured MAG/SAG-style inputs.
- **Not a chatbot.** The first product surface is a discovery/ranking system with structured outputs,
  not free-form biological advice.

## Product Shape

The product should behave like a microbial discovery engine:

1. User provides one genome or a large candidate set.
2. The system predicts useful functions and application scores.
3. The system ranks organisms by target use case.
4. Each result includes evidence genes/pathways, uncertainty, and biosafety warnings.
5. The user exports candidates for wet-lab validation, downstream annotation, or partner review.

Primary use cases:

| Application area | Example predictions |
|---|---|
| Environmental / terraforming | carbon fixation, nitrogen cycling, sulfur metabolism, stress tolerance, metal reduction, plastic degradation, desiccation/radiation/salinity tolerance |
| Therapeutics / antimicrobials | biosynthetic gene cluster potential, antimicrobial peptide potential, pathogen suppression, microbiome-relevant functions, toxin and virulence risk |
| Biofuels / industrial enzymes | cellulose/xylan/lignin degradation, lipid accumulation, fermentation pathways, thermostable enzymes, acid/salt tolerance |
| Food / fermentation / agriculture | fermentation traits, flavor/aroma metabolism, probiotic-relevant functions, plant growth promotion, nitrogen fixation, spoilage and safety flags |
| Biosafety | pathogenicity, virulence factors, AMR phenotype, toxin genes, confidence and novelty caveats |

## Architecture

### Shared Genome Encoder

The shared encoder represents a genome as a set of biological evidence tokens:

- protein embeddings from frozen or fine-tuned protein language models
- gene/domain annotations such as Pfam, eggNOG, KEGG, COG, TIGRFAM, and CAZy
- pathway/module completeness features
- optional nucleotide-context embeddings for regulatory or non-coding signals
- taxonomic novelty features used for calibration, not label leakage

The current paper result already supports this direction: per-protein attention can find
virulence-relevant proteins, but family-level generalization is the binding constraint. The next
encoder should therefore focus less on "does attention beat mean pooling?" and more on robust
function transfer across unfamiliar clades.

Recommended first model:

- protein set encoder over per-protein ESM-2 embeddings
- pathway/domain token stream for interpretable functional evidence
- set attention or Perceiver-style latent bottleneck to model combinations of genes
- parallel lightweight heads for compositional traits where mean-like aggregation is sufficient
- calibrated uncertainty head conditioned on split distance / taxonomic novelty

### Application Heads

One shared encoder should feed separate output heads:

- environmental function head
- therapeutics / antimicrobial potential head
- industrial enzyme / biofuel head
- food / agriculture head
- biosafety head
- cultivation support head, used as an enabling capability for downstream validation

Each head can use its own label space and metrics while sharing the same genome representation.
This keeps the broad company vision without forcing every application to have equal data maturity
on day one.

### Discovery Ranking Layer

The product layer turns model outputs into ranked candidates:

```text
candidate_score =
  function_score
  * confidence
  * novelty_weight
  * biosafety_penalty
  * evidence_quality
```

The exact scoring function should be configurable by application. For example, therapeutics may
weight novelty highly but penalize virulence and AMR risk, while industrial enzyme discovery may
weight stability and pathway completeness more heavily.

## Outputs

For each genome, the model should produce structured output:

```json
{
  "genome_id": "candidate_001",
  "application_scores": [
    {
      "area": "biofuels_industrial",
      "score": 0.82,
      "confidence": 0.71,
      "top_functions": ["cellulose degradation", "thermostable glycoside hydrolases"]
    }
  ],
  "functions": [
    {
      "name": "cellulose degradation",
      "score": 0.88,
      "evidence": [
        {"type": "protein", "id": "protein_123", "annotation": "GH5 glycoside hydrolase"},
        {"type": "pathway", "id": "CAZy cellulase module", "status": "partial"}
      ]
    }
  ],
  "biosafety": {
    "pathogenicity_risk": 0.12,
    "amr_risk": 0.08,
    "warnings": []
  },
  "novelty": {
    "nearest_training_family": "unknown_or_distant",
    "generalization_risk": "high"
  }
}
```

## Benchmark

The flagship benchmark should test whether the model can discover useful functions in organisms
outside the taxonomic comfort zone.

### Primary Protocol

- family-held-out evaluation as the minimum standard
- phylum/class-held-out challenge subsets where label coverage supports it
- close homolog removal for gene-level function claims
- taxonomy-majority and nearest-neighbor baselines for every label family
- separate metrics for prediction accuracy, ranking quality, calibration, and evidence quality

### Core Metrics

| Metric | Purpose |
|---|---|
| AUROC / macro-F1 / accuracy | Basic per-function prediction quality |
| precision@k | Whether the top ranked candidates are useful for a search workflow |
| calibration error | Whether confidence is trustworthy under novelty |
| novelty-stratified performance | Whether the model fails gracefully on distant clades |
| evidence enrichment | Whether top genes/pathways match known functional databases |
| ablation sensitivity | Whether predictions depend on the evidence the model highlights |

### Baselines

Every benchmark report should include:

- taxonomy-majority baseline
- nearest-neighbor by genome/protein similarity
- eggNOG/KEGG/Pfam logistic or tree baseline
- ESM-2 mean-pool baseline
- per-protein attention baseline from the current paper
- BacBench or other public benchmark numbers where labels overlap

## Data Sources

The first implementation should organize labels into application panels:

| Panel | Candidate sources |
|---|---|
| Traits / phenotype | BacDive, MediaDive, BacBench-overlap labels |
| Pathways / metabolism | KEGG, MetaCyc, eggNOG, COG, GTDB-linked annotations |
| Industrial enzymes | CAZy, BRENDA-derived enzyme labels where licensing allows, UniProt reviewed annotations |
| Antimicrobials / BGCs | antiSMASH/BiG-SCAPE outputs, MIBiG, BAGEL/AMP catalogs |
| Biosafety | VFDB, CARD/AMRFinderPlus, BacDive/BSL-derived labels |
| Environment | IMG/M, MGnify, GTDB metadata, curated biogeochemical pathway markers |

The data system should preserve provenance per label. A prediction is only as useful as the
label definition behind it, especially for weak labels mined from annotations.

## Implementation Phases

### Phase 1: Define the Useful-Function Benchmark

Deliverables:

- `FUNCTION_DISCOVERY_BENCHMARK.md`
- application panel schema
- label provenance table
- baseline list and evaluation rules
- JSON output contract for per-genome predictions and ranked search results

This phase should not require new model training. It turns the product direction into a benchmark
others can understand and that future model work can target.

### Phase 2: Build the First Multi-Panel Dataset

Deliverables:

- function label matrix keyed by genome id
- panel-level train/val/test splits with family-held-out primary evaluation
- gene/pathway evidence tables for attribution validation
- baseline feature matrices from existing eggNOG/Pfam/ESM-2 pipelines

The initial dataset can reuse BacDive, existing per-protein ESM-2 artifacts, eggNOG/Pfam features,
VFDB attribution code, and current split infrastructure.

### Phase 3: Train the Shared Encoder + Heads

Deliverables:

- shared genome encoder using protein and pathway/domain evidence
- application-specific output heads
- calibrated novelty/uncertainty reporting
- benchmark results against Phase 1 baselines

The first success criterion is not "beats every method on every panel." It is: beats simple
annotation and mean-pool baselines on at least one high-value panel while producing evidence that
is biologically coherent and useful for ranking.

### Phase 4: Discovery Engine Demo

Deliverables:

- upload one genome or batch candidate set
- choose target application
- ranked candidate table
- function predictions with confidence
- evidence genes/pathways
- biosafety and novelty caveats
- exportable JSON/CSV report

This should extend the existing public tool contract rather than replacing it.

## Risks and Mitigations

- **Scope too broad.** Mitigation: broad shared encoder, but benchmark and launch by panels. Each
  panel has its own label quality and success criteria.
- **Weak labels create false confidence.** Mitigation: preserve label provenance, separate curated
  labels from annotation-derived labels, and report confidence/calibration by provenance class.
- **Taxonomy confounding dominates.** Mitigation: taxonomy baselines are mandatory; primary
  evaluation is family-held-out or harder; no claim is accepted without novelty-stratified results.
- **Attribution looks persuasive but is not faithful.** Mitigation: use external database enrichment,
  matched controls, and evidence ablation as in the pathogenicity paper.
- **MAG completeness/contamination hurts predictions.** Mitigation: include assembly quality fields,
  warn on incomplete genomes, and train/evaluate with MAG-like perturbations.
- **Commercial story becomes vague.** Mitigation: product surface is search/ranking for concrete
  target functions, not a generic "biology AI" dashboard.

## Testing and Validation

- Unit tests for schema loading, panel definitions, and prediction JSON validation.
- Split tests proving families do not cross train/val/test boundaries.
- Baseline regression tests for taxonomy-majority and nearest-neighbor baselines.
- Smoke test on a small genome subset that produces a full ranked discovery report.
- Evidence validation tests for known panels, starting with VFDB-style enrichment and ablation.
- Calibration reports stratified by taxonomic novelty and genome completeness.

## Open Decisions

1. **First panel to benchmark deeply.** Recommendation: start with environmental/industrial
   functions plus biosafety, because they support many downstream applications and can reuse
   pathway/domain evidence.
2. **Initial model backbone.** Recommendation: reuse per-protein ESM-2 + eggNOG/Pfam/KEGG tokens
   first, then compare larger protein/DNA encoders once the benchmark exists.
3. **Public release shape.** Recommendation: release the benchmark and demo before claiming a
   production-grade foundation model. The benchmark creates credibility and a target for the model.

## Decisions Locked

- Broad company vision: useful-function discovery across uncultured microbial life.
- Product shape: shared foundation model plus application-specific heads plus discovery ranking.
- Required outputs: functions, application scores, biosafety, confidence, and gene/pathway evidence.
- Primary technical challenge: cross-clade generalization, not just within-family accuracy.
- First repo deliverable: benchmark/product specification before new model implementation.
