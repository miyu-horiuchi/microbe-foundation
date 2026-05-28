# BacDive Label Coverage — Per-Trait Estimates

**Investigated:** 2026-05-28
**BacDive version probed:** v2 REST API (`api.bacdive.dsmz.de/v2/fetch`)
**Total strains in database:** **100,866** (per dashboard 2026-05-28)
**Total type strains:** ~22,125 (22% of all strains)
**License:** CC-BY 4.0 (free, even for ML training — citation required)
**API auth:** **No longer required as of Feb 2026** — `api.bacdive.dsmz.de/v2/fetch/{id}` is fully public

## Methodology

There is **no bulk-download endpoint and no aggregation endpoint** in the v2 API. The Koblitz et al. 2025 NAR paper (database paper) does **not publish a per-trait coverage table** — Figure 4 only covers oxygen tolerance, Gram stain, motility, and spore formation (the 4 traits they had genome-prediction models for). The 2025 Commun Biol paper (Koblitz et al., `s42003-025-08313-3`) reports specific dataset sizes only for the 6 traits they modeled: motility 7,090 strains; oxygen tolerance 8,405 data points; thermophilic growth retained 75.5% of temperature-labeled strains; Gram stain "~17 % of all strains." It also states their **selection threshold was ≥3,000 strains** for ML training.

To get coverage for every trait the user asked about, I drew a **random sample of 600 BacDive IDs** from the range [1, 170000], fetched each via the free v2 API in parallel, and got 324 valid strain records (the rest were unassigned IDs). All coverage numbers below are **point estimates plus 95% Wilson CI on the projected count** to the full 100,866-strain database.

Caveats:
- n=324 random strains → 95% CIs on rare traits are wide. Anything <1% in the sample (0 observations) has CI upper bound ~1,200 strains.
- The sample is across all strains, not weighted toward type strains. Chemotaxonomy is concentrated in type strains, so my estimates likely **under-count** chemotaxonomy fields slightly.
- BacDive's `Genome-based predictions` section was empty in all 324 records I fetched — this is the ML-imputed phenotypes from Koblitz et al. 2025, which **may require an API key** to access (the public `/fetch/` endpoint returns experimental data only).

---

## Coverage table

Estimated # of strains with each field populated, projected from n=324 random strains over the full 100,866-strain BacDive.

### Tier 1 — binary / categorical
| Trait | Field path | Sample n | % | Projected total | 95% CI | ≥500? |
|---|---|---:|---:|---:|---|---|
| **Gram stain** | `Morphology > cell morphology > gram stain` | 51/324 | 15.7% | **~15,900** | [12,300, 20,300] | ✅ confident |
| **Cell shape** | `Morphology > cell morphology > cell shape` | 51/324 | 15.7% | **~15,900** | [12,300, 20,300] | ✅ confident |
| **Motility** | `Morphology > cell morphology > motility` | 49/324 | 15.1% | **~15,300** | [11,700, 19,600] | ✅ confident |
| **Sporulation** | `multicellular morphology > spore formation` OR `P&M > spore formation` | 18/324 | 5.6% | **~5,600** | [3,600, 8,700] | ✅ confident |
| **Spore type** | (no dedicated field) | — | — | — | — | ❓ |
| **Pigmentation** | `Morphology > pigmentation` | 22/324 | 6.8% | **~6,800** | [4,600, 10,200] | ✅ confident |
| **Catalase (tested)** | enzyme value "catalase" | 47/324 | 14.5% | **~14,600** | [11,200, 18,900] | ✅ confident |
| **Cytochrome oxidase** | enzyme value "cytochrome oxidase" | 42/324 | 13.0% | **~13,100** | [9,800, 17,200] | ✅ confident |
| **Oxygen tolerance** | `P&M > oxygen tolerance` | 76/324 | 23.5% | **~23,700** | [19,300, 28,600] | ✅ confident |
| **Pathogenicity (human)** | `Interaction & safety > risk assessment > pathogenicity human` | 9/324 | 2.8% | **~2,800** | [1,500, 5,200] | ✅ confident |
| **Pathogenicity (animal)** | `pathogenicity animal` | 9/324 | 2.8% | **~2,800** | [1,500, 5,200] | ✅ confident |
| **Pathogenicity (plant)** | (no path found in sample) | 0/324 | 0% | likely <1,200 | [0, 1,200] | ⚠️ borderline |
| **Biosafety level (BSL)** | `risk assessment > biosafety level` | 84/324 | 25.9% | **~26,200** | [21,600, 31,200] | ✅ confident |

### Tier 2 — binned-continuous / curated
| Trait | Field path | Sample n | % | Projected total | 95% CI | ≥500? |
|---|---|---:|---:|---:|---|---|
| **Temperature growth (any)** | `Culture > culture temp` | 157/324 | 48.5% | **~48,900** | [43,400, 54,400] | ✅ confident |
| **pH growth (any)** | `Culture > culture pH` | 25/324 | 7.7% | **~7,800** | [5,300, 11,200] | ✅ confident |
| **NaCl / halophily** | `P&M > halophily` | 42/324 | 13.0% | **~13,100** | [9,800, 17,200] | ✅ confident |
| **Cultivation medium (any)** | `Culture > culture medium` | 131/324 | 40.4% | **~40,800** | [35,500, 46,300] | ✅ confident |
| **Cultivation medium (linked to MediaDive)** | `culture medium` entry with `link` containing `mediadive` | 90/324 | 27.8% | **~28,000** | [23,400, 33,200] | ✅ confident |
| **Carbon source utilization** | `P&M > metabolite utilization` | 95/324 | 29.3% | **~29,600** | [24,800, 34,800] | ✅ confident (per-substrate breakdown requires API per-strain analysis) |
| **Metabolite production** | `P&M > metabolite production` | 79/324 | 24.4% | **~24,600** | [20,200, 29,600] | ✅ confident |
| **API test panel (any)** | `P&M > API zym/20NE/50CHac/biotype100/etc.` | 68/324 | 21.0% | **~21,200** | [17,100, 26,000] | ✅ confident |
| **Isolation source** | `Isolation > isolation` | 194/324 | 59.9% | **~60,400** | [54,900, 65,600] | ✅ confident |
| **Geographic origin (country)** | `isolation > country` | 173/324 | 53.4% | **~53,900** | [48,400, 59,300] | ✅ confident |

### Tier 4 — chemotaxonomy (white-space candidates)
| Trait | Field path | Sample n | % | Projected total | 95% CI | ≥500? |
|---|---|---:|---:|---:|---|---|
| **Fatty acid profile (FAME)** | `P&M > fatty acid profile` | 23/324 | 7.1% | **~7,200** | [4,800, 10,500] | ✅ confident |
| **Polar lipid composition** | (no dedicated structured field found in v2 API) | 0/324 | 0% | likely <1,200 | [0, 1,200] | ⚠️ **see caveat** |
| **Peptidoglycan type (`murein`)** | `P&M > murein > type` | 4/324 | 1.2% | **~1,250** | [485, 3,160] | ⚠️ borderline |
| **Respiratory quinones** | (no dedicated structured field found in v2 API) | 0/324 | 0% | likely <1,200 | [0, 1,200] | ⚠️ **see caveat** |
| **mol% G+C content** | `Sequence information > GC content` | 62/324 | 19.1% | **~19,300** | [15,400, 24,000] | ✅ confident |

### Adjacent useful fields
| Trait | Field path | Sample n | % | Projected total |
|---|---|---:|---:|---:|
| Genome sequence available | `Sequence > Genome sequences` | 75/324 | 23.1% | **~23,300** |
| 16S sequence available | `Sequence > 16S sequences` | 91/324 | 28.1% | **~28,300** |
| Type strain | `Name > type strain == yes` | 73/324 | 22.5% | **~22,700** (matches dashboard's 22,125) |

The fact that my type-strain estimate (~22,700) matches the dashboard's published 22,125 within the CI is a **sanity check** that the sampling is representative.

---

## Caveats on chemotaxonomy "zero hits"

For polar lipids and respiratory quinones, my sample returned 0 hits in 324 strains. This is consistent with one of two possibilities:

**A) BacDive does not have a structured field for these traits in the v2 schema.**
I walked every nested key in the JSON for 324 strains plus a manual probe of strain `165467` (Bifidobacterium faecale) and could not find any key containing `quinone`, `isoprenoid`, `menaquinone`, or `polar lipid`. The `Physiology and metabolism` section enumerates: `metabolite utilization`, `enzymes`, `metabolite production`, `oxygen tolerance`, `metabolite tests`, `observation`, `halophily`, multiple `API ...` panels, `antibiotic resistance`, `fatty acid profile`, `spore formation`, `compound production`, `murein`, `nutrition type`, `tolerance`. No quinone / polar lipid keys.

The BacDive 2019 NAR paper documents that fatty acid profiles were added (395 initially) but does not mention quinones or polar lipids as structured fields. They are likely captured only in free-text observations or in the underlying literature references (which BacDive links via `Reference` blocks).

**B) They exist but are extremely rare (< ~0.4%).**
Unlikely given the literature cites these as standard taxonomic descriptors. The structured-field hypothesis is much more probable.

**Recommended next step before relying on these traits:** post a feature-request issue to the BacDive team (`bacdive-helpdesk@dsmz.de`) asking whether polar lipid and quinone composition will become structured fields. If not, parsing these from the linked IJSEM species descriptions (PDF text) becomes a real engineering task — but the upside is genuinely uncovered ML territory.

---

## Threshold verdict (≥500 labeled genomes bar)

### ✅ Clear ≥500 with high confidence (CI lower bound > 500)
Gram stain · Cell shape · Motility · Sporulation · Pigmentation · Catalase · Oxidase · Oxygen tolerance · Pathogenicity (human) · Pathogenicity (animal) · BSL · Temperature · pH · NaCl/halophily · Cultivation medium (incl. MediaDive-linked) · Carbon source utilization · Metabolite production · API panels · Isolation source · Geographic origin · **Fatty acid (FAME) profile** · mol% G+C · 16S/genome sequence

### ⚠️ Borderline (CI straddles 500 or just above)
- **Peptidoglycan type (murein)** — point estimate ~1,250 but lower CI bound 485 (right at the bar). Likely actually meets the bar but worth verifying with a larger pull.
- **Pathogenicity (plant)** — 0/324 in sample, upper CI bound 1,200. Possibly meets bar; need targeted query.

### ❌ Probably below threshold (or not a structured field)
- **Polar lipid composition** — no structured field detected; would need free-text mining or external data merge
- **Respiratory quinones** — no structured field detected; same
- **Spore type** (vs. spore formation binary) — no dedicated field seen
- Some specific carbon substrates if you require per-substrate ≥500 (depends on which substrate — the Madin/BacBench labels show ~2,900 strains per substrate)

---

## Recommendation: v1 microbe-foundation schema scope

### Include (high-confidence ≥500 labels each, BacDive-native)

**Morphology block** (~5–16k labels each, perfect Tier-1 baseline):
- Gram stain, cell shape, motility, sporulation, pigmentation

**Physiology block** (5–25k labels each):
- Oxygen tolerance, catalase, oxidase, halophily/NaCl, BSL, pathogenicity (human, animal)

**Growth conditions block** (8–50k labels each):
- Temperature growth (binned), pH growth (binned), cultivation medium ID (link to MediaDive)

**Carbon/metabolism block** (per-substrate ~3k each, ~30k strains with any data):
- Carbon source utilization matrix (~80 substrates from Madin overlap)
- Metabolite production
- API panel results (β-galactosidase, urease, indole, nitrate reduction, etc.)

**Environment/origin block** (~50–60k labels each):
- Isolation source (free text → category)
- Geographic origin (country)

**Chemotaxonomy white-space (THE differentiator)** (~7k strains):
- **Fatty acid (FAME) composition** — this is the one big chemotaxonomy field BacDive does structure, BacBench ignores, and clears the bar with confidence (~7,200 strains).
- mol% G+C of DNA (~19,300 strains)

### Defer / external integration needed
- **Polar lipids** and **respiratory quinones** — not structured in BacDive. Defer to v2 unless you do IJSEM PDF extraction (separate engineering project).
- **Peptidoglycan type** — borderline at the bar; include experimentally but expect noisy / small test sets per class.
- **Pathogenicity (plant)** — needs targeted query before committing.

### v1 schema in one sentence

> A 25-trait BacDive-native benchmark: 11 morphology+physiology Tier-1 binaries, 3 growth-condition continuous traits, ~80 carbon-substrate utilizations as a multi-label head, a 4-class pathogenicity head, a fatty-acid composition multi-output head (the chemotaxonomy differentiator), mol% G+C regression, and a 50k+ isolation-source/medium-context block — all benchmarked on family-held-out splits to surpass BacBench's genus-only protocol.

---

## Sources

- BacDive homepage (totals): https://bacdive.dsmz.de/
- BacDive REST API v2 (free, no auth, Feb 2026): https://api.bacdive.dsmz.de/
- Random sample of 324 strains fetched 2026-05-28 from `api.bacdive.dsmz.de/v2/fetch/{id}` (raw JSON cached at `/tmp/bacdive_sample.json`)
- BacDive 2025 (NAR database paper): Reimer LC et al. "BacDive in 2025: the core database for prokaryotic strain data." NAR 53(D1):D748 — https://academic.oup.com/nar/article/53/D1/D748/7848838 (PMC11701647)
- BacDive 2019 (NAR paper that introduced fatty acid profiles): Reimer LC et al. NAR 47(D1):D631
- Koblitz J et al. 2025 Commun Biol "Predicting bacterial phenotypic traits…" `s42003-025-08313-3` — PMC12145430 (cites motility 7,090; oxygen 8,405; trait selection threshold ≥3,000)
- BacDive Wikipedia (totals): https://en.wikipedia.org/wiki/BacDive
