# Does It Actually Help? A Benchmark and Validated Metric for Comprehension-Gain from Generative-UI Explanations

**Submitted to:** Thinking Machines Lab — Interactivity Research Grants
**Primary contact:** [YOUR EMAIL]
**Principal Investigator:** [YOUR NAME], [ROLE], [INSTITUTION / "Independent Researcher"]
**Requested:** $100,000 + $25,000 Tinker credits · **Duration:** 6 months

---

## 1. Summary

When an AI system explains a complex result, it increasingly does so not in prose but by *generating an interface* — an SVG chart, an annotated diagram, an interactive HTML widget. Thinking Machines explicitly flags this ("Explaining complex information with generative UI") as a frontier of interactivity. But the field has no rigorous way to answer the only question that matters: **did the generated UI actually make a human understand the information better and faster — or did it just look good?**

Today these outputs are judged by *generation quality* (does it render, is it factually accurate, does a rater prefer it). Generation quality is not comprehension gain. A beautiful, fluent, confidently-wrong chart can *lower* understanding while scoring high on every current proxy. Interactivity that doesn't measurably improve human understanding is decoration, not collaboration.

We will build **ComprehendUI**: an open benchmark, a human-grounded metric, and a validated automated proxy that measure the *comprehension gain* a generative-UI explanation delivers — accuracy of understanding, time-to-correct, calibration of the human's confidence, and downstream decision quality. The central contribution is **construct validity**: we don't just propose a number, we adversarially demonstrate that it measures comprehension and not confounds (aesthetics, verbosity, prior knowledge). The deliverable is a benchmark the open community can use to push generative-UI explanation forward on the metric that actually counts.

---

## 2. Why this problem, why now

- **Generative output is newly capable.** Frontier models can now emit competent SVG/HTML/diagrams on demand. The capability arrived faster than any way to evaluate whether it *helps the person on the other side*.
- **The obvious metrics are confounded.** "A rater preferred output B" conflates aesthetics, length, fluency, and the rater's prior knowledge with actual understanding. Optimizing those proxies risks training models to produce persuasive-looking explanations rather than clarifying ones — a subtle but real safety/trust failure for interactive systems.
- **This is a measurement problem, and measurement is tractable for the open community.** It needs no proprietary training run — it needs careful task design, human studies, and validation. That is precisely where independent researchers can contribute a fair, reproducible standard, and it maps directly onto the grant's stated criteria of *construct validity* and *simplicity & generality*.

---

## 3. Research agenda

### 3.1 The benchmark (task suite)
A curated set of ~200–300 **complex-information items** spanning 3–4 modalities of difficulty where a UI plausibly helps:
- **Quantitative results with uncertainty** (e.g. an experimental result with confidence intervals — does the UI convey the uncertainty, not just the point estimate?)
- **Multi-variable / tabular relationships** (trends, interactions, comparisons)
- **Process / causal structure** (multi-step pipelines, dependency graphs)
- **Spatial / structural information** (layouts, geometries)

Each item ships with: the underlying ground-truth information, a set of **probe questions** a person who truly understood it should answer correctly, and a **downstream decision task** (act-on-the-information) that understanding should improve.

### 3.2 The conditions
For each item, the model explains it under matched conditions: (a) plain text, (b) static generative UI (SVG/HTML), (c) interactive generative UI. Holding the underlying content fixed isolates the *interactivity/visual* contribution.

### 3.3 The metric (comprehension gain)
A human-grounded composite, each component pre-registered:
- **Comprehension accuracy** on probe questions (vs. a no-explanation and text-only baseline → *gain*, not absolute).
- **Time-to-correct** (does the UI make understanding faster, or just prettier?).
- **Confidence calibration** — the human's self-rated confidence vs. their actual correctness; a good explanation should *improve* calibration, a misleading one inflates confidence without accuracy. (This is the failure mode current proxies are blind to.)
- **Decision quality** on the downstream task.

### 3.4 Construct validity (the core contribution)
This is where most evals quietly fail, and where we concentrate effort:
- **Adversarial decoys:** purpose-built UIs that are fluent and attractive but subtly wrong or uninformative. A valid metric must score these *low*; a metric that rewards them is measuring polish, not comprehension.
- **Confound controls:** match/regress out length, render time, aesthetic rating, and participant prior knowledge, and report how much of the signal survives.
- **Convergent / discriminant validity:** show the metric tracks independent comprehension signals (e.g. free-recall, teach-back) and *diverges* from pure preference/aesthetic ratings.
- **Honest negative space:** we will document what the metric does *not* capture, the same way we'd report a label-noise floor rather than pretend perfect ceiling.

### 3.5 The automated proxy (simplicity & generality)
Human studies are the ground truth but don't scale. We will build and **validate** a model-based comprehension-gain proxy (an LLM-judge / probe-answering simulator, optionally fine-tuned using the Tinker credits) *against* the human ground truth, reporting agreement and failure cases. A validated cheap proxy is what lets anyone re-run the benchmark on a new model in an afternoon — the generality the grant asks for.

---

## 4. Timeline, milestones, deliverables (6 months)

| Month | Milestone | Deliverable |
|---|---|---|
| **1** | Task taxonomy; author/collect ~200–300 items with probe questions + decision tasks across 3–4 domains; **pre-register** metric definitions. | Item set v1 + pre-registration doc |
| **2** | Build the eval harness (model → generative UI) and the human-study web app (probes, timing, confidence capture); pilot (N≈20). | Open harness v0 + pilot report |
| **3** | Main human study: text vs. static-UI vs. interactive-UI across frontier + Tinker-accessible models. | Human comprehension-gain ground-truth dataset |
| **4** | Construct-validity battery: adversarial decoys, confound controls, convergent/discriminant validity. | Validity report (what the metric does / doesn't measure) |
| **5** | Build + validate the automated comprehension-gain proxy against human data; measure agreement. | Validated automated metric + agreement analysis |
| **6** | Public release: benchmark, dataset, harness, leaderboard, preprint; reproducibility pass. | ComprehendUI release + paper |

**Final deliverables (all open, permissive license):** the benchmark dataset, the evaluation harness and human-study app, the validated automated proxy, a public leaderboard scoring current frontier models, and a preprint.

---

## 5. Why this team

My research is built around a single discipline: **never trust a metric until you've shown it measures what it claims.** In my current work on predicting biological phenotypes from genome sequence, I demonstrated that an intuitively-obvious confidence signal (novelty/out-of-distribution distance) actually *anti*-predicted model error — a result you only find if you stress-test the metric instead of assuming it. I build deliberately harder evaluation splits rather than the easy ones that flatter a model, and I report against noise floors rather than a fictional perfect ceiling. That construct-validity-first instinct is exactly what this proposal requires and what the grant explicitly rewards.

I also work hands-on at the generative-UI surface this benchmark studies: I generate SVG/HTML to explain complex quantitative results (including communicating uncertainty, the hardest case here) and ship public interactive tools that put model outputs in front of real users. So the proposed work is not a pivot — it is the formalization and rigorous measurement of something I already practice.

*[Add: degree / lab / advisor; any co-PIs or contributors with one-line roles. CVs attached separately.]*

---

## 6. Budget (1 page)

**Independent-researcher version (no institutional overhead) — total $100,000:**

| Line item | Amount |
|---|---|
| PI research time / stipend (6 months) | $55,000 |
| Human-subjects participant compensation (pilot + main study + validity battery; ~700 participants via crowd platform) | $18,000 |
| Expert annotation: authoring probe questions + ground-truth + inter-rater reliability | $8,000 |
| Compute / model API beyond Tinker credits (UI generation + proxy runs) | $12,000 |
| Infrastructure: study web app + public leaderboard + harness hosting (12 mo) | $4,000 |
| Platform & misc (crowd-platform fees, storage, tooling) | $3,000 |
| **Total** | **$100,000** |

*$25,000 Tinker credits* will be used to fine-tune and validate the automated comprehension-gain proxy against the human ground truth (Milestone 5).

**If applying through a university/institution:** indirect costs are capped at 10% ($10,000). Carve a $10,000 "Institutional overhead (10%)" line from the direct costs above (reduce the stipend line accordingly), or have the institution waive amounts above the cap, per the grant terms.

---

## 7. Organizational details

- **Applicant:** [Independent Researcher / Institution name]
- **Location:** [CITY, COUNTRY]
- **Tax ID / EIN:** [if applicable]
- **Administrative contact:** [NAME, EMAIL] *(if applying through an institution)*
- **Primary contact email:** [YOUR EMAIL]
