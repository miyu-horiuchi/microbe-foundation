# Dev Environment Notes

_Last updated: 2026-06-13_

## The numpy / Anaconda hang (sandbox)

### Symptom
Any script that imports `numpy` (or `pandas`) under the **Anaconda interpreter**
(`/opt/anaconda3/bin/python3`) **hangs indefinitely** during import. The process
spins emitting `sysctlbyname denied` messages and never completes.

### Root cause
Anaconda's numpy build probes CPU/hardware info via the `sysctlbyname` syscall at
import time. This sandbox **denies/blocks** that syscall, and the import spins
instead of returning cleanly. This is an **environment/sandbox limitation, not a
bug in this repo**.

### What we ruled out
- **Thread / core-detection env vars do NOT fix it.** Tested with all of:
  `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `OPENBLAS_CORETYPE=Haswell`,
  `OPENBLAS_MAIN_FREE=1` — still hangs. So it is not OpenBLAS thread/core
  auto-detection.
- A forced `faulthandler.dump_traceback_later(..., exit=True)` could not even get
  the process to exit, confirming the block is deep in a C syscall holding the GIL.

### The working path
Use **system Python**: `/usr/local/bin/python3`. It imports numpy/pandas fine in
this sandbox. All analysis scripts and the `ANALYSIS_ONLY=1` path in
`scripts/tier1_runs.sh` run cleanly under it.

### Recommendation
Never let scripts fall back to the Anaconda interpreter.

`scripts/tier1_runs.sh` now has an **interpreter guard** that does this
automatically: it routes every `python3` call through an interpreter that can
actually import numpy. It probes candidates (`python3` on PATH, then
`/usr/local/bin/python3`, then `/usr/bin/python3`) with an ~8s watchdog and
skips any that hang on the numpy import.

- On the **GPU box (Lambda)** the torch-enabled interpreter is usually the PATH
  `python3` and is selected automatically. If it isn't, set `MICROBE_PY`:
  ```bash
  MICROBE_PY=/path/to/python ANALYSIS_ONLY=1 bash scripts/tier1_runs.sh
  ```
  `MICROBE_PY`, when set, is trusted as-is (no probing).
- A hung Anaconda import can sit in an uninterruptible syscall that even SIGKILL
  cannot reap; the guard kills + disowns and moves on rather than blocking.

---

## Project status (feat/set-transformer-tier1)

Committed:
- Set Transformer pooler, family-balanced sampling, seed aggregation
- Cross-clade retrieval on the trained model
- Failure-mode taxonomy (Table 20) + coverage panel (Table 21)
- Three remedies quantified: clade coverage, stronger encoder, label fixes
  (Tables 22–26)
- Multilabel reformulation + Elkan-Noto PU correction
  (cultivation_medium F1 0.38 -> 0.64)
- `tier1_runs.sh` wired for larger-ESM extraction + full analysis regen
  (`ANALYSIS_ONLY=1` CPU-only fast path)

Pending / GPU-dependent:
- Larger-ESM (650M / 3B) embedding extraction + full training matrix
- Multi-seed runs for mean +/- 95% CI tables

---

## Where the per-protein embeddings live (IMPORTANT)

The full ~100GB per-protein ESM-2 embeddings are **NOT on this laptop** (only the
manifest + an 86MB smoke subset). They were extracted on **Modal** by
`modal_esm2_perprotein.py` and persisted to a Modal Volume:

- **Volume name:** `microbe-esm2-perprotein`
- **Layout:**
  - `proteins/<bid>.txt.gz` -- cached protein AA sequences (Phase A, NCBI fetch)
  - `<bid>.npy`             -- float16 [n_proteins, 640] 150M embeddings (Phase B)
  - `manifest.parquet`
- Modal CLI is installed + authed locally (`~/.modal.toml`).

Verify it still exists:
```bash
modal volume ls microbe-esm2-perprotein            # expect proteins/, manifest.parquet, many *.npy
modal volume ls microbe-esm2-perprotein | wc -l    # -> approaches ~19,600
```
Pull 150M embeddings locally (only needed for laptop/non-Modal training):
```bash
modal volume get microbe-esm2-perprotein "*.npy" ./data/esm2_perprotein
modal volume get microbe-esm2-perprotein manifest.parquet ./data/esm2_perprotein
```

### Recommended path: train ON Modal (no 100GB download, no Lambda)
Because the embeddings already live on the Modal Volume and proteins are cached,
the cleanest plan is Modal-native:
1. Train the tier-1 matrix as Modal GPU functions reading `/out/<bid>.npy` from
   the Volume directly (no transfer). (Needs a small `modal_train_tier1.py`.)
2. **650M stronger-encoder lever is cheap** -- re-run Phase B with
   `--model facebook/esm2_t33_650M_UR50D` (proteins cached -> no NCBI re-fetch).

### KNOWN FIX needed before 650M re-embed
`modal_esm2_perprotein.py` Phase B writes `<bid>.npy` flat to the Volume root and
**skips any genome whose `<bid>.npy` already exists**. Re-embedding at 650M on the
same Volume would therefore skip everything (150M files present) and produce no
650M output. Fix: model-tag the npy output dir (e.g. `/out/emb_650M/<bid>.npy`)
and the manifest, keeping `proteins/` shared. Backward-compatible if the legacy
tag defaults to the flat root layout.

## scripts/lambda_launch.sh
Lambda provisioner kept as a fallback (e.g. if Modal capacity is short). Needs a
networked terminal + `LAMBDA_API_KEY`. Not the primary path given the data is on
Modal.
