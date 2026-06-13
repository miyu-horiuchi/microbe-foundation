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
Never let scripts fall back to the Anaconda interpreter. Either:
- invoke `/usr/local/bin/python3` explicitly, or
- add an interpreter guard at the top of `scripts/tier1_runs.sh` and python
  helpers that selects system python and rejects `/opt/anaconda3`.

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
