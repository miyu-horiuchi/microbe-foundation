#!/usr/bin/env bash
#
# run_all.sh — end-to-end microbe-foundation pipeline.
#
# Each step is idempotent and resumable; rerunning skips work already done.
# Steps:
#   1. fetch_bacdive.py        ~2-3 hr full / ~30 s smoke
#   2. parse_bacdive.py        ~30 s
#   3. splits.py               ~2 s
#   4. vocab.py                ~5 s
#   5. extract_genome_accessions.py
#   6. compute_esm2_features.py  GPU recommended; many hours at full scale
#   7. model.py --save-metrics   ~10 min on CPU
#   8. paper/generate_tables.py
#   9. compare_to_priors.py
#
# Usage:
#   bash scripts/run_all.sh                # full pipeline (current --end on fetch)
#   bash scripts/run_all.sh --smoke        # IDs 1-1000, smallest ESM-2 model
#   bash scripts/run_all.sh --skip-fetch   # skip step 1 (already have raw JSONL)
#   bash scripts/run_all.sh --skip-features # skip step 6 (already have features.npz)
#
# Stops on the first failed step. All output goes to stdout/stderr; nothing
# is captured to log files automatically — use `tee` if you want logs.

set -euo pipefail

# Parse flags
SMOKE=0
SKIP_FETCH=0
SKIP_FEATURES=0
for arg in "$@"; do
    case "$arg" in
        --smoke)         SMOKE=1 ;;
        --skip-fetch)    SKIP_FETCH=1 ;;
        --skip-features) SKIP_FEATURES=1 ;;
        --help|-h)
            cat <<'HELP'
run_all.sh — end-to-end microbe-foundation pipeline.

Each step is idempotent and resumable; rerunning skips work already done.

Steps:
  1. fetch_bacdive.py            ~2-3 hr full / ~30 s smoke
  2. parse_bacdive.py            ~30 s
  3. splits.py                   ~2 s
  4. vocab.py                    ~5 s
  5. extract_genome_accessions.py
  6. compute_esm2_features.py    GPU recommended; many hours at full scale
  7. model.py --save-metrics     ~10 min on CPU
  8. paper/generate_tables.py
  9. compare_to_priors.py

Usage:
  bash scripts/run_all.sh                 # full pipeline
  bash scripts/run_all.sh --smoke         # IDs 1-1000, smallest ESM-2 model
  bash scripts/run_all.sh --skip-fetch    # skip step 1 (already have raw JSONL)
  bash scripts/run_all.sh --skip-features # skip step 6 (already have features.npz)

Stops on the first failed step.
HELP
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg" >&2
            echo "Try: bash $0 --help" >&2
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke-test settings
if [ "$SMOKE" = 1 ]; then
    FETCH_ARGS="--start 1 --end 1000 --workers 5"
    ESM_MODEL="facebook/esm2_t6_8M_UR50D"
    ESM_ARGS="--sample-n 20 --batch-size 4 --limit 50"
    EPOCHS=3
    RUN_NAME="smoke"
else
    FETCH_ARGS="--start 1 --end 200000 --workers 10"
    ESM_MODEL="facebook/esm2_t30_150M_UR50D"
    ESM_ARGS="--sample-n 50 --batch-size 16"
    EPOCHS=30
    RUN_NAME="esm2-150M-family"
fi

step() {
    local n="$1"; shift
    local desc="$1"; shift
    echo
    echo "============================================================"
    echo "STEP $n  $desc"
    echo "============================================================"
    "$@"
}

if [ "$SKIP_FETCH" = 0 ]; then
    step 1 "Fetch BacDive (resumable)" \
        python3 fetch_bacdive.py $FETCH_ARGS
else
    echo "[skip] step 1 fetch_bacdive.py"
fi

step 2 "Parse BacDive -> traits.parquet" \
    python3 parse_bacdive.py

step 3 "Build phylogeny-aware splits" \
    python3 splits.py

step 4 "Discover class vocabularies" \
    python3 vocab.py

step 5 "Extract NCBI genome accessions" \
    python3 extract_genome_accessions.py

if [ "$SKIP_FEATURES" = 0 ]; then
    step 6 "Compute ESM-2 features (requires Python 3.11+)" \
        python3 compute_esm2_features.py --model "$ESM_MODEL" $ESM_ARGS
else
    echo "[skip] step 6 compute_esm2_features.py"
fi

step 7 "Train multi-task model on family-held-out split" \
    python3 model.py \
        --features data/esm2_features.npz \
        --split-level family \
        --epochs "$EPOCHS" \
        --save-metrics "runs/${RUN_NAME}.json" \
        --run-name "$RUN_NAME"

step 8 "Refresh paper tables from data files" \
    python3 paper/generate_tables.py

step 9 "Compare against published prior-work numbers" \
    python3 compare_to_priors.py --our "runs/${RUN_NAME}.json"

echo
echo "============================================================"
echo "DONE.  Metrics:  runs/${RUN_NAME}.json"
echo "       Tables:   paper/tables/"
echo "       Draft:    paper/paper.md"
echo "============================================================"
