#!/usr/bin/env bash
#
# rerun_attn_family.sh — recover the 5 attention/family tier-1 runs that were
# lost when the attention shard hit an empty-label batch (now fixed in model.py:
# the training loop skips batches whose masked loss has no grad_fn).
#
# Runs all 5 missing configs IN PARALLEL on one GPU, then uploads everything
# under runs/ to S3 and (if INST + LAMBDA_API_KEY are set) terminates the box.
#
# SAFETY GATES (added after an early launch self-terminated a box mid-sync):
#   * PRE  — refuses to start if an `aws s3 sync` is still running or if the
#            per-protein embeddings are incomplete vs the manifest.
#   * POST — only terminates the box if the S3 upload succeeded AND all 5
#            expected metrics files were produced. Otherwise it leaves the box
#            up so nothing is lost and you can investigate.
#
# Missing configs (attention pooling, family split):
#   normal:   seed 1, seed 2            (seed 0 normal already exists elsewhere)
#   balanced: seed 0, seed 1, seed 2    (--balanced-families)
#
# Usage (on the box, AFTER the embedding sync is 100% done):
#   INST=<id> LAMBDA_API_KEY=secret_... \
#     nohup bash scripts/rerun_attn_family.sh >~/rerun.out 2>&1 &
#
# Env knobs: PERPROTEIN, OUT_DIR, EPOCHS, WORKERS, RESULTS_S3, MICROBE_PY.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Cap math-library threads so N parallel runs x M dataloader workers don't
# oversubscribe the cores (that thrashing made epoch 1 take 25+ min). One math
# thread per worker; throughput comes from running several jobs at once.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PERPROTEIN="${PERPROTEIN:-data/esm2_perprotein}"
OUT_DIR="${OUT_DIR:-runs/tier1}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-128}"
HIDDEN="${HIDDEN:-512}"
WORKERS="${WORKERS:-4}"          # lower than the matrix (8) since 5 procs share the GPU/CPU
MAXPROT="${MAXPROT:-2048}"
ST_HEADS="${ST_HEADS:-4}"
ST_INDUCING="${ST_INDUCING:-16}"
INST="${INST:-}"
RESULTS_S3="${RESULTS_S3:-s3://microbe-foundation-esm2-perprotein/tier1_results/}"
PY="${MICROBE_PY:-python3}"
LOG="$HOME/rerun.log"

# The 5 runs this script must produce (terminate only if all 5 land).
EXPECT=(
    attention_family_s1
    attention_family_s2
    attention_family_s0_balancedfamilies
    attention_family_s1_balancedfamilies
    attention_family_s2_balancedfamilies
)

mkdir -p "$OUT_DIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# ---- PRE-FLIGHT GATES ------------------------------------------------------
if [ ! -d "$PERPROTEIN" ]; then
    log "ERROR: per-protein dir '$PERPROTEIN' not found. Sync embeddings first. (box left up)"
    exit 1
fi
if pgrep -f "aws s3 sync" >/dev/null 2>&1; then
    log "ERROR: an 'aws s3 sync' is STILL RUNNING -- embeddings not fully downloaded yet."
    log "       Aborting WITHOUT touching the box. Wait for SYNC_DONE, then relaunch."
    exit 1
fi
NPY_HAVE="$(find "$PERPROTEIN" -name '*.npy' | wc -l | tr -d ' ')"
NPY_NEED="$("$PY" - "$PERPROTEIN" <<'PYEOF'
import sys, pathlib, pandas as pd
d = pathlib.Path(sys.argv[1])
try:
    m = pd.read_parquet(d / "manifest.parquet")
    print(int((m["status"] == "ok").sum()) if "status" in m.columns else len(m))
except Exception:
    print(-1)
PYEOF
)"
log "embeddings present: ${NPY_HAVE} .npy / manifest needs ${NPY_NEED}"
if [ "$NPY_NEED" -lt 0 ]; then
    log "ERROR: could not read manifest. Aborting WITHOUT touching the box."
    exit 1
fi
if [ "$NPY_HAVE" -lt "$NPY_NEED" ]; then
    log "ERROR: embeddings INCOMPLETE (${NPY_HAVE}/${NPY_NEED}). Aborting WITHOUT terminating."
    exit 1
fi

# ---- TRAIN (5 runs in parallel) --------------------------------------------
run_one() {  # run_one <seed> [--balanced-families]
    local seed="$1"; shift
    local extra=("$@")
    local tag="attention_family_s${seed}"
    [ "${#extra[@]}" -gt 0 ] && tag="${tag}_$(echo "${extra[*]}" | tr -d ' -')"
    local metrics="${OUT_DIR}/${tag}.json"
    if [ -f "$metrics" ]; then
        log "[skip] $tag (metrics already present)"
        return 0
    fi
    log "start $tag"
    "$PY" model.py \
        --per-protein "$PERPROTEIN" \
        --pooling attention \
        --split-level family \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --hidden "$HIDDEN" \
        --num-workers "$WORKERS" \
        --max-proteins "$MAXPROT" \
        --st-heads "$ST_HEADS" \
        --st-inducing "$ST_INDUCING" \
        --class-weights \
        --scheduler cosine \
        --save-metrics "$metrics" \
        --save-model "${OUT_DIR}/${tag}.pt" \
        --run-name "$tag" \
        --save-all-predictions "${OUT_DIR}/${tag}_preds.parquet" \
        "${extra[@]}" >"${OUT_DIR}/${tag}.trainlog" 2>&1
    log "done  $tag (exit $?)"
}

MAXPAR="${MAXPAR:-3}"   # concurrent runs; 5-at-once chokes the box on the family split
SPECS=( "1" "2" "0 --balanced-families" "1 --balanced-families" "2 --balanced-families" )
log "=== rerun start: ${#SPECS[@]} runs, max ${MAXPAR} parallel (epochs=$EPOCHS, workers=$WORKERS) ==="
for spec in "${SPECS[@]}"; do
    # throttle: wait until fewer than MAXPAR background jobs are running
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXPAR" ]; do sleep 5; done
    # shellcheck disable=SC2086
    run_one $spec &
done
wait

# ---- VERIFY which expected runs landed -------------------------------------
missing=0
for t in "${EXPECT[@]}"; do
    if [ -f "${OUT_DIR}/${t}.json" ]; then
        log "[ok]      ${t}.json"
    else
        log "[MISSING] ${t}.json"
        missing=$((missing + 1))
    fi
done
log "=== runs finished. produced $(( ${#EXPECT[@]} - missing ))/${#EXPECT[@]} expected metrics ==="

# ---- UPLOAD (always, to preserve whatever we got) --------------------------
log "uploading runs/ -> ${RESULTS_S3}"
aws s3 sync runs/ "$RESULTS_S3" --no-progress >>"$LOG" 2>&1
RC=$?
log "s3 sync exit code: ${RC}"

# ---- TERMINATE only if upload OK AND all 5 runs succeeded ------------------
if [ "$RC" -ne 0 ]; then
    log "UPLOAD FAILED -- leaving box RUNNING so nothing is lost."
    exit 1
fi
if [ "$missing" -ne 0 ]; then
    log "${missing} run(s) MISSING -- upload done, but leaving box RUNNING for investigation."
    log "   (re-run this script after fixing; completed runs will be skipped.)"
    exit 1
fi
if [ -n "$INST" ] && [ -n "${LAMBDA_API_KEY:-}" ]; then
    log "all 5 runs OK + upload OK. Terminating instance ${INST} ..."
    curl -s -u "${LAMBDA_API_KEY}:" \
        -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\":[\"${INST}\"]}" >>"$LOG" 2>&1
    log "terminate request sent. Goodbye."
else
    log "INST or LAMBDA_API_KEY not set -- upload done, box left RUNNING (terminate manually)."
fi
