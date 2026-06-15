#!/usr/bin/env bash
#
# auto_shutdown.sh — wait for the tier-1 matrix to finish, save results to S3,
# then terminate this Lambda instance. Designed to run detached (nohup) on the
# GPU box so the run cleans up after itself with zero babysitting.
#
# SAFETY: terminates ONLY if the S3 upload succeeds. If the upload fails, it
# leaves the box running so nothing is lost.
#
# Usage (on the box):
#   LAMBDA_API_KEY=secret_... nohup bash scripts/auto_shutdown.sh >~/watchdog.out 2>&1 &
#
# Override defaults via env: INST, RESULTS_S3, POLL_SECONDS.

set -uo pipefail

INST="${INST:-8c5ae89d58d9491cb0350f41e9673cd7}"
RESULTS_S3="${RESULTS_S3:-s3://microbe-foundation-esm2-perprotein/tier1_results/}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOG="$HOME/auto_shutdown.log"

cd "$HOME/microbe-foundation" || { echo "repo not found" | tee -a "$LOG"; exit 1; }

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "watchdog started (instance=$INST). Waiting for matrix shards to finish..."

# 1. Wait until every tier1_runs.sh shard has exited.
#    The [t] bracket trick keeps pgrep from matching its own command line.
while pgrep -f '[t]ier1_runs.sh' >/dev/null 2>&1; do
    sleep "$POLL_SECONDS"
done

DONE_COUNT="$(ls runs/tier1/*.json 2>/dev/null | wc -l | tr -d ' ')"
log "all shards finished. metrics JSONs present: ${DONE_COUNT}/36"

# 2. Upload everything under runs/ (metrics, predictions, checkpoints, logs) to S3.
log "uploading results to ${RESULTS_S3} ..."
aws s3 sync runs/ "$RESULTS_S3" --no-progress >>"$LOG" 2>&1
RC=$?
log "s3 sync exit code: ${RC}"

if [ "$RC" -ne 0 ]; then
    log "UPLOAD FAILED -- leaving instance RUNNING so results are not lost. Investigate, then terminate manually."
    exit 1
fi

# 3. Upload succeeded -> safe to terminate.
log "upload OK. Terminating instance ${INST} ..."
curl -s -u "${LAMBDA_API_KEY:?set LAMBDA_API_KEY}:" \
    -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"${INST}\"]}" >>"$LOG" 2>&1
log "terminate request sent. Goodbye."
