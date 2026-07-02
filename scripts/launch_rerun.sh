#!/usr/bin/env bash
#
# launch_rerun.sh — one-shot, typo-proof launcher for the attention/family rerun.
# Bakes in INST + LAMBDA_API_KEY (no env vars on the command line = no glued-on
# "nohup" typo), caps concurrency, and REFUSES to start if a rerun is already
# running (this is what stopped the 8x over-launch / oversubscription mess).
#
# Usage on the box:   bash scripts/launch_rerun.sh
#
set -euo pipefail

# INST and LAMBDA_API_KEY must be supplied by the environment (never hardcode the
# key in a committed file). Export them before running, e.g.:
#   INST=<instance-id> LAMBDA_API_KEY=secret_... bash scripts/launch_rerun.sh
: "${INST:?set INST=<lambda-instance-id> so the box can self-terminate when done}"
: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY=secret_... so terminate is authorized}"
export INST LAMBDA_API_KEY
export WORKERS="${WORKERS:-3}"   # 3 runs x 3 workers = 9 dataloader procs, safe on 30 cores
export MAXPAR="${MAXPAR:-3}"     # at most 3 family runs at once

cd "$HOME/microbe-foundation"

# Guard: never stack a second instance. ([r]... avoids matching this shell.)
running="$(pgrep -fc '[r]erun_attn_family' || true)"
if [ "${running:-0}" -gt 0 ]; then
    echo "ABORT: a rerun is already running (${running} instance/s). Not launching another."
    pgrep -af '[r]erun_attn_family' || true
    exit 1
fi

: > "$HOME/rerun.log"          # fresh log so diagnostics are unambiguous
nohup bash scripts/rerun_attn_family.sh >"$HOME/rerun.out" 2>&1 &
echo "launched rerun pid=$! (single instance, MAXPAR=${MAXPAR}, WORKERS=${WORKERS})"
