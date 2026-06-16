#!/usr/bin/env bash
# lambda_full_launch.sh -- ONE local command to stand up a Lambda A100 and start
# the full LoRA encoder-adaptation run end to end. Run on your LAPTOP (it needs
# network + your ssh key; a sandboxed agent has neither).
#
# Every fragile manual step (scp creds, clone, write secrets, tmux launch) is
# done programmatically here, so nothing is hand-pasted and no space can be lost.
#
# Usage:
#   export LAMBDA_API_KEY=secret_...
#   ssh-add ~/.ssh/id_ed25519        # cache passphrase once (optional but nice)
#   bash scripts/lambda_full_launch.sh
#
# Knobs (env): GPU_KIND (set a multi-GPU type e.g. gpu_8x_a100_sxm4 / gpu_2x_a100
# to parallelise), SSH_KEY_NAME, REGION, BRANCH, plus any run_full.sh knob
# (EPOCHS, MAX_GENOMES, MODEL, MAX_PROTEINS, MODES, SEEDS, PARALLEL, NUM_GPUS,
# ENC_MICROBATCH, REMOTE_DIR) forwarded to the box. E.g. to relaunch ONLY the
# LoRA arm on a single H100 with the memory fix:
#   GPU_KIND=gpu_1x_h100_sxm5 MODES=lora bash scripts/lambda_full_launch.sh
set -euo pipefail

API="https://cloud.lambdalabs.com/api/v1"
GPU_KIND="${GPU_KIND:-gpu_1x_a100_sxm4}"
SSH_KEY_NAME="${SSH_KEY_NAME:-miyu-microbe-mac}"
REGION="${REGION:-}"
NAME="${NAME:-microbe-lora-full}"
BRANCH="${BRANCH:-feat/set-transformer-tier1}"
REPO_URL="${REPO_URL:-https://github.com/miyu-horiuchi/microbe-foundation}"
MODAL_TOML="${MODAL_TOML:-$HOME/.modal.toml}"
# Durable save sink on the box (NO Modal by default):
#   git  = box commits runs/lora and `git push`es it (auto-terminate only after a
#          confirmed push). Needs a GitHub token on the box; provisioned below if
#          one is available locally, otherwise the box keeps the commit and you
#          scp it off (non-terminating).
#   none = box leaves results in runs/lora and never auto-terminates; scp them off.
#   modal= legacy `modal volume put` (needs a usable Modal subscription).
SAVE_MODE="${SAVE_MODE:-git}"
# Hard cost backstop: a detached watchdog force-terminates the box after N hours
# no matter what (no-op if auto-terminate already fired). Always armed.
WATCHDOG_HOURS="${WATCHDOG_HOURS:-8}"
# owner/repo for the box's git push, derived from REPO_URL.
GH_REPO_SLUG="$(printf '%s' "$REPO_URL" | sed -E 's#^https?://github.com/##; s#\.git$##')"
# Optional GitHub token so the box can push results itself (hands-off). Sourced
# from GH_TOKEN/GITHUB_TOKEN, else `gh auth token`. If empty, SAVE_MODE=git still
# works -- it just keeps the commit on the box for you to scp (non-terminating).
GH_TOKEN_VAL="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -z "$GH_TOKEN_VAL" ] && command -v gh >/dev/null 2>&1; then
  GH_TOKEN_VAL="$(gh auth token 2>/dev/null || true)"
fi

# Optional run knobs forwarded to run_full.sh on the box (empty = its defaults).
# PARALLEL/NUM_GPUS make a multi-GPU box (GPU_KIND=gpu_8x_a100_sxm4 etc.) actually
# fan the (mode, seed) jobs out one-per-GPU instead of running them sequentially.
# (SAVE_MODE/GH_TOKEN are NOT forwarded here -- they go via ~/run_env.sh so the
# token never lands on the logged tmux command line.)
FWD=""
for k in MODEL MAX_GENOMES EPOCHS MAX_PROTEINS MODES SEEDS PARALLEL NUM_GPUS \
         ENC_MICROBATCH REMOTE_DIR AUTO_TERMINATE INCREMENTAL_MODAL; do
  v="${!k:-}"; [ -n "$v" ] && FWD="$FWD $k=$v"
done

: "${LAMBDA_API_KEY:?export LAMBDA_API_KEY=secret_...}"
# Modal creds are needed for (a) SAVE_MODE=modal and (b) the box's protein-data
# sync (`modal volume get`) on a fresh clone. Only hard-require them for the
# legacy modal save path; otherwise warn (the run can still proceed if the box
# already has data/esm2_proteins or you provide .modal.toml for the data sync).
if [ ! -f "$MODAL_TOML" ]; then
  if [ "$SAVE_MODE" = "modal" ]; then
    echo "missing $MODAL_TOML but SAVE_MODE=modal -- run 'modal token new' or pick SAVE_MODE=git/none" >&2
    exit 1
  fi
  echo "note: $MODAL_TOML not found -- not copying Modal creds. The box's protein-data sync uses 'modal volume get'; provide \$MODAL_TOML or pre-stage data/esm2_proteins on the box if the clone has none." >&2
fi
PY="$(command -v python3)"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

_api() {  # _api METHOD PATH [json-body]
  local m="$1" p="$2" b="${3:-}"
  if [ -n "$b" ]; then
    curl -fsS -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" -X "$m" "$API$p" -d "$b"
  else
    curl -fsS -u "$LAMBDA_API_KEY:" -X "$m" "$API$p"
  fi
}

# --- pick a region with capacity ---
region="$REGION"
if [ -z "$region" ]; then
  region="$(_api GET /instance-types | GPU_KIND="$GPU_KIND" "$PY" -c '
import sys, json, os
d = json.load(sys.stdin)["data"]
it = d.get(os.environ["GPU_KIND"])
regs = [r["name"] for r in it["regions_with_capacity_available"]] if it else []
print(regs[0] if regs else "")')"
fi
[ -z "$region" ] && { echo "No live capacity for $GPU_KIND. See: bash scripts/lambda_launch.sh list" >&2; exit 1; }

# --- launch ---
echo "Launching $GPU_KIND in $region (ssh key: $SSH_KEY_NAME)..."
body="$(REGION_PICK="$region" SSH_KEY_NAME="$SSH_KEY_NAME" GPU_KIND="$GPU_KIND" NAME="$NAME" "$PY" -c '
import json, os
print(json.dumps({
    "region_name": os.environ["REGION_PICK"],
    "instance_type_name": os.environ["GPU_KIND"],
    "ssh_key_names": [os.environ["SSH_KEY_NAME"]],
    "name": os.environ["NAME"]}))')"
ID="$(_api POST /instance-operations/launch "$body" \
      | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"]["instance_ids"][0])')"
echo "instance=$ID  (polling for boot...)"

IP=""
for i in $(seq 1 60); do
  sleep 10
  info="$(_api GET "/instances/$ID")"
  st="$(printf '%s' "$info" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"]["status"])')"
  IP="$(printf '%s' "$info" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"].get("ip") or "")')"
  echo "  [$((i*10))s] status=$st ip=${IP:-<pending>}"
  [ "$st" = "active" ] && [ -n "$IP" ] && break
done
[ -z "$IP" ] && { echo "instance not active yet; check dashboard. id=$ID" >&2; exit 1; }

# --- wait for sshd ---
echo "waiting for ssh on $IP ..."
for i in $(seq 1 40); do
  ssh "${SSH_OPTS[@]}" ubuntu@"$IP" true 2>/dev/null && break
  sleep 5
done

# --- provision + launch (programmatic; $ID/$LAMBDA_API_KEY expand locally) ---
if [ -f "$MODAL_TOML" ]; then
  echo "copying Modal creds (for protein-data sync / SAVE_MODE=modal)..."
  scp "${SSH_OPTS[@]}" "$MODAL_TOML" ubuntu@"$IP":~/
else
  echo "skipping Modal creds copy (SAVE_MODE=$SAVE_MODE)."
fi

echo "cloning repo @ $BRANCH..."
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "rm -rf microbe-foundation && git clone $REPO_URL && cd microbe-foundation && git checkout $BRANCH"

# ~/run_env.sh carries every secret/knob the run needs. Written via ssh STDIN (a
# heredoc string) so the GitHub token never appears as a process argument on the
# box. run_full.sh sources this file.
echo "writing ~/run_env.sh (creds + SAVE_MODE=$SAVE_MODE)..."
RUNENV="export INSTANCE_ID=$ID
export LAMBDA_API_KEY=$LAMBDA_API_KEY
export SAVE_MODE=$SAVE_MODE
export GH_REPO_SLUG=$GH_REPO_SLUG
export BRANCH=$BRANCH"
if [ -n "$GH_TOKEN_VAL" ]; then
  RUNENV="$RUNENV
export GH_TOKEN=$GH_TOKEN_VAL"
  echo "  GitHub token provisioned -> box can push results itself (hands-off)."
else
  echo "  no GitHub token available -> box will keep results committed locally; pull via scp (printed at end)."
fi
printf '%s\n' "$RUNENV" | ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "cat > ~/run_env.sh"
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "grep -v GH_TOKEN ~/run_env.sh"   # echo back WITHOUT the token

echo "starting run in tmux (knobs:${FWD:- defaults})..."
PREFIX=""; [ -n "$FWD" ] && PREFIX="export$FWD; "
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "cd microbe-foundation && tmux new-session -d -s full '${PREFIX}bash scripts/run_full.sh 2>&1 | tee ~/pilot.log'"

# Arm the hard cost cap (detached). Reads creds from ~/run_env.sh so the API key
# is never on the command line. No-op if the run auto-terminates earlier.
echo "arming watchdog cost cap (${WATCHDOG_HOURS}h)..."
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "cd microbe-foundation && source ~/run_env.sh && nohup bash scripts/lambda_watchdog.sh $WATCHDOG_HOURS >/tmp/watchdog.log 2>&1 & disown; sleep 1; head -1 /tmp/watchdog.log"

cat <<EOF

============================================================
RUNNING.  instance=$ID  ip=$IP  save_mode=$SAVE_MODE  watchdog=${WATCHDOG_HOURS}h
  watch:  ssh ubuntu@$IP 'tail -20 ~/pilot.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
  stop :  bash scripts/lambda_launch.sh terminate $ID
SAVE_MODE=git : box commits runs/lora and pushes to GitHub (if a token was
                provisioned), then auto-terminates ONLY after a confirmed push.
SAVE_MODE=none: box leaves results in runs/lora and does NOT auto-terminate.
No token / push failed -> box stays up; pull results with:
  scp -r ubuntu@$IP:microbe-foundation/runs/lora ./runs/lora_from_box
  then: bash scripts/lambda_launch.sh terminate $ID   # stop billing when done
The ${WATCHDOG_HOURS}h watchdog is the hard cost backstop either way.
============================================================
EOF
