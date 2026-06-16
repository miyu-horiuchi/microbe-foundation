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
# Knobs (env): GPU_KIND, SSH_KEY_NAME, REGION, BRANCH, plus any run_full.sh knob
# (EPOCHS, MAX_GENOMES, MODEL, MAX_PROTEINS) forwarded to the box.
set -euo pipefail

API="https://cloud.lambdalabs.com/api/v1"
GPU_KIND="${GPU_KIND:-gpu_1x_a100_sxm4}"
SSH_KEY_NAME="${SSH_KEY_NAME:-miyu-microbe-mac}"
REGION="${REGION:-}"
NAME="${NAME:-microbe-lora-full}"
BRANCH="${BRANCH:-feat/set-transformer-tier1}"
REPO_URL="${REPO_URL:-https://github.com/miyu-horiuchi/microbe-foundation}"
MODAL_TOML="${MODAL_TOML:-$HOME/.modal.toml}"
# Optional run knobs forwarded to run_full.sh on the box (empty = its defaults).
FWD=""
for k in MODEL MAX_GENOMES EPOCHS MAX_PROTEINS MODES SEEDS; do
  v="${!k:-}"; [ -n "$v" ] && FWD="$FWD $k=$v"
done

: "${LAMBDA_API_KEY:?export LAMBDA_API_KEY=secret_...}"
[ -f "$MODAL_TOML" ] || { echo "missing $MODAL_TOML (Modal creds) -- run 'modal token new'" >&2; exit 1; }
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
echo "copying Modal creds..."
scp "${SSH_OPTS[@]}" "$MODAL_TOML" ubuntu@"$IP":~/

echo "cloning repo @ $BRANCH..."
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "rm -rf microbe-foundation && git clone $REPO_URL && cd microbe-foundation && git checkout $BRANCH"

echo "writing ~/run_env.sh (auto-terminate creds)..."
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "echo export INSTANCE_ID=$ID > ~/run_env.sh; echo export LAMBDA_API_KEY=$LAMBDA_API_KEY >> ~/run_env.sh; cat ~/run_env.sh"

echo "starting run in tmux (knobs:${FWD:- defaults})..."
PREFIX=""; [ -n "$FWD" ] && PREFIX="export$FWD; "
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "cd microbe-foundation && tmux new-session -d -s full '${PREFIX}bash scripts/run_full.sh 2>&1 | tee ~/pilot.log'"

cat <<EOF

============================================================
RUNNING.  instance=$ID  ip=$IP
  watch:  ssh ubuntu@$IP 'tail -20 ~/pilot.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
  stop :  bash scripts/lambda_launch.sh terminate $ID
The box auto-terminates itself when the run finishes + uploads to Modal.
Results land in Modal volume microbe-esm2-perprotein:/lora_full
============================================================
EOF
