#!/usr/bin/env bash
#
# lambda_launch.sh — provision a Lambda Cloud GPU box for the tier-1 runs.
#
# Run this from a machine WITH network access (your laptop), not from a
# sandboxed agent. It picks a region that currently has capacity for the
# requested GPU type, launches an instance, waits for it to boot, and prints a
# ready-to-paste ssh command + bootstrap steps.
#
# Auth: export your Lambda Cloud API key first (never commit it):
#     export LAMBDA_API_KEY=secret_...
#
# Usage:
#     export LAMBDA_API_KEY=secret_...
#     # list GPU types + live capacity + price:
#     bash scripts/lambda_launch.sh list
#     # launch (defaults: 1x A100, first SSH key on your account):
#     GPU_KIND=gpu_1x_a100 bash scripts/lambda_launch.sh launch
#     # bigger box for per-protein extraction:
#     GPU_KIND=gpu_8x_h100_sxm bash scripts/lambda_launch.sh launch
#     # tear down when done:
#     bash scripts/lambda_launch.sh terminate <instance-id>
#
# Env knobs:
#     GPU_KIND       Lambda instance_type_name (default gpu_1x_a100)
#     SSH_KEY_NAME   SSH key registered in Lambda (default: first on account)
#     REGION         force a region (default: first with capacity)
#     NAME           instance name tag (default: microbe-tier1)
#
set -euo pipefail

API="https://cloud.lambdalabs.com/api/v1"
GPU_KIND="${GPU_KIND:-gpu_1x_a100}"
SSH_KEY_NAME="${SSH_KEY_NAME:-}"
REGION="${REGION:-}"
NAME="${NAME:-microbe-tier1}"

: "${LAMBDA_API_KEY:?Set LAMBDA_API_KEY (export LAMBDA_API_KEY=secret_...)}"

# Prefer the system python so we don't trip the Anaconda numpy hang. JSON only,
# so any python3 works; this just avoids a conda interpreter.
PY="$(command -v /usr/local/bin/python3 || command -v python3)"

_api() {  # _api METHOD PATH [json-body]
    local method="$1" path="$2" body="${3:-}"
    if [ -n "$body" ]; then
        curl -fsS -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
            -X "$method" "$API$path" -d "$body"
    else
        curl -fsS -u "$LAMBDA_API_KEY:" -X "$method" "$API$path"
    fi
}

cmd_list() {
    echo "GPU types with live capacity (name | $/hr | regions):"
    _api GET /instance-types | "$PY" -c '
import sys, json
d = json.load(sys.stdin)["data"]
rows = []
for name, it in d.items():
    price = it["instance_type"]["price_cents_per_hour"] / 100.0
    regs = [r["name"] for r in it["regions_with_capacity_available"]]
    rows.append((name, price, regs))
for name, price, regs in sorted(rows, key=lambda r: r[1]):
    flag = ",".join(regs) if regs else "(no capacity)"
    print(f"  {name:24s} ${price:5.2f}/hr  {flag}")
'
}

_first_ssh_key() {
    _api GET /ssh-keys | "$PY" -c '
import sys, json
d = json.load(sys.stdin)["data"]
print(d[0]["name"] if d else "")
'
}

_pick_region() {  # echo a region with capacity for $GPU_KIND (or honor $REGION)
    _api GET /instance-types | "$PY" -c '
import sys, json, os
d = json.load(sys.stdin)["data"]
kind = os.environ["GPU_KIND"]
forced = os.environ.get("REGION", "")
it = d.get(kind)
if not it:
    print("ERR:unknown-gpu-kind"); sys.exit()
regs = [r["name"] for r in it["regions_with_capacity_available"]]
if forced:
    print(forced if forced in regs else "ERR:no-capacity-in-region")
else:
    print(regs[0] if regs else "ERR:no-capacity")
'
}

cmd_launch() {
    if [ -z "$SSH_KEY_NAME" ]; then
        SSH_KEY_NAME="$(_first_ssh_key)"
        [ -z "$SSH_KEY_NAME" ] && {
            echo "ERROR: no SSH key on your Lambda account. Add one in the dashboard" >&2
            echo "       (Cloud > SSH keys) or set SSH_KEY_NAME=..." >&2
            exit 1
        }
    fi
    local region; region="$(_pick_region)"
    case "$region" in
        ERR:unknown-gpu-kind) echo "ERROR: unknown GPU_KIND='$GPU_KIND'. Run 'list' to see options." >&2; exit 1 ;;
        ERR:no-capacity)      echo "ERROR: no live capacity for '$GPU_KIND'. Run 'list' or try another GPU_KIND." >&2; exit 1 ;;
        ERR:no-capacity-in-region) echo "ERROR: '$GPU_KIND' not available in REGION='$REGION'." >&2; exit 1 ;;
    esac

    echo "Launching $GPU_KIND in $region (ssh key: $SSH_KEY_NAME)..."
    local body
    body="$(REGION_PICK="$region" SSH_KEY_NAME="$SSH_KEY_NAME" GPU_KIND="$GPU_KIND" NAME="$NAME" "$PY" -c '
import json, os
print(json.dumps({
    "region_name": os.environ["REGION_PICK"],
    "instance_type_name": os.environ["GPU_KIND"],
    "ssh_key_names": [os.environ["SSH_KEY_NAME"]],
    "name": os.environ["NAME"],
}))' )"
    local id
    id="$(_api POST /instance-operations/launch "$body" \
          | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"]["instance_ids"][0])')"
    echo "Instance launching: $id   (polling for boot...)"

    local ip="" status="" i=0
    while [ "$i" -lt 60 ]; do
        sleep 10; i=$((i + 1))
        local info; info="$(_api GET "/instances/$id")"
        status="$(printf '%s' "$info" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"]["status"])')"
        ip="$(printf '%s' "$info" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"].get("ip") or "")')"
        echo "  [$((i*10))s] status=$status ip=${ip:-<pending>}"
        [ "$status" = "active" ] && [ -n "$ip" ] && break
    done
    if [ "$status" != "active" ] || [ -z "$ip" ]; then
        echo "WARNING: instance not active yet. Check the dashboard / re-poll:" >&2
        echo "  bash scripts/lambda_launch.sh status $id" >&2
        exit 0
    fi

    cat <<EOF

============================================================
READY. instance=$id  ip=$ip
============================================================
  ssh ubuntu@$ip

  # On the box:
  git clone https://github.com/miyu-horiuchi/microbe-foundation
  cd microbe-foundation
  bash scripts/lambda_install.sh
  # then follow docs/RUNNING_PERPROTEIN_ON_LAMBDA.md (extraction)
  # or, if embeddings are present: bash scripts/tier1_runs.sh

  # Tear down when finished (stops billing):
  bash scripts/lambda_launch.sh terminate $id
============================================================
EOF
}

cmd_status() {  # status <id>
    _api GET "/instances/$1" | "$PY" -c '
import sys, json
d = json.load(sys.stdin)["data"]
iid = d["id"]; st = d["status"]; ip = d.get("ip")
ty = d["instance_type"]["name"]; rg = d["region"]["name"]
print(f"id={iid} status={st} ip={ip} type={ty} region={rg}")
'
}

cmd_terminate() {  # terminate <id>
    local body; body="$("$PY" -c 'import json,sys; print(json.dumps({"instance_ids":[sys.argv[1]]}))' "$1")"
    _api POST /instance-operations/terminate "$body" | "$PY" -c '
import sys, json
d = json.load(sys.stdin)["data"]["terminated_instances"]
print("terminated:", ", ".join(i["id"] for i in d) or "(none)")
'
}

case "${1:-list}" in
    list)       cmd_list ;;
    launch)     cmd_launch ;;
    status)     cmd_status "${2:?usage: status <instance-id>}" ;;
    terminate)  cmd_terminate "${2:?usage: terminate <instance-id>}" ;;
    *) echo "usage: $0 {list|launch|status <id>|terminate <id>}" >&2; exit 1 ;;
esac
