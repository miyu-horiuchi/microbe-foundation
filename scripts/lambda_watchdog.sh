#!/usr/bin/env bash
# lambda_watchdog.sh -- hard cost backstop. Force-terminate THIS Lambda instance
# after N hours no matter what, so a hung/failed run can't bill indefinitely.
# Run detached on the box:
#   LAMBDA_API_KEY=secret_... INSTANCE_ID=... nohup bash scripts/lambda_watchdog.sh 6 \
#     >/tmp/watchdog.log 2>&1 &
# If the normal auto-terminate already fired, terminating an absent instance is a
# harmless no-op.
set -euo pipefail
HOURS="${1:-6}"
: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY}"
: "${INSTANCE_ID:?set INSTANCE_ID}"
echo "watchdog armed: will terminate $INSTANCE_ID in ${HOURS}h ($(date))"
sleep "${HOURS}h"
echo "watchdog firing at $(date) -- terminating $INSTANCE_ID"
curl -fsS -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
  -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" && echo " terminate sent."
