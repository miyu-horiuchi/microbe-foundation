#!/usr/bin/env bash
# lora_box_bootstrap.sh -- run ON the Lambda GPU box after cloning the repo.
#
# Installs deps, pulls raw protein sequences from the Modal volume, and runs the
# scoped LoRA pilot (frozen vs lora, family split, 1 seed). Leaves result JSONs in
# runs/lora/ -- scp them back to your laptop and push from there (the box has no
# GitHub push creds). Single entrypoint so the launching ssh command is one line
# with no heredoc.
#
# Prereq: ~/.modal.toml must already be on the box (scp it before running) so
# `modal volume get` is authenticated.
#
# Override the pilot via env, e.g.:
#   MODES="frozen lora full" SEEDS="0 1 2" EPOCHS=15 MAX_PROTEINS=256 \
#     bash scripts/lora_box_bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"   # pip --user installs land here (modal CLI)

echo "=== [1/4] base install ==="
bash scripts/lambda_install.sh || true
pip install -r requirements.txt
pip install -q modal
# The box's system torch is built against NumPy 1.x; NumPy 2 breaks torch<->numpy
# interop ("_ARRAY_API not found"). Pin <2 so tensor/array conversions work.
pip install -q "numpy<2"
python3 -c "import torch, numpy as np; torch.from_numpy(np.zeros(2)); print('torch/numpy OK', np.__version__)"

echo "=== [2/4] sync raw protein sequences from Modal volume ==="
mkdir -p data/esm2_proteins
if [ -z "$(ls -A data/esm2_proteins 2>/dev/null)" ]; then
  python3 -m modal volume get microbe-esm2-perprotein proteins data/esm2_proteins/
else
  echo "  data/esm2_proteins already populated, skipping."
fi
echo "  genomes with sequences: $(find data/esm2_proteins -name '*.txt.gz' | wc -l)"

echo "=== [3/4] run scoped LoRA pilot ==="
MODES="${MODES:-frozen lora}" SEEDS="${SEEDS:-0}" EPOCHS="${EPOCHS:-5}" \
  MAX_PROTEINS="${MAX_PROTEINS:-128}" PROTEINS="${PROTEINS:-data/esm2_proteins}" \
  EXTRA="${EXTRA:---grad-checkpoint --balanced-families --class-weights --max-genomes 15000}" \
  bash scripts/lora_runs.sh

echo "=== [4/5] save results to durable storage (Modal volume) ==="
SAVE_OK=0
if ls runs/lora/*.json >/dev/null 2>&1; then
  cp -f "${PILOT_LOG:-$HOME/pilot.log}" runs/lora/pilot.log 2>/dev/null || true
  # Table 31 (best-effort) so the artifact travels with the JSONs.
  python3 paper/lora_finetune_compare.py --runs-dir runs/lora || true
  cp -f paper/tables/31_lora_finetune.md runs/lora/ 2>/dev/null || true
  if python3 -m modal volume put microbe-esm2-perprotein runs/lora "${REMOTE_DIR:-lora_pilot}" --force; then
    SAVE_OK=1
    echo "  saved runs/lora -> modal volume microbe-esm2-perprotein:/${REMOTE_DIR:-lora_pilot}"
  else
    echo "  WARNING: modal volume put failed -- NOT terminating so you can recover."
  fi
else
  echo "  no result JSONs -- NOT terminating; inspect the log above."
fi

echo "=== [5/5] auto-terminate ==="
if [ "${AUTO_TERMINATE:-0}" = "1" ] && [ "$SAVE_OK" = "1" ] \
   && [ -n "${LAMBDA_API_KEY:-}" ] && [ -n "${INSTANCE_ID:-}" ]; then
  echo "  results saved; terminating instance $INSTANCE_ID to stop billing..."
  curl -fsS -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
    -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" && echo "  terminate request sent." \
    || echo "  WARNING: terminate API call failed -- terminate manually."
else
  echo "  auto-terminate skipped (AUTO_TERMINATE=${AUTO_TERMINATE:-0}, save_ok=$SAVE_OK)."
fi

echo
echo "=== DONE. To pull results onto your laptop (then the agent reads them): ==="
echo "  modal volume get microbe-esm2-perprotein ${REMOTE_DIR:-lora_pilot} ./runs/lora_pilot --force"
