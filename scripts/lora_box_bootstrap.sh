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

echo "=== [1/4] base install ==="
bash scripts/lambda_install.sh || true
pip install -r requirements.txt
pip install -q modal

echo "=== [2/4] sync raw protein sequences from Modal volume ==="
mkdir -p data/esm2_proteins
if [ -z "$(ls -A data/esm2_proteins 2>/dev/null)" ]; then
  modal volume get microbe-esm2-perprotein proteins data/esm2_proteins/
else
  echo "  data/esm2_proteins already populated, skipping."
fi
echo "  genomes with sequences: $(find data/esm2_proteins -name '*.txt.gz' | wc -l)"

echo "=== [3/4] run scoped LoRA pilot ==="
MODES="${MODES:-frozen lora}" SEEDS="${SEEDS:-0}" EPOCHS="${EPOCHS:-5}" \
  MAX_PROTEINS="${MAX_PROTEINS:-128}" PROTEINS="${PROTEINS:-data/esm2_proteins}" \
  EXTRA="${EXTRA:---grad-checkpoint --balanced-families --class-weights --max-genomes 15000}" \
  bash scripts/lora_runs.sh

echo "=== [4/4] DONE ==="
echo "results:"
ls -la runs/lora/*.json 2>/dev/null || echo "  (no JSONs -- check the log above for errors)"
echo
echo "Now, from your LAPTOP, pull the JSONs back and push to GitHub:"
echo "  mkdir -p runs/lora && scp \"ubuntu@\$IP:~/microbe-foundation/runs/lora/*.json\" runs/lora/"
echo "  git add runs/lora/*.json && git commit -m 'pilot LoRA results' && git push"
