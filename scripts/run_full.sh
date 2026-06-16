#!/usr/bin/env bash
# run_full.sh -- ONE command to run the full-scale LoRA encoder-adaptation
# experiment on the GPU box. Exists to kill the fragile-ssh-one-liner problem:
# every knob is fixed here and Lambda creds are read from ~/run_env.sh, so the
# launch line carries NO secrets and NO nested quoting.
#
# On the box, write creds once (each on its own line so no space can be lost):
#   echo 'export INSTANCE_ID=<id>'        >  ~/run_env.sh
#   echo 'export LAMBDA_API_KEY=<key>'    >> ~/run_env.sh
#
# Then launch (this is the ONLY thing you ever paste):
#   tmux new-session -d -s full 'bash scripts/run_full.sh 2>&1 | tee ~/pilot.log'
#
# Override any knob inline if needed, e.g. a quick check:
#   EPOCHS=1 MAX_GENOMES=2000 bash scripts/run_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Lambda creds for auto-terminate (optional -- run still works without them).
[ -f "$HOME/run_env.sh" ] && . "$HOME/run_env.sh"

# Full-scale defaults (all overridable via env).
export MODEL="${MODEL:-facebook/esm2_t30_150M_UR50D}"
export MAX_GENOMES="${MAX_GENOMES:-15000}"
export EPOCHS="${EPOCHS:-5}"
export MAX_PROTEINS="${MAX_PROTEINS:-128}"
export MODES="${MODES:-frozen lora}"
export SEEDS="${SEEDS:-0}"
export AUTO_TERMINATE="${AUTO_TERMINATE:-1}"
export INCREMENTAL_MODAL="${INCREMENTAL_MODAL:-1}"
export REMOTE_DIR="${REMOTE_DIR:-lora_full}"

echo "=== run_full.sh ==="
echo "  model=$MODEL genomes=$MAX_GENOMES epochs=$EPOCHS max_proteins=$MAX_PROTEINS"
echo "  modes=[$MODES] seeds=[$SEEDS] auto_terminate=$AUTO_TERMINATE remote_dir=$REMOTE_DIR"
echo "  instance_id=${INSTANCE_ID:-<unset>} lambda_key=$([ -n "${LAMBDA_API_KEY:-}" ] && echo set || echo UNSET)"

exec bash scripts/lora_box_bootstrap.sh
