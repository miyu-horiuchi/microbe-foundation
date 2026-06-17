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

# Cap CUDA allocator fragmentation for every training process spawned downstream
# (bootstrap -> runner -> finetune_lora.py). The OOM traceback explicitly
# recommends this; it is harmless for the frozen job.
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

# Full-scale defaults (all overridable via env).
export MODEL="${MODEL:-facebook/esm2_t30_150M_UR50D}"
export MAX_GENOMES="${MAX_GENOMES:-15000}"
export EPOCHS="${EPOCHS:-5}"
export MAX_PROTEINS="${MAX_PROTEINS:-128}"
# Proteins per ESM-2 forward. Caps peak encoder activation memory with NO change
# to the science (identical grads/results) -- the LoRA full-backprop path OOM'd
# on an 80GB H100 at the old default of 128. 8 fits 80GB (and 40GB). Overridable.
export ENC_MICROBATCH="${ENC_MICROBATCH:-8}"
export MODES="${MODES:-frozen lora}"
export SEEDS="${SEEDS:-0}"
export AUTO_TERMINATE="${AUTO_TERMINATE:-1}"
# Durable save sink (NO Modal by default). git = commit+push runs/lora to the
# branch (auto-terminate only fires after a confirmed push); none = leave results
# on the box and never auto-terminate (pull via scp); modal = legacy volume put.
# INCREMENTAL_MODAL only matters when SAVE_MODE=modal.
export SAVE_MODE="${SAVE_MODE:-git}"
export INCREMENTAL_MODAL="${INCREMENTAL_MODAL:-1}"
export REMOTE_DIR="${REMOTE_DIR:-lora_full}"
# Across-GPU parallelism (Option A): set PARALLEL=1 to fan the (mode, seed) jobs
# out one-per-GPU on a multi-GPU box. Bootstrap also auto-enables it when >1 GPU
# is detected. NUM_GPUS overrides the auto-detected fan-out width. Both passed
# through to the box; unset/empty = single-GPU sequential default.
export PARALLEL="${PARALLEL:-}"
export NUM_GPUS="${NUM_GPUS:-}"
# Multi-GPU DATA PARALLELISM (DDP): shard EACH batch of the slow LoRA arm across
# N GPUs via torchrun -> ~Nx faster. NPROC (alias GPUS_PER_NODE) = processes/GPUs
# per job: ""/1 = single-GPU (default, unchanged), "auto" = all visible GPUs,
# "<N>" = N GPUs. Takes precedence over the across-GPU parallel runner; when NPROC>1
# EVERY arm (frozen included) shards across all N GPUs so none sits idle. Forwarded
# to the box.
export NPROC="${NPROC:-${GPUS_PER_NODE:-}}"

echo "=== run_full.sh ==="
echo "  model=$MODEL genomes=$MAX_GENOMES epochs=$EPOCHS max_proteins=$MAX_PROTEINS enc_microbatch=$ENC_MICROBATCH"
echo "  modes=[$MODES] seeds=[$SEEDS] auto_terminate=$AUTO_TERMINATE save_mode=$SAVE_MODE remote_dir=$REMOTE_DIR"
echo "  parallel=${PARALLEL:-<off>} num_gpus=${NUM_GPUS:-<auto>} nproc(ddp)=${NPROC:-<off>}"
echo "  instance_id=${INSTANCE_ID:-<unset>} lambda_key=$([ -n "${LAMBDA_API_KEY:-}" ] && echo set || echo UNSET)"

exec bash scripts/lora_box_bootstrap.sh
