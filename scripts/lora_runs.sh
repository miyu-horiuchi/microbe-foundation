#!/usr/bin/env bash
# lora_runs.sh -- the decisive encoder-adaptation experiment on a GPU box.
#
# Runs frozen vs LoRA (optionally full) ESM-2 fine-tuning on the family split for
# N seeds, writes per-run metric JSONs, and builds Table 31 (frozen vs LoRA vs
# full). Mirrors scripts/tier1_runs.sh conventions (env-var knobs, runs/ outputs,
# optional S3 sync). GPU box only -- backprop through ESM-2 needs real VRAM.
#
# Prereqs on the box (see docs/RUNNING_LORA.md):
#   1. pip install -r requirements.txt   (pulls peft + accelerate)
#   2. raw protein sequences synced to $PROTEINS as <bacdive_id>.txt.gz
#      (from the Modal volume `microbe-esm2-perprotein` proteins/ dir)
#   3. data/traits.parquet + data/splits.parquet present (already in repo)
#
# Usage:
#   bash scripts/lora_runs.sh                 # frozen + lora, 3 seeds, 150M
#   MODES="frozen lora full" SEEDS="0" bash scripts/lora_runs.sh
#   MODEL=facebook/esm2_t33_650M_UR50D bash scripts/lora_runs.sh   # 650M
set -euo pipefail

# Cap CUDA allocator fragmentation. A full-backprop LoRA run pushes the 150M
# encoder hard; expandable segments let freed blocks be reused instead of
# stranded, which the OOM message itself recommends. Harmless for frozen.
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

PROTEINS="${PROTEINS:-data/esm2_proteins}"
MODEL="${MODEL:-facebook/esm2_t30_150M_UR50D}"
POOLING="${POOLING:-set_transformer}"
SPLIT="${SPLIT:-family}"
MODES="${MODES:-frozen lora}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-4}"
MAX_PROTEINS="${MAX_PROTEINS:-256}"
# Proteins per ESM-2 forward. Caps peak encoder activation memory WITHOUT
# changing the science (identical grads/results) -- the LoRA full-backprop path
# OOM'd on an 80GB H100 at the old default of 128. 8 fits 80GB (and 40GB).
ENC_MICROBATCH="${ENC_MICROBATCH:-8}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-5e-4}"
EXTRA="${EXTRA:---grad-checkpoint --balanced-families --class-weights}"
OUTDIR="${OUTDIR:-runs/lora}"
S3_DEST="${S3_DEST:-}"   # e.g. s3://microbe-foundation-esm2-perprotein/lora_results/

# --------------------------------------------------------------------------- #
# Multi-GPU DATA PARALLELISM (DDP) for ONE job -- shard each batch across N GPUs
# with torchrun (one process per GPU). This is the lever that makes the slow
# LoRA arm ~Nx faster: finetune_lora.py auto-detects torchrun's RANK/WORLD_SIZE,
# wraps the model in DistributedDataParallel, shards genomes with a
# DistributedSampler, and all-reduces the loss + all-gathers the val/test metrics
# so the reported numbers equal the single-GPU result within fp noise.
#
# NPROC (a.k.a. GPUS_PER_NODE) selects the fan-out for the LoRA/full arms:
#   unset / "" / 1  single-process plain `python3` (EXISTING behaviour, unchanged)
#   auto            all visible GPUs (nvidia-smi -L), else 1
#   <N>             exactly N processes/GPUs
# The FROZEN arm intentionally stays SINGLE-GPU: it already runs the no_grad fast
# path (~2x faster, no encoder backward) and is not the wall-clock bottleneck, so
# DDP would add sync overhead for ~no benefit. Only frozen is exempt; lora/full
# shard. Frozen and a DDP lora arm run sequentially here, each using the GPUs it
# needs (frozen: 1; lora: all NPROC).
NPROC="${NPROC:-${GPUS_PER_NODE:-}}"
_detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n; n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; echo "${n:-1}"
  else echo 1; fi
}
if [[ "$NPROC" == "auto" ]]; then NPROC="$(_detect_gpus)"; fi
NPROC="${NPROC:-1}"; [[ "$NPROC" =~ ^[0-9]+$ ]] || NPROC=1

mkdir -p "$OUTDIR"
echo "model=$MODEL split=$SPLIT pooling=$POOLING modes=[$MODES] seeds=[$SEEDS]"
echo "proteins=$PROTEINS epochs=$EPOCHS batch=$BATCH max_proteins=$MAX_PROTEINS enc_microbatch=$ENC_MICROBATCH lora_r=$LORA_R"
if [[ "$NPROC" -gt 1 ]]; then
  echo "DDP: lora/full arms shard each batch across nproc_per_node=$NPROC GPUs (torchrun); frozen stays single-GPU"
fi

# Incremental durable upload: on a long multi-hour run a watchdog/crash can kill
# the box mid-experiment. Upload each mode's JSON to the Modal volume the instant
# it's written so a completed `frozen` result is never lost waiting for `lora`.
# ONLY runs under the legacy Modal path: gated on SAVE_MODE=modal (the default
# SAVE_MODE=git/none never touches Modal here), plus INCREMENTAL_MODAL=1 and a
# REMOTE_DIR target. Best-effort.
incremental_upload() {
  [[ "${SAVE_MODE:-git}" == "modal" ]] || return 0
  [[ "${INCREMENTAL_MODAL:-1}" == "1" && -n "${REMOTE_DIR:-}" ]] || return 0
  python3 -m modal volume put microbe-esm2-perprotein "$OUTDIR" "$REMOTE_DIR" --force \
    >/dev/null 2>&1 && echo "  [modal] synced $OUTDIR -> $REMOTE_DIR" \
    || echo "  [modal] WARNING: incremental upload failed (continuing)"
}

for mode in $MODES; do
  for seed in $SEEDS; do
    out="$OUTDIR/${mode}_${SPLIT}_s${seed}.json"
    if [[ -f "$out" ]]; then echo "[skip] $out exists"; continue; fi
    # DDP only for the trainable-encoder arms; frozen stays single-process.
    if [[ "$mode" != "frozen" && "$NPROC" -gt 1 ]]; then
      # Static single-node rendezvous pinned to IPv4 loopback. Avoids torchrun's
      # c10d/--standalone FQDN lookup (which can stall on hosts whose hostname
      # doesn't resolve) and needs no network. A fresh high random port per job
      # keeps sequential jobs from colliding on a lingering socket.
      DDP_PORT="$(( 20000 + RANDOM % 20000 ))"
      LAUNCH=(torchrun --nnodes=1 --nproc_per_node="$NPROC"
              --master_addr=127.0.0.1 --master_port="$DDP_PORT")
      echo "=== mode=$mode seed=$seed -> $out  [DDP nproc_per_node=$NPROC port=$DDP_PORT] ==="
    else
      LAUNCH=(python3)
      echo "=== mode=$mode seed=$seed -> $out ==="
    fi
    "${LAUNCH[@]}" finetune_lora.py \
      --proteins-dir "$PROTEINS" --model-name "$MODEL" --mode "$mode" \
      --split-level "$SPLIT" --pooling "$POOLING" \
      --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
      --max-proteins "$MAX_PROTEINS" --batch "$BATCH" --epochs "$EPOCHS" \
      --enc-microbatch "$ENC_MICROBATCH" \
      --lr "$LR" --seed "$seed" --run-name "${mode}_${SPLIT}_s${seed}" \
      --save-metrics "$out" $EXTRA
    incremental_upload
  done
done

echo "=== building Table 31 ==="
python3 paper/lora_finetune_compare.py --runs-dir "$OUTDIR" || true

if [[ -n "$S3_DEST" ]]; then
  echo "=== syncing $OUTDIR -> $S3_DEST ==="
  aws s3 sync "$OUTDIR" "$S3_DEST"
fi
echo "done."
