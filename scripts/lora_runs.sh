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

PROTEINS="${PROTEINS:-data/esm2_proteins}"
MODEL="${MODEL:-facebook/esm2_t30_150M_UR50D}"
POOLING="${POOLING:-set_transformer}"
SPLIT="${SPLIT:-family}"
MODES="${MODES:-frozen lora}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-4}"
MAX_PROTEINS="${MAX_PROTEINS:-256}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-5e-4}"
EXTRA="${EXTRA:---grad-checkpoint --balanced-families --class-weights}"
OUTDIR="${OUTDIR:-runs/lora}"
S3_DEST="${S3_DEST:-}"   # e.g. s3://microbe-foundation-esm2-perprotein/lora_results/

mkdir -p "$OUTDIR"
echo "model=$MODEL split=$SPLIT pooling=$POOLING modes=[$MODES] seeds=[$SEEDS]"
echo "proteins=$PROTEINS epochs=$EPOCHS batch=$BATCH max_proteins=$MAX_PROTEINS lora_r=$LORA_R"

# Incremental durable upload: on a long multi-hour run a watchdog/crash can kill
# the box mid-experiment. Upload each mode's JSON to the Modal volume the instant
# it's written so a completed `frozen` result is never lost waiting for `lora`.
# Best-effort + gated on INCREMENTAL_MODAL=1 and a REMOTE_DIR target.
incremental_upload() {
  [[ "${INCREMENTAL_MODAL:-0}" == "1" && -n "${REMOTE_DIR:-}" ]] || return 0
  python3 -m modal volume put microbe-esm2-perprotein "$OUTDIR" "$REMOTE_DIR" --force \
    >/dev/null 2>&1 && echo "  [modal] synced $OUTDIR -> $REMOTE_DIR" \
    || echo "  [modal] WARNING: incremental upload failed (continuing)"
}

for mode in $MODES; do
  for seed in $SEEDS; do
    out="$OUTDIR/${mode}_${SPLIT}_s${seed}.json"
    if [[ -f "$out" ]]; then echo "[skip] $out exists"; continue; fi
    echo "=== mode=$mode seed=$seed -> $out ==="
    python3 finetune_lora.py \
      --proteins-dir "$PROTEINS" --model-name "$MODEL" --mode "$mode" \
      --split-level "$SPLIT" --pooling "$POOLING" \
      --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
      --max-proteins "$MAX_PROTEINS" --batch "$BATCH" --epochs "$EPOCHS" \
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
