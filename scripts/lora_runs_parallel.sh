#!/usr/bin/env bash
# lora_runs_parallel.sh -- across-GPU parallel sibling of lora_runs.sh.
#
# Drop-in replacement that runs the independent (mode, seed) jobs SIMULTANEOUSLY,
# one job per GPU, on a multi-GPU box. Every job is a self-contained
# finetune_lora.py invocation (Option A: launcher-only parallelism -- NO DDP, no
# DistributedSampler, zero changes to model/training code). Each job is pinned to
# its own GPU via CUDA_VISIBLE_DEVICES so they don't contend for VRAM.
#
# Same knobs/semantics/outputs as lora_runs.sh (PROTEINS, MODEL, POOLING, SPLIT,
# MODES, SEEDS, EPOCHS, BATCH, MAX_PROTEINS, LORA_R, LORA_ALPHA, LR, EXTRA,
# OUTDIR, S3_DEST, plus INCREMENTAL_MODAL/REMOTE_DIR for durable banking). Writes
# the identical runs/lora/<mode>_<split>_s<seed>.json plus a per-job log under
# runs/lora/logs/. Resumable ([skip] when the JSON already exists) and robust (a
# single mode/seed crash never aborts the others -- per-job exit codes are
# reported at the end).
#
# Extra knob over lora_runs.sh:
#   NUM_GPUS   concurrency cap / number of GPUs to fan out across. Auto-detected
#              from `nvidia-smi -L` when unset; falls back to 1.
#
# Usage (on a multi-GPU box):
#   PARALLEL is wired in run_full.sh; to invoke directly:
#   MODES="frozen lora" SEEDS="0 1 2" bash scripts/lora_runs_parallel.sh
set -euo pipefail

# Cap CUDA allocator fragmentation (see lora_runs.sh). Set once here; it is
# inherited by every per-GPU job below. Harmless for frozen.
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
# Proteins per ESM-2 forward -- caps peak encoder activation memory without
# changing results (see lora_runs.sh). 8 keeps LoRA inside an 80GB/40GB GPU.
ENC_MICROBATCH="${ENC_MICROBATCH:-8}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-5e-4}"
EXTRA="${EXTRA:---grad-checkpoint --balanced-families --class-weights}"
OUTDIR="${OUTDIR:-runs/lora}"
S3_DEST="${S3_DEST:-}"   # e.g. s3://microbe-foundation-esm2-perprotein/lora_results/
LOGDIR="$OUTDIR/logs"

# GPU fan-out width. Honour an explicit NUM_GPUS, else count the box's GPUs, else
# assume 1 (e.g. the CPU mock matrix). Pinning to a non-existent device is a no-op
# on CPU, so this is safe to over- or under-shoot in a smoke.
if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  NUM_GPUS="${NUM_GPUS:-1}"
fi
[[ "$NUM_GPUS" -ge 1 ]] 2>/dev/null || NUM_GPUS=1

mkdir -p "$OUTDIR" "$LOGDIR"
echo "model=$MODEL split=$SPLIT pooling=$POOLING modes=[$MODES] seeds=[$SEEDS]"
echo "proteins=$PROTEINS epochs=$EPOCHS batch=$BATCH max_proteins=$MAX_PROTEINS enc_microbatch=$ENC_MICROBATCH lora_r=$LORA_R"
echo "PARALLEL across num_gpus=$NUM_GPUS (one job per GPU, logs -> $LOGDIR)"

# Incremental durable upload -- identical to lora_runs.sh. On a long multi-hour
# run a watchdog/crash can kill the box mid-experiment; upload the whole OUTDIR to
# the Modal volume the instant a job's JSON lands so a completed result is never
# lost. Best-effort + gated on INCREMENTAL_MODAL=1 and a REMOTE_DIR target. Called
# from inside each background job so it fires when (and only when) that job exits.
incremental_upload() {
  [[ "${INCREMENTAL_MODAL:-0}" == "1" && -n "${REMOTE_DIR:-}" ]] || return 0
  python3 -m modal volume put microbe-esm2-perprotein "$OUTDIR" "$REMOTE_DIR" --force \
    >/dev/null 2>&1 && echo "  [modal] synced $OUTDIR -> $REMOTE_DIR" \
    || echo "  [modal] WARNING: incremental upload failed (continuing)"
}

# One job = one (mode, seed) finetune_lora.py invocation pinned to one GPU. Runs
# backgrounded; never aborts the batch (captures its own exit code into a .exit
# sentinel and banks its JSON to Modal on success). Stdout/stderr -> per-job log.
run_job() {
  local mode="$1" seed="$2" gpu="$3"
  local out="$OUTDIR/${mode}_${SPLIT}_s${seed}.json"
  local log="$LOGDIR/${mode}_${SPLIT}_s${seed}.log"
  local rc=0
  CUDA_VISIBLE_DEVICES="$gpu" python3 finetune_lora.py \
    --proteins-dir "$PROTEINS" --model-name "$MODEL" --mode "$mode" \
    --split-level "$SPLIT" --pooling "$POOLING" \
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
    --max-proteins "$MAX_PROTEINS" --batch "$BATCH" --epochs "$EPOCHS" \
    --enc-microbatch "$ENC_MICROBATCH" \
    --lr "$LR" --seed "$seed" --run-name "${mode}_${SPLIT}_s${seed}" \
    --save-metrics "$out" $EXTRA >"$log" 2>&1 || rc=$?
  if [[ "$rc" == "0" ]]; then
    echo "  [done] mode=$mode seed=$seed gpu=$gpu -> $out"
    incremental_upload          # bank this job's result the instant it finishes
  else
    echo "  [FAIL] mode=$mode seed=$seed gpu=$gpu rc=$rc (see $log)"
  fi
  # Sentinel written LAST so the scheduler can treat its presence as "slot free".
  echo "$rc" > "$LOGDIR/${mode}_${SPLIT}_s${seed}.exit"
  return 0
}

# Per-GPU slot bookkeeping (bash 3.2-safe: indexed arrays only, no `wait -n`).
#   slot_pid[g]   pid of the job currently occupying GPU g ("" = free)
#   slot_exit[g]  path to that job's .exit sentinel (appears when it finishes)
slot_pid=()
slot_exit=()
for ((g=0; g<NUM_GPUS; g++)); do slot_pid[$g]=""; slot_exit[$g]=""; done

launched=()   # "mode seed" of every job we actually started (for the final tally)

for mode in $MODES; do
  for seed in $SEEDS; do
    out="$OUTDIR/${mode}_${SPLIT}_s${seed}.json"
    if [[ -f "$out" ]]; then echo "[skip] $out exists"; continue; fi

    # Acquire a free GPU slot, waiting (polling) until one frees up. A slot is
    # free when it has never been used, or its job's .exit sentinel has appeared.
    gpu=""
    while [[ -z "$gpu" ]]; do
      for ((g=0; g<NUM_GPUS; g++)); do
        if [[ -z "${slot_pid[$g]}" ]]; then
          gpu="$g"; break
        elif [[ -f "${slot_exit[$g]}" ]]; then
          wait "${slot_pid[$g]}" 2>/dev/null || true   # reap; never blocks (done)
          slot_pid[$g]=""; gpu="$g"; break
        fi
      done
      [[ -z "$gpu" ]] && sleep 1
    done

    rm -f "$LOGDIR/${mode}_${SPLIT}_s${seed}.exit"   # clear any stale sentinel
    echo "=== launch mode=$mode seed=$seed gpu=$gpu -> $out ==="
    run_job "$mode" "$seed" "$gpu" &
    slot_pid[$g]=$!
    slot_exit[$g]="$LOGDIR/${mode}_${SPLIT}_s${seed}.exit"
    launched+=("$mode $seed")
  done
done

wait   # let every still-running job finish (returns 0 regardless of job status)

echo "=== per-job exit codes ==="
n_fail=0
for job in "${launched[@]:-}"; do
  [[ -n "$job" ]] || continue
  set -- $job; jmode="$1"; jseed="$2"
  ef="$LOGDIR/${jmode}_${SPLIT}_s${jseed}.exit"
  rc="$(cat "$ef" 2>/dev/null || echo '?')"
  echo "  ${jmode}_${SPLIT}_s${jseed}: exit=$rc  (log: $LOGDIR/${jmode}_${SPLIT}_s${jseed}.log)"
  [[ "$rc" == "0" ]] || n_fail=$((n_fail + 1))
done
[[ "$n_fail" -gt 0 ]] && echo "  WARNING: $n_fail job(s) failed -- other results still banked above."

echo "=== building Table 31 ==="
python3 paper/lora_finetune_compare.py --runs-dir "$OUTDIR" || true

if [[ -n "$S3_DEST" ]]; then
  echo "=== syncing $OUTDIR -> $S3_DEST ==="
  aws s3 sync "$OUTDIR" "$S3_DEST"
fi
echo "done."
