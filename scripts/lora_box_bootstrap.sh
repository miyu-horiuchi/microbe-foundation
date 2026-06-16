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

# Cap CUDA allocator fragmentation for the training processes (the OOM message
# itself recommends expandable_segments). Exported here so it reaches the runner
# and every finetune_lora.py child even when the bootstrap is invoked directly.
# Harmless for the frozen job.
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

# --------------------------------------------------------------------------- #
# [1/5] ISOLATED venv install.
#
# Why a venv: the box's previous failure ("Could not find EsmModel neither in
# transformers.models.esm ...") was NOT a code bug. It came from `pip install
# --user` layering transformers on top of the box's system/conda packages in
# /usr/lib (or the conda env), producing a Frankenstein transformers whose
# modeling_esm failed to import -- and AutoModel's lazy loader masked the real
# error. Verified locally: a clean isolated transformers==4.57.x imports
# EsmModel fine. So we install the ML stack into a dedicated venv that SHADOWS
# any system packages, eliminating the collision entirely.
#
# --system-site-packages: reuse the box's already-working CUDA torch (zero risk
# of a CUDA/driver mismatch from a fresh torch wheel) and heavy compiled deps
# (pyrodigal/biopython/pandas). transformers/peft/accelerate/numpy installed
# into the venv take precedence over anything inherited.
# --------------------------------------------------------------------------- #
echo "=== [1/5] isolated venv install ==="
VENV="${VENV:-$HOME/.venv_lora}"
if [ ! -d "$VENV" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python3 -m pip install -q --upgrade pip

# Shadow the box's ANCIENT apt packages with modern venv-local ones. Two have
# bitten us:
#   * Pillow < 9.1 lacks PIL.Image.Resampling, which transformers' image_utils
#     imports on the EsmModel path -> the real cause of the "Could not find
#     EsmModel" / "cannot import name EsmModel" failures. `pip install
#     transformers` does NOT pull Pillow (it's an optional vision dep), so the
#     stale system Pillow stayed exposed under --system-site-packages.
#   * numpy 1.21.x (apt) is old; pin a newer 1.x (NumPy 2 breaks the box torch's
#     NumPy-1.x C-ABI: "_ARRAY_API not found").
python3 -m pip install -q "numpy>=1.23,<2" "Pillow>=10"
# Data stack for model.py/finetune_lora.py (read_parquet needs pyarrow). The box's
# pyarrow lived in ~/.local, which the venv does not see -> install venv-local so
# pandas + pyarrow + numpy are a consistent set. (model.py/finetune_lora.py use
# only numpy/pandas/torch -- no sklearn/scipy/xgboost.)
python3 -m pip install -q "pandas>=2.0" "pyarrow>=15"
# The LoRA stack, isolated in the venv (shadows any system transformers/peft).
python3 -m pip install -q "transformers>=4.40,<4.58" "peft>=0.11,<0.14" \
  "accelerate>=0.30" modal

# --------------------------------------------------------------------------- #
# PREFLIGHT -- fail fast BEFORE syncing data or spending GPU time. The two
# things that actually broke the box: torch<->numpy interop and the masked
# EsmModel import. If either is wrong, abort now (cheap) instead of after the
# data download (expensive).
# --------------------------------------------------------------------------- #
echo "--- preflight: torch/numpy interop + CUDA + EsmModel import ---"
python3 - <<'PYEOF'
import sys
import numpy as np, torch
torch.from_numpy(np.zeros(2))                     # ABI interop
print(f"  torch {torch.__version__}  numpy {np.__version__}  "
      f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("PREFLIGHT FAIL: CUDA not available to torch inside the venv.")
import transformers
print(f"  transformers {transformers.__version__}")
try:
    from transformers import EsmModel  # explicit -> real traceback if broken
except Exception as e:
    raise SystemExit(
        "PREFLIGHT FAIL: EsmModel import broken (env pollution). "
        f"Real error: {e!r}")
print("  EsmModel import OK")
print("PREFLIGHT OK")
PYEOF

echo "=== [2/5] sync raw protein sequences from Modal volume ==="
mkdir -p data/esm2_proteins
if [ -z "$(ls -A data/esm2_proteins 2>/dev/null)" ]; then
  python3 -m modal volume get microbe-esm2-perprotein proteins data/esm2_proteins/
else
  echo "  data/esm2_proteins already populated, skipping."
fi
echo "  genomes with sequences: $(find data/esm2_proteins -name '*.txt.gz' | wc -l)"

echo "=== [3/5] run scoped LoRA pilot ==="
# Knobs (all space-free so they're easy to pass over ssh/tmux):
#   MODEL         ESM-2 checkpoint (smaller = much faster pilot)
#   EPOCHS        epochs per mode
#   MAX_PROTEINS  proteins/genome encoded (memory + speed)
#   MAX_GENOMES   genomes/split cap (the dominant cost lever)
# NOTE: 150M x 15k-genomes x 5 epochs is HOURS/epoch on one A100 -- too heavy to
# finish inside the watchdog window. For a fast frozen-vs-LoRA signal use e.g.
# MODEL=facebook/esm2_t12_35M_UR50D MAX_GENOMES=2000 EPOCHS=3 MAX_PROTEINS=96.
#
# Runner selection: on a multi-GPU box the (mode, seed) jobs are independent, so
# fan them out one-per-GPU with lora_runs_parallel.sh -- ~Nx faster wall-clock for
# N GPUs. Opt in with PARALLEL=1, or auto-detect when the box has >1 GPU. Default
# (single GPU / PARALLEL unset) keeps the sequential lora_runs.sh behaviour.
GPU_COUNT=1
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  GPU_COUNT="${GPU_COUNT:-1}"
fi
RUNNER="scripts/lora_runs.sh"
if [ "${PARALLEL:-0}" = "1" ] || [ "${GPU_COUNT:-1}" -gt 1 ] 2>/dev/null; then
  RUNNER="scripts/lora_runs_parallel.sh"
  echo "  parallel runner selected (PARALLEL=${PARALLEL:-0}, gpus=$GPU_COUNT)"
fi
MODES="${MODES:-frozen lora}" SEEDS="${SEEDS:-0}" EPOCHS="${EPOCHS:-5}" \
  MODEL="${MODEL:-facebook/esm2_t30_150M_UR50D}" \
  MAX_PROTEINS="${MAX_PROTEINS:-128}" PROTEINS="${PROTEINS:-data/esm2_proteins}" \
  ENC_MICROBATCH="${ENC_MICROBATCH:-8}" \
  EXTRA="${EXTRA:---grad-checkpoint --balanced-families --class-weights --max-genomes ${MAX_GENOMES:-15000}}" \
  SAVE_MODE="${SAVE_MODE:-git}" \
  INCREMENTAL_MODAL="${INCREMENTAL_MODAL:-1}" REMOTE_DIR="${REMOTE_DIR:-lora_pilot}" \
  NUM_GPUS="${NUM_GPUS:-}" \
  bash "$RUNNER"

# --------------------------------------------------------------------------- #
# [4/5] SAVE results durably -- NO Modal required.
#
# SAVE_MODE selects the durable sink (default: git):
#   git    commit runs/lora (JSONs + logs + Table 31) to the current branch and
#          `git push` to GitHub. Tiny files, lands regardless of the box dying.
#          Requires a push credential on the box (GH_TOKEN, provisioned by
#          lambda_full_launch.sh). If none is present OR the push fails, the
#          results stay committed locally and we DO NOT terminate -- pull via scp.
#   none   write results to runs/lora only; never push, never terminate. Pull via
#          scp from your laptop (instructions printed below). Watchdog caps cost.
#   modal  legacy: `modal volume put` to microbe-esm2-perprotein (needs a usable
#          Modal subscription). Kept for back-compat; no longer the default.
#
# save_results() returns 0 ONLY when results are durably OFF the box (git push or
# modal put succeeded) -- and auto-terminate is gated on that, so a box never
# self-destructs before the results are safe.
# --------------------------------------------------------------------------- #
echo "=== [4/5] save results (SAVE_MODE=${SAVE_MODE:-git}) ==="

git_save_and_push() {
  command -v git >/dev/null 2>&1 || { echo "  [git] git not found on box"; return 1; }
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "  [git] not a git repo"; return 1; }
  # The box clone has no committer identity; set a throwaway one.
  git config user.email >/dev/null 2>&1 || git config user.email "lambda-box@microbe.local"
  git config user.name  >/dev/null 2>&1 || git config user.name  "lambda-box"
  # runs/ is gitignored -> force-add the (small) result artifacts only.
  git add -f runs/lora >/dev/null 2>&1 || true
  if git diff --cached --quiet; then
    echo "  [git] nothing new to commit (results already committed)"
  else
    git commit -q -m "results: lora run $(date -u +%FT%TZ) modes=[${MODES:-}] seeds=[${SEEDS:-}]" \
      || { echo "  [git] commit failed"; return 1; }
    echo "  [git] committed runs/lora to $(git rev-parse --abbrev-ref HEAD)"
  fi
  # Push needs a token. lambda_full_launch.sh writes GH_TOKEN into ~/run_env.sh
  # when one is available locally; without it we keep the commit on the box only.
  if [ -z "${GH_TOKEN:-}" ]; then
    echo "  [git] no GH_TOKEN on box -> committed locally only; pull via scp (below). NOT terminating."
    return 1
  fi
  local slug="${GH_REPO_SLUG:-miyu-horiuchi/microbe-foundation}"
  local branch; branch="$(git rev-parse --abbrev-ref HEAD)"
  # Token only ever lives in this ephemeral push URL; output suppressed so it is
  # never echoed into the log.
  if git push "https://x-access-token:${GH_TOKEN}@github.com/${slug}.git" "HEAD:${branch}" >/dev/null 2>&1; then
    echo "  [git] pushed runs/lora -> github.com/${slug} ($branch)"
    return 0
  fi
  echo "  [git] WARNING: push failed -- results are committed locally; pull via scp. NOT terminating."
  return 1
}

SAVE_OK=0
if ls runs/lora/*.json >/dev/null 2>&1; then
  cp -f "${PILOT_LOG:-$HOME/pilot.log}" runs/lora/pilot.log 2>/dev/null || true
  # Table 31 (best-effort) so the artifact travels with the JSONs.
  python3 paper/lora_finetune_compare.py --runs-dir runs/lora || true
  cp -f paper/tables/31_lora_finetune.md runs/lora/ 2>/dev/null || true
  case "${SAVE_MODE:-git}" in
    git)
      if git_save_and_push; then SAVE_OK=1; fi ;;
    modal)
      if python3 -m modal volume put microbe-esm2-perprotein runs/lora "${REMOTE_DIR:-lora_pilot}" --force; then
        SAVE_OK=1
        echo "  saved runs/lora -> modal volume microbe-esm2-perprotein:/${REMOTE_DIR:-lora_pilot}"
      else
        echo "  WARNING: modal volume put failed -- NOT terminating so you can recover."
      fi ;;
    none|*)
      echo "  SAVE_MODE=none: results left in runs/lora -- pull via scp (below). NOT terminating." ;;
  esac
else
  echo "  no result JSONs -- NOT terminating; inspect the log above."
fi

echo "=== [5/5] auto-terminate ==="
if [ "${AUTO_TERMINATE:-0}" = "1" ] && [ "$SAVE_OK" = "1" ] \
   && [ -n "${LAMBDA_API_KEY:-}" ] && [ -n "${INSTANCE_ID:-}" ]; then
  echo "  results durably saved; terminating instance $INSTANCE_ID to stop billing..."
  curl -fsS -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
    -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" && echo "  terminate request sent." \
    || echo "  WARNING: terminate API call failed -- terminate manually."
else
  echo "  auto-terminate skipped (AUTO_TERMINATE=${AUTO_TERMINATE:-0}, save_ok=$SAVE_OK)."
  echo "  box is STILL UP -- a watchdog (if armed) will cap cost; results are in runs/lora."
fi

echo
echo "=== DONE. Pull results onto your laptop (then the agent reads them): ==="
case "${SAVE_MODE:-git}" in
  modal) echo "  modal volume get microbe-esm2-perprotein ${REMOTE_DIR:-lora_pilot} ./runs/lora_pilot --force" ;;
  git)   echo "  # if pushed: git fetch && git checkout ${BRANCH:-feat/set-transformer-tier1} && git pull" ;;
esac
echo "  # always works (no Modal, no token): scp the JSONs straight off the box:"
echo "  #   scp -r ubuntu@<BOX_IP>:microbe-foundation/runs/lora ./runs/lora_from_box"
