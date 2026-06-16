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
MODES="${MODES:-frozen lora}" SEEDS="${SEEDS:-0}" EPOCHS="${EPOCHS:-5}" \
  MODEL="${MODEL:-facebook/esm2_t30_150M_UR50D}" \
  MAX_PROTEINS="${MAX_PROTEINS:-128}" PROTEINS="${PROTEINS:-data/esm2_proteins}" \
  EXTRA="${EXTRA:---grad-checkpoint --balanced-families --class-weights --max-genomes ${MAX_GENOMES:-15000}}" \
  INCREMENTAL_MODAL="${INCREMENTAL_MODAL:-1}" REMOTE_DIR="${REMOTE_DIR:-lora_pilot}" \
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
