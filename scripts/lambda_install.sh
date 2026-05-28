#!/usr/bin/env bash
#
# lambda_install.sh — bootstrap a Lambda Labs GPU instance for ESM-2 feature
# extraction. Idempotent: safe to re-run.
#
# Run from inside the cloned microbe-foundation repo:
#     git clone https://github.com/miyu-horiuchi/microbe-foundation
#     cd microbe-foundation
#     bash scripts/lambda_install.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "microbe-foundation — Lambda GPU install"
echo "============================================================"

# --- Python version ---
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "python3: $PY_VERSION  (need >= 3.11 for ESM-2 path)"
if python3 -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    echo "  OK"
else
    echo "  WARNING: Python < 3.11 detected. The transformers/Pillow path will likely fail."
    echo "           On Lambda, try: conda activate pytorch (their default env is 3.11+)"
fi

# --- GPU visible ---
echo
echo "GPU check:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found. Are you on a GPU instance?"
fi

# --- Install deps ---
echo
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -r requirements.txt

# --- Sanity check ---
echo
echo "Sanity checks:"
python3 -c "import torch; print(f'  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'  transformers {transformers.__version__}')"
python3 -c "import pyrodigal; print(f'  pyrodigal {pyrodigal.__version__}')"
python3 schema.py >/dev/null && echo "  schema.py OK"

# --- NCBI API key reminder ---
echo
if [ -z "${NCBI_API_KEY:-}" ]; then
    echo "============================================================"
    echo "  RECOMMENDED: set NCBI_API_KEY for 3x faster genome fetches"
    echo "  Get a free key at https://www.ncbi.nlm.nih.gov/account/settings/"
    echo "  then: export NCBI_API_KEY=<your-key>"
    echo "============================================================"
else
    echo "  NCBI_API_KEY is set (good)"
fi

# --- Next step prompt ---
echo
echo "============================================================"
echo "Install complete. Next steps:"
echo
echo "  1. Make sure data/genome_accessions.tsv exists. If you ran"
echo "     fetch+parse+extract_genome_accessions locally, scp it up:"
echo "       scp local:microbe-foundation/data/genome_accessions.tsv data/"
echo
echo "     Otherwise generate it here (3-4 hours fetch):"
echo "       python fetch_bacdive.py && python parse_bacdive.py"
echo "       python extract_genome_accessions.py"
echo
echo "  2. Smoke-test on 10 genomes:"
echo "       python compute_esm2_features.py \\"
echo "           --model facebook/esm2_t6_8M_UR50D \\"
echo "           --sample-n 20 --batch-size 4 --limit 10"
echo
echo "  3. Full run (recommended ESM-2 size for A100):"
echo "       python compute_esm2_features.py \\"
echo "           --model facebook/esm2_t30_150M_UR50D \\"
echo "           --sample-n 50 --batch-size 32"
echo "============================================================"
