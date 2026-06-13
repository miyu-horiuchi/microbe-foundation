#!/usr/bin/env bash
#
# tier1_runs.sh — reproduce the tier-1 (ICML/NeurIPS/AAAI) experiment matrix.
#
# Runs the two submission-blocking experiments on the *real* per-protein ESM-2
# embeddings and produces all metrics/checkpoints the manuscript needs:
#
#   A. Pooling comparison (mean vs attention vs set_transformer) across the
#      species/genus/family generalization regimes, with multiple seeds + CIs.
#   B. Family-balanced training (the fix the cross-clade diagnostic, Table 15,
#      prescribes) with the set_transformer pooler on the family-held-out split.
#   C. The cross-clade diagnostic + retrieval-head tables (Tables 15-17).
#
# This needs a GPU box and the per-protein embedding directory produced by
# compute_esm2_perprotein_mp.py (a dir of <bacdive_id>.npy + manifest.parquet).
# It does NOT run inside the dev sandbox (DataLoader workers crash there).
#
# Usage:
#   PERPROTEIN=data/esm2_perprotein bash scripts/tier1_runs.sh
#   PERPROTEIN=... SEEDS="0 1 2 3 4" EPOCHS=40 bash scripts/tier1_runs.sh
#   POOLINGS="attention set_transformer" bash scripts/tier1_runs.sh   # subset
#
# Each model.py invocation is independent; re-run with a different OUT_DIR to
# avoid clobbering. Aggregate with: python paper/aggregate_seeds.py (see end).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- config (override via env) ---------------------------------------------
PERPROTEIN="${PERPROTEIN:-data/esm2_perprotein}"   # per-protein .npy dir + manifest
POOLINGS="${POOLINGS:-mean attention set_transformer}"
SPLITS="${SPLITS:-species genus family}"
SEEDS="${SEEDS:-0 1 2 3 4}"                          # >=5 seeds -> CIs for headline numbers
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-128}"
HIDDEN="${HIDDEN:-512}"
WORKERS="${WORKERS:-8}"
MAXPROT="${MAXPROT:-2048}"                           # cap proteins/genome to bound GPU mem
ST_HEADS="${ST_HEADS:-4}"
ST_INDUCING="${ST_INDUCING:-16}"
OUT_DIR="${OUT_DIR:-runs/tier1}"
# ----------------------------------------------------------------------------

if [ ! -d "$PERPROTEIN" ]; then
    echo "ERROR: per-protein dir '$PERPROTEIN' not found." >&2
    echo "Build it first with: python compute_esm2_perprotein_mp.py ..." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Per-protein embeddings: $PERPROTEIN"
echo "Poolings: $POOLINGS | Splits: $SPLITS | Seeds: $SEEDS | Epochs: $EPOCHS"
echo "Writing metrics/checkpoints under: $OUT_DIR"

train_one() {
    local pooling="$1" split="$2" seed="$3"; shift 3
    local extra=("$@")
    local tag="${pooling}_${split}_s${seed}"
    [ "${#extra[@]}" -gt 0 ] && tag="${tag}_$(echo "${extra[*]}" | tr -d ' -')"
    local metrics="${OUT_DIR}/${tag}.json"
    if [ -f "$metrics" ]; then
        echo "[skip] $tag (metrics exist)"
        return 0
    fi
    echo "=== train $tag ==="
    python3 model.py \
        --per-protein "$PERPROTEIN" \
        --pooling "$pooling" \
        --split-level "$split" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --hidden "$HIDDEN" \
        --num-workers "$WORKERS" \
        --max-proteins "$MAXPROT" \
        --st-heads "$ST_HEADS" \
        --st-inducing "$ST_INDUCING" \
        --class-weights \
        --scheduler cosine \
        --save-metrics "$metrics" \
        --save-model "${OUT_DIR}/${tag}.pt" \
        --run-name "$tag" \
        "${extra[@]}"
}

# ---- A. Pooling x split x seed ---------------------------------------------
echo; echo "######## A. POOLING COMPARISON ########"
for pooling in $POOLINGS; do
    for split in $SPLITS; do
        for seed in $SEEDS; do
            train_one "$pooling" "$split" "$seed"
        done
    done
done

# ---- B. Family-balanced training (the Table-15 fix) on family split --------
echo; echo "######## B. FAMILY-BALANCED TRAINING ########"
for pooling in $POOLINGS; do
    for seed in $SEEDS; do
        train_one "$pooling" family "$seed" --balanced-families
    done
done

# ---- C. Cross-clade diagnostic + retrieval heads (Tables 15-17) ------------
echo; echo "######## C. CROSS-CLADE DIAGNOSTIC + RETRIEVAL ########"
python3 cross_clade_diagnostic.py --out-dir paper/tables || echo "[warn] diagnostic failed"
python3 retrieval_head.py        --out-dir paper/tables || echo "[warn] retrieval_head failed"
python3 adaptive_retrieval.py    --out-dir paper/tables || echo "[warn] adaptive_retrieval failed"

echo
echo "============================================================"
echo "DONE. Per-run metrics + checkpoints in: $OUT_DIR"
echo "Next:"
echo "  1. Aggregate seeds into mean +/- CI tables for the paper:"
echo "       python3 paper/aggregate_seeds.py --runs-dir $OUT_DIR --out paper/tables"
echo "  2. Cross-clade results on the trained model (open item): blend each"
echo "     checkpoint's per-genome predictions (model.py --save-predictions"
echo "     --single-task <trait>) with k-NN, mirroring retrieval_head.py but"
echo "     consuming model probs instead of a fresh LR probe."
echo "============================================================"
