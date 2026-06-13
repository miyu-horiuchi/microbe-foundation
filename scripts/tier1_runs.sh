#!/usr/bin/env bash
#
# tier1_runs.sh — reproduce the tier-1 (ICML/NeurIPS/AAAI) experiment matrix.
#
# Runs the submission-blocking experiments on the *real* per-protein ESM-2
# embeddings and produces all metrics/checkpoints/tables the manuscript needs:
#
#   0. (optional) Re-extract embeddings from a larger ESM-2 (650M/3B) — the
#      "stronger encoder" lever (§4.8, Table 23). Gated on ESM_MODEL being set.
#   A. Pooling comparison (mean vs attention vs set_transformer) across the
#      species/genus/family generalization regimes, with multiple seeds + CIs.
#   B. Family-balanced training (the fix the cross-clade diagnostic, Table 15,
#      prescribes) with the set_transformer pooler on the family-held-out split.
#   C. The cross-clade diagnostic + retrieval-head tables (Tables 15-17).
#   D. Checkpoint-output cross-clade retrieval on the trained model (Table 19).
#   E. All frozen-feature analysis tables (Tables 20-26 + figures): failure-mode
#      taxonomy, coverage panel/scaling, encoder comparison, label audit, the
#      multi-label reformulation, and the cultivation_medium PU correction.
#
# A-D need a GPU box and the per-protein embedding directory produced by
# compute_esm2_perprotein_mp.py (a dir of <bacdive_id>.npy + manifest.parquet);
# they do NOT run inside the dev sandbox (DataLoader workers crash there).
# Section E is CPU-only and runs anywhere.
#
# Usage:
#   PERPROTEIN=data/esm2_perprotein bash scripts/tier1_runs.sh
#   PERPROTEIN=... SEEDS="0 1 2 3 4" EPOCHS=40 bash scripts/tier1_runs.sh
#   POOLINGS="attention set_transformer" bash scripts/tier1_runs.sh   # subset
#   ANALYSIS_ONLY=1 bash scripts/tier1_runs.sh                        # just Section E (CPU)
#   # Larger-encoder lever: extract 650M embeddings, then train + compare on them:
#   ESM_MODEL=facebook/esm2_t33_650M_UR50D ESM_TAG=650M bash scripts/tier1_runs.sh
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
# Stronger-encoder lever (Section 0). Leave ESM_MODEL empty to use existing 150M.
ESM_MODEL="${ESM_MODEL:-}"                           # e.g. facebook/esm2_t33_650M_UR50D
ESM_TAG="${ESM_TAG:-650M}"                           # short label for output paths
EXTRACT_WORKERS="${EXTRACT_WORKERS:-16}"
ANALYSIS_ONLY="${ANALYSIS_ONLY:-0}"                  # 1 -> skip GPU sections, run E only
# Genome-level pooled features for the encoder-comparison probe (Section E).
BASE_FEATURES="${BASE_FEATURES:-data/esm2_features.npz}"
BIG_FEATURES="${BIG_FEATURES:-data/esm2_features_${ESM_TAG}.npz}"
# ----------------------------------------------------------------------------

run_analysis_tables() {
    echo; echo "######## E. FROZEN-FEATURE ANALYSIS TABLES (Tables 20-26) ########"
    # CPU-only; regenerate the full analysis suite the manuscript cites.
    python3 paper/failure_mode_analysis.py    --out-dir paper/tables || echo "[warn] failure_mode_analysis"
    python3 paper/coverage_panel.py           --out-dir paper/tables || echo "[warn] coverage_panel"
    python3 paper/coverage_scaling.py         --out-dir paper/tables || echo "[warn] coverage_scaling"
    python3 paper/label_quality_audit.py      --out-dir paper/tables || echo "[warn] label_quality_audit"
    python3 paper/multilabel_reformulation.py --out-dir paper/tables || echo "[warn] multilabel_reformulation"
    python3 paper/cultivation_medium_pu.py    --out-dir paper/tables || echo "[warn] cultivation_medium_pu"
    # Encoder comparison: if a larger-ESM genome feature store exists, compare
    # 150M vs that; otherwise fall back to the script default (150M vs bacformer).
    if [ -f "$BIG_FEATURES" ]; then
        echo "[encoder-comparison] esm2_150M vs esm2_${ESM_TAG}"
        python3 paper/encoder_comparison.py --out-dir paper/tables \
            --encoders "esm2_150M=${BASE_FEATURES}" "esm2_${ESM_TAG}=${BIG_FEATURES}" \
            || echo "[warn] encoder_comparison (big)"
    else
        python3 paper/encoder_comparison.py --out-dir paper/tables || echo "[warn] encoder_comparison"
    fi
}

# Fast path: regenerate analysis tables only (no GPU, no training).
if [ "$ANALYSIS_ONLY" = "1" ]; then
    run_analysis_tables
    echo; echo "ANALYSIS_ONLY=1: regenerated Tables 20-26. Done."
    exit 0
fi

# ---- 0. (optional) larger-ESM extraction (stronger-encoder lever) ----------
if [ -n "$ESM_MODEL" ]; then
    echo; echo "######## 0. EXTRACT EMBEDDINGS FROM $ESM_MODEL ########"
    PERPROTEIN="data/esm2_perprotein_${ESM_TAG}"
    if [ ! -d "$PERPROTEIN" ]; then
        echo "=== per-protein extraction -> $PERPROTEIN ==="
        python3 compute_esm2_perprotein_mp.py \
            --model "$ESM_MODEL" --out-dir "$PERPROTEIN" --workers "$EXTRACT_WORKERS"
    else
        echo "[skip] per-protein dir exists: $PERPROTEIN"
    fi
    if [ ! -f "$BIG_FEATURES" ]; then
        echo "=== genome-level pooled features -> $BIG_FEATURES (for encoder comparison) ==="
        python3 compute_esm2_features_mp.py \
            --model "$ESM_MODEL" --out "$BIG_FEATURES" --workers "$EXTRACT_WORKERS"
    else
        echo "[skip] genome features exist: $BIG_FEATURES"
    fi
fi

if [ ! -d "$PERPROTEIN" ]; then
    echo "ERROR: per-protein dir '$PERPROTEIN' not found." >&2
    echo "Build it first with: python compute_esm2_perprotein_mp.py ..." >&2
    echo "(or set ESM_MODEL=... to extract it here, or ANALYSIS_ONLY=1 for CPU tables)" >&2
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
    # Cross-clade retrieval needs the trained model's val+test probabilities, so
    # dump them on the family split (the cross-clade regime).
    local pred_args=()
    [ "$split" = "family" ] && pred_args=(--save-all-predictions "${OUT_DIR}/${tag}_preds.parquet")

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
        "${pred_args[@]}" \
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

# ---- D. Real-system cross-clade retrieval on the trained model (Table 19) ---
# Blend each family checkpoint's own probabilities with cross-clade k-NN.
echo; echo "######## D. CHECKPOINT RETRIEVAL (real system) ########"
for preds in "${OUT_DIR}"/*_family_s*_preds.parquet; do
    [ -e "$preds" ] || continue
    base="$(basename "$preds" _preds.parquet)"
    echo "=== checkpoint retrieval: $base ==="
    python3 checkpoint_retrieval.py --preds "$preds" \
        --features data/esm2_features.npz \
        --out-dir "${OUT_DIR}/retrieval_${base}" || echo "[warn] retrieval failed for $base"
done

# ---- E. Frozen-feature analysis tables (Tables 20-26 + figures) -------------
run_analysis_tables

echo
echo "============================================================"
echo "DONE. Per-run metrics + checkpoints in: $OUT_DIR"
echo "Analysis tables/figures regenerated under: paper/tables, paper/figures"
[ -n "$ESM_MODEL" ] && echo "Encoder: $ESM_MODEL (per-protein: $PERPROTEIN)"
echo "Next:"
echo "  Aggregate seeds into mean +/- CI tables for the paper:"
echo "     python3 paper/aggregate_seeds.py --runs-dir $OUT_DIR --out paper/tables"
echo "============================================================"
