#!/bin/bash
# Run all exp_ext3 analyses for a given dataset
# Usage: bash exp_ext3/run_all_ext3.sh <dataset> <subset_size> <seed> [device]

set -e

DATASET=${1:-hpo}
SUBSET_SIZE=${2:-5000}
SEED=${3:-0}
DEVICE=${4:-cpu}
DATA_DIR="data/processed"

echo "=============================================="
echo "Running exp_ext3 analyses"
echo "Dataset: $DATASET"
echo "Subset size: $SUBSET_SIZE"
echo "Seed: $SEED"
echo "=============================================="

# 1. Safety Valve Analysis (M2)
echo ""
echo "[1/5] Running Safety Valve Analysis..."
python exp_ext3/analyze_safety_valve.py \
    --dataset "$DATASET" \
    --subset_size "$SUBSET_SIZE" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR"

# 2. Depth-Stratified Analysis (M1, M2) - only if depth.json exists
DEPTH_PATH="${DATA_DIR}/${DATASET}/${SUBSET_SIZE}_seed${SEED}/depth.json"
if [ -f "$DEPTH_PATH" ]; then
    echo ""
    echo "[2/5] Running Depth-Stratified Analysis..."
    python exp_ext3/depth_stratified_analysis.py \
        --dataset "$DATASET" \
        --subset_size "$SUBSET_SIZE" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --depth_buckets 4
else
    echo ""
    echo "[2/5] Skipping Depth-Stratified Analysis (depth.json not found)"
    echo "      To enable, run: python scripts/01_build_subset.py to generate depth.json"
fi

# 3. Candidate Pooling Ablation (m3)
echo ""
echo "[3/5] Running Candidate Pooling Ablation..."
python exp_ext3/candidate_pooling_ablation.py \
    --dataset "$DATASET" \
    --subset_size "$SUBSET_SIZE" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR"

# 4. Gate Robustness Test (M3) - only if gate and embeddings exist
GATE_PATH="${DATA_DIR}/${DATASET}/${SUBSET_SIZE}_seed${SEED}/gate_linear.pt"
EMB_PATH="${DATA_DIR}/${DATASET}/${SUBSET_SIZE}_seed${SEED}/emb_queries_test.npy"
if [ -f "$GATE_PATH" ] && [ -f "$EMB_PATH" ]; then
    echo ""
    echo "[4/5] Running Gate Robustness Test..."
    python exp_ext3/gate_robustness_test.py \
        --dataset "$DATASET" \
        --subset_size "$SUBSET_SIZE" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --device "$DEVICE" \
        --noise_levels "0.0,0.1,0.2,0.3"
else
    echo ""
    echo "[4/5] Skipping Gate Robustness Test (gate or embeddings not found)"
    echo "      Required files: gate_linear.pt, emb_queries_test.npy"
fi

echo ""
echo "=============================================="
echo "exp_ext3 analyses complete for $DATASET"
echo "Results saved to: ${DATA_DIR}/${DATASET}/${SUBSET_SIZE}_seed${SEED}/results/"
echo "=============================================="

# 5. Generate revision figures (optional, run after all datasets are processed)
echo ""
echo "[5/5] To generate combined revision figures, run:"
echo "      python exp_ext3/make_revision_figures.py --data_dir $DATA_DIR --out_dir paper_artifacts/ext3"