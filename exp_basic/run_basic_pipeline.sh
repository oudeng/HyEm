#!/usr/bin/env bash
# =============================================================================
# run_basic_pipeline.sh — Basic experiment pipeline for HyEm
# =============================================================================
# This script runs the complete HyEm pipeline from data preparation to evaluation.
#
# Usage:
#   bash exp_basic/run_basic_pipeline.sh <dataset> <subset_size> <seed> <device>
#
# Examples:
#   bash exp_basic/run_basic_pipeline.sh hpo 5000 0 cuda
#   bash exp_basic/run_basic_pipeline.sh do  5000 0 cpu
# =============================================================================
set -euo pipefail

# Get the repository root directory (parent of exp_basic)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Add src to PYTHONPATH so hyem module can be found
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATASET=${1:-hpo}
SUBSET_SIZE=${2:-5000}
SEED=${3:-0}
DEVICE=${4:-cpu}

echo "=========================================="
echo " HyEm Basic Pipeline"
echo " Dataset: $DATASET"
echo " Subset:  $SUBSET_SIZE nodes"
echo " Seed:    $SEED"
echo " Device:  $DEVICE"
echo "=========================================="

# Step 1: Build subset
echo ""
echo "== Step 1/8: Build subset =="
python scripts/01_build_subset.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED"

# Step 2: Build queries
echo ""
echo "== Step 2/8: Build query benchmarks (Q-E / Q-H / Q-M) =="
python scripts/02_build_queries.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED"

# Step 3: Encode text
echo ""
echo "== Step 3/8: Encode node and query texts =="
python scripts/03_encode_text.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE"

# Step 4: Train embeddings
echo ""
echo "== Step 4/8: Train graph embeddings (HYEM + baselines) =="
python scripts/04_train_embeddings.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --epochs 10 \
  --dim 32 \
  --radius_budget 3.0 \
  --device "$DEVICE"

# Step 5: Train adapters
echo ""
echo "== Step 5/8: Train adapters =="
python scripts/05_train_adapters.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --epochs 5 \
  --device "$DEVICE"

# Step 6: Train gate
echo ""
echo "== Step 6/8: Train gate (rule / linear / MLP) =="
python scripts/06_train_gate.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --epochs 5 \
  --device "$DEVICE"

# Step 7: Build indexes
echo ""
echo "== Step 7/8: Build ANN indexes (HNSW) =="
python scripts/07_build_indexes.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED"

# Step 8: Evaluate
echo ""
echo "== Step 8/8: Evaluate retrieval =="
python scripts/08_eval_retrieval.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --k 10 \
  --L_h 200 \
  --L_e 200 \
  --device "$DEVICE"

echo ""
echo "=========================================="
echo " Pipeline completed!"
echo " Results: data/processed/${DATASET}/${SUBSET_SIZE}_seed${SEED}/results/"
echo "=========================================="
