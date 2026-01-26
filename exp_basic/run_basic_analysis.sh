#!/usr/bin/env bash
# =============================================================================
# run_basic_analysis.sh — Basic analysis (indexability + efficiency benchmarks)
# =============================================================================
# This script runs indexability stress test and efficiency benchmarks.
# Should be run AFTER run_basic_pipeline.sh completes.
#
# Usage:
#   bash exp_basic/run_basic_analysis.sh <dataset> <subset_size> <seed> <device>
#
# Examples:
#   bash exp_basic/run_basic_analysis.sh hpo 5000 0 cuda
#   bash exp_basic/run_basic_analysis.sh do  5000 0 cpu
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
echo " HyEm Basic Analysis"
echo " Dataset: $DATASET"
echo " Subset:  $SUBSET_SIZE nodes"
echo " Seed:    $SEED"
echo " Device:  $DEVICE"
echo "=========================================="

# Step 1: Indexability stress test
echo ""
echo "== Step 1/2: Indexability stress test (recall curve) =="
python scripts/09_indexability_test.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --k 10 \
  --Ls 20 50 100 200 500 1000 \
  --query_type QE \
  --max_queries 200 \
  --device "$DEVICE"

# Step 2: Efficiency benchmark
echo ""
echo "== Step 2/2: Efficiency benchmark (latency + index size) =="
python scripts/10_efficiency_benchmark.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --num_queries 200 \
  --device "$DEVICE"

echo ""
echo "=========================================="
echo " Analysis completed!"
echo " Output: data/processed/${DATASET}/${SUBSET_SIZE}_seed${SEED}/analysis/"
echo "=========================================="
