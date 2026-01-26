#!/usr/bin/env bash
set -euo pipefail

# Get the repository root directory (parent of exp_ext1)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Add src to PYTHONPATH so hyem module can be found
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATASET=${1:-hpo}
SUBSET_SIZE=${2:-5000}
SEED=${3:-0}
DEVICE=${4:-cuda}
R_LIST_STR=${5:-"2.0 3.0 4.0 5.0"}

# This script is a TEMPLATE.
# Depending on how your repo stores run artifacts, you may want to:
#   - set an explicit output/run name for each R (if supported), or
#   - move/copy the produced analysis files after each run so they are not overwritten.

# Example usage:
#   bash exp_ext1/run_radius_sweep.sh hpo 5000 0 cuda "2.0 3.0 4.0 5.0"

for R in $R_LIST_STR; do
  echo "=== Radius sweep: R=$R ($DATASET, subset=$SUBSET_SIZE, seed=$SEED) ==="

  python scripts/04_train_embeddings.py \
    --dataset "$DATASET" \
    --subset_size "$SUBSET_SIZE" \
    --seed "$SEED" \
    --radius_budget "$R" \
    --dim 32 \
    --device "$DEVICE"

  python scripts/05_train_adapters.py \
    --dataset "$DATASET" \
    --subset_size "$SUBSET_SIZE" \
    --seed "$SEED" \
    --device "$DEVICE"

  python scripts/07_build_indexes.py \
    --dataset "$DATASET" \
    --subset_size "$SUBSET_SIZE" \
    --seed "$SEED"

  python scripts/09_indexability_test.py \
    --dataset "$DATASET" \
    --subset_size "$SUBSET_SIZE" \
    --seed "$SEED" \
    --device "$DEVICE"

  # Copy out the recall curve so multiple R values can be plotted together.
  mkdir -p data/processed/${DATASET}/${SUBSET_SIZE}_seed${SEED}/analysis
  cp data/processed/${DATASET}/${SUBSET_SIZE}_seed${SEED}/analysis/indexability_recall_curve.csv \
    data/processed/${DATASET}/${SUBSET_SIZE}_seed${SEED}/analysis/indexability_recall_curve_R${R}.csv

done