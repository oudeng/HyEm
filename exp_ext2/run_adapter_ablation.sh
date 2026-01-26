#!/usr/bin/env bash
set -euo pipefail

# Get the repository root directory (parent of exp_ext2)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Add src to PYTHONPATH so hyem module can be found
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATASET=${1:-hpo}
SUBSET_SIZE=${2:-5000}
SEED=${3:-0}
DEVICE=${4:-cuda}

# Assumes exp_basic pipeline is already run for the target dataset/subset.
python exp_ext2/adapter_ablation.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[done] adapter ablation: ${DATASET}/${SUBSET_SIZE}_seed${SEED}"
