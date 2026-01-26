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
MODEL_NAME=${4:-pritamdeka/S-BioBERT-snli-multinli-stsb}
DEVICE=${5:-cuda}

# This script reruns the pipeline in a realistic Q-E setting:
#   - index at least one synonym per entity (node_text_mode=label_def_1syn)
#   - use a biomedical sentence encoder (S-BioBERT)

python scripts/03_encode_text.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --model_name "$MODEL_NAME" \
  --node_text_mode label_def_1syn \
  --device "$DEVICE"

python scripts/04_train_embeddings.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --radius_budget 3.0 \
  --dim 32 \
  --device "$DEVICE"

python scripts/05_train_adapters.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE"

python scripts/06_train_gate.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE"

python scripts/07_build_indexes.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED"

python scripts/08_eval_retrieval.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[done] Q-E high-baseline run completed: ${DATASET}/${SUBSET_SIZE}_seed${SEED}"
