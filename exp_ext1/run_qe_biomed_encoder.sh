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
MODEL_NAME=${4:-pritamdeka/S-BioBERT-snli-multinli-stsb}
DEVICE=${5:-cuda}

# This script assumes it is executed from the reproducibility repo root.
# It reruns the pipeline with a biomedical sentence encoder to obtain non-trivial Q-E baselines.

python scripts/03_encode_text.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --model_name "$MODEL_NAME" \
  --device "$DEVICE"

# Keep hyperparameters consistent with the paper defaults unless you want to tune.
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
  --seed "$SEED"

# Optional: regenerate paper artifacts after running BOTH datasets.
# python scripts/11_make_paper_artifacts.py --datasets hpo do --subset_size 5000 --seed 0 \
#   --data_dir data/processed --out_dir paper_artifacts
