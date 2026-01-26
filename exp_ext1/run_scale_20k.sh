#!/usr/bin/env bash
set -euo pipefail

# Get the repository root directory (parent of exp_ext1)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Add src to PYTHONPATH so hyem module can be found
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATASET=${1:-hpo}
SUBSET_SIZE=${2:-20000}
SEED=${3:-0}
DEVICE=${4:-cuda}

# This script assumes it is executed from the reproducibility repo root.
# It runs the standard pipeline on a larger subset (default: 20k) and then runs indexability + efficiency benchmarks.

python scripts/01_build_subset.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED"

python scripts/02_build_queries.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED"

python scripts/03_encode_text.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
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
  --seed "$SEED"

python scripts/09_indexability_test.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE"

python scripts/10_efficiency_benchmark.py \
  --dataset "$DATASET" \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED"