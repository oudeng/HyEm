#!/usr/bin/env bash
# =============================================================================
# run_paper_artifacts.sh — Generate LaTeX tables and figures for paper
# =============================================================================
# This script generates paper artifacts from experiment results.
# Should be run AFTER run_basic_pipeline.sh and run_basic_analysis.sh complete.
#
# Usage:
#   bash exp_basic/run_paper_artifacts.sh <datasets> <subset_size> <seed>
#
# Examples:
#   bash exp_basic/run_paper_artifacts.sh "hpo do" 5000 0
#   bash exp_basic/run_paper_artifacts.sh "hpo" 5000 0
# =============================================================================
set -euo pipefail

# Get the repository root directory (parent of exp_basic)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Add src to PYTHONPATH so hyem module can be found
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATASETS=${1:-"hpo do"}
SUBSET_SIZE=${2:-5000}
SEED=${3:-0}

echo "=========================================="
echo " HyEm Paper Artifacts Generation"
echo " Datasets: $DATASETS"
echo " Subset:   $SUBSET_SIZE nodes"
echo " Seed:     $SEED"
echo "=========================================="

python scripts/11_make_paper_artifacts.py \
  --datasets $DATASETS \
  --subset_size "$SUBSET_SIZE" \
  --seed "$SEED" \
  --data_dir data/processed \
  --out_dir paper_artifacts

echo ""
echo "=========================================="
echo " Paper artifacts generated!"
echo " Output: paper_artifacts/"
echo "=========================================="
