#!/bin/bash
# Run end-to-end retrieval evaluation under noise (Table 9)

set -e

DATASET=$1
SUBSET_SIZE=${2:-5000}
SEED=${3:-0}
DATA_DIR=${4:-"data/processed"}

if [ -z "$DATASET" ]; then
    echo "Usage: bash exp_ext4/run_noise_retrieval.sh <dataset> [subset_size] [seed] [data_dir]"
    echo ""
    echo "Arguments:"
    echo "  dataset       Dataset name (hpo or do)"
    echo "  subset_size   Subset size (default: 5000)"
    echo "  seed          Random seed (default: 0)"
    echo "  data_dir      Data directory (default: data/processed)"
    echo ""
    echo "Example:"
    echo "  bash exp_ext4/run_noise_retrieval.sh hpo 5000 0"
    exit 1
fi

echo "============================================================"
echo "Exp_ext4: End-to-End Retrieval Under Noise (ext4)"
echo "============================================================"
echo "Dataset:      $DATASET"
echo "Subset:       $SUBSET_SIZE nodes"
echo "Seed:         $SEED"
echo "Data dir:     $DATA_DIR"
echo "============================================================"
echo ""

# Check if baseline results exist
ROOT_DIR="$DATA_DIR/$DATASET/${SUBSET_SIZE}_seed${SEED}"

if [ ! -f "$ROOT_DIR/gate_linear.pt" ]; then
    echo "ERROR: Gate not found at $ROOT_DIR/gate_linear.pt"
    echo "Please run the baseline pipeline first:"
    echo "  bash exp_basic/run_basic_pipeline.sh $DATASET $SUBSET_SIZE $SEED"
    exit 1
fi

if [ ! -f "$ROOT_DIR/indexes/index_hyem.bin" ]; then
    echo "ERROR: Indexes not found at $ROOT_DIR/indexes/"
    echo "Please run the baseline pipeline first:"
    echo "  bash exp_basic/run_basic_pipeline.sh $DATASET $SUBSET_SIZE $SEED"
    exit 1
fi

echo "✓ Found baseline results"
echo ""

# Run noise retrieval test
echo "Running noise retrieval test..."
python exp_ext4/noise_retrieval_test.py \
    --dataset $DATASET \
    --subset_size $SUBSET_SIZE \
    --seed $SEED \
    --data_dir $DATA_DIR \
    --noise_levels "0.0,0.1,0.2,0.3" \
    --L_h 50 \
    --L_e 50 \
    --device cpu

echo ""
echo "============================================================"
echo "✓ Noise retrieval test complete!"
echo "============================================================"
echo "Results saved to:"
echo "  - $ROOT_DIR/results/ext4_noise_retrieval.csv"
echo "  - $ROOT_DIR/results/ext4_noise_retrieval.tex"
echo ""
echo "To run for both datasets:"
echo "  bash exp_ext4/run_noise_retrieval.sh hpo"
echo "  bash exp_ext4/run_noise_retrieval.sh do"
echo "============================================================"
