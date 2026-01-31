#!/bin/bash
# Run complete exp_ext4 pipeline for Table 9 generation

set -e

SUBSET_SIZE=${1:-5000}
SEED=${2:-0}
DATA_DIR=${3:-"data/processed"}

echo "============================================================"
echo "Exp_ext4: Complete ext4 Generation Pipeline"
echo "============================================================"
echo "Subset:       $SUBSET_SIZE nodes"
echo "Seed:         $SEED"
echo "Data dir:     $DATA_DIR"
echo "============================================================"
echo ""

# Check if baseline exists
echo "[Step 0/4] Checking for baseline results..."

for DATASET in hpo do; do
    ROOT_DIR="$DATA_DIR/$DATASET/${SUBSET_SIZE}_seed${SEED}"
    
    if [ ! -f "$ROOT_DIR/gate_linear.pt" ] || [ ! -d "$ROOT_DIR/indexes" ]; then
        echo "ERROR: Baseline not found for $DATASET"
        echo "Please run: bash exp_basic/run_basic_pipeline.sh $DATASET $SUBSET_SIZE $SEED"
        exit 1
    fi
    
    echo "  ✓ $DATASET baseline found"
done

echo ""

# Run noise retrieval tests
echo "[Step 1/4] Running noise retrieval tests..."
echo ""

for DATASET in hpo do; do
    echo "─────────────────────────────────────────────────────────"
    echo "Testing: $DATASET"
    echo "─────────────────────────────────────────────────────────"
    bash exp_ext4/run_noise_retrieval.sh $DATASET $SUBSET_SIZE $SEED $DATA_DIR
    echo ""
done

# Aggregate results
echo "[Step 2/4] Aggregating results from both datasets..."
python exp_ext4/aggregate_ext4_results.py \
    --data_dir $DATA_DIR \
    --out_dir paper_artifacts/ext4 \
    --subset_size $SUBSET_SIZE \
    --seed $SEED

echo ""

# Display summary
echo "[Step 3/4] Displaying results..."
echo ""

for DATASET in hpo do; do
    RESULTS_FILE="$DATA_DIR/$DATASET/${SUBSET_SIZE}_seed${SEED}/results/ext4_noise_retrieval.csv"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "─────────────────────────────────────────────────────────"
        echo "$DATASET results:"
        echo "─────────────────────────────────────────────────────────"
        cat "$RESULTS_FILE" | column -t -s,
        echo ""
    fi
done

# Final summary
echo "[Step 4/4] Pipeline complete!"
echo ""
echo "============================================================"
echo "✓ All exp_ext4 experiments completed successfully!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  Individual datasets:"
for DATASET in hpo do; do
    echo "    - data/processed/$DATASET/${SUBSET_SIZE}_seed${SEED}/results/ext4_noise_retrieval.csv"
done
echo ""
echo "  Aggregated outputs:"
echo "    - paper_artifacts/ext4/ext4_combined.tex"
echo "    - paper_artifacts/ext4/fig_ext4_degradation.pdf"
echo "    - paper_artifacts/ext4/fig_ext4_retention.pdf"
echo "    - paper_artifacts/ext4/ext4_SUMMARY.md"
echo ""
echo "Next steps:"
echo "  1. Review ext4_SUMMARY.md for key findings"
echo "  2. Insert ext4_combined.tex into paper Section 6.12"
echo "  3. Add figures to supplementary materials"
echo "============================================================"
