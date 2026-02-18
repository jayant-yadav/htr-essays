#!/bin/bash
#
# Evaluation script for HTR model
#
# Usage:
#   bash scripts/evaluate.sh outputs/final_model
#   bash scripts/evaluate.sh outputs/final_model test
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$PROJECT_DIR")/200essays"

CHECKPOINT="${1:-outputs/final_model}"
SPLIT="${2:-test}"
OUTPUT_DIR="${PROJECT_DIR}/evaluation_results"
BATCH_SIZE=16

echo "=========================================="
echo "HTR Evaluation Script"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT"
    exit 1
fi

# Run evaluation
cd "$PROJECT_DIR"
poetry run python3 -m htr_essays.evaluation.evaluate \
    --checkpoint $CHECKPOINT \
    --data_dir "$DATA_DIR" \
    --annotations_file "$DATA_DIR/json_full.json" \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --save_predictions

echo
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
