#!/bin/bash
#
# Inference script for HTR model
#
# Usage:
#   bash scripts/infer.sh outputs/final_model --image path/to/image.jpg
#   bash scripts/infer.sh outputs/final_model --image_dir path/to/images/
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT="${1:-outputs/final_model}"
shift

echo "=========================================="
echo "HTR Inference Script"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="
echo

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT"
    exit 1
fi

# Run inference
cd "$PROJECT_DIR"
poetry run python3 -m htr_essays.inference.infer \
    --checkpoint $CHECKPOINT \
    --visualize \
    "$@"

echo
echo "=========================================="
echo "Inference complete!"
echo "=========================================="
