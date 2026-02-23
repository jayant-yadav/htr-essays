#!/usr/bin/env bash
#SBATCH -A naiss2026-4-110
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 00:10:00
#SBATCH --output=logs/infer/%j.log

# Inference script for HTR model
#
# Usage:
#   bash scripts/infer.sh outputs/final_model --image path/to/image.jpg
#   bash scripts/infer.sh outputs/final_model --image_dir path/to/images/
#

# Get the directory where this script is located
SCRIPT_DIR="/mimer/NOBACKUP/groups/studentessays/htr-essays/scripts"
PROJECT_DIR="/mimer/NOBACKUP/groups/studentessays/htr-essays"

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

module load poetry

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
