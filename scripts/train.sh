#!/usr/bin/env bash
#SBATCH -A naiss2026-4-110
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:4
#SBATCH -t 1-00:00:00
#SBATCH --output=logs/train/%j.log

#
# Training script for HTR model on 4 A40 GPUs
#
# Usage:
#   sbatch scripts/train.sh              # Full training
#   sbatch scripts/train.sh --debug      # Debug mode with small dataset
#

# Get the directory where this script is located
SCRIPT_DIR="/mimer/NOBACKUP/groups/studentessays/htr-essays/scripts"
PROJECT_DIR="/mimer/NOBACKUP/groups/studentessays/htr-essays"
DATA_DIR="/mimer/NOBACKUP/groups/studentessays/200essays"

# Default values
DEBUG=""
NUM_GPUS=4
EPOCHS=50
BATCH_SIZE=8
OUTPUT_DIR="${PROJECT_DIR}/outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG="--debug"
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "HTR Training Script"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Debug mode: ${DEBUG:-No}"
echo "=========================================="
echo

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "$DATA_DIR/json_full.json" ]; then
    echo "ERROR: Annotations file not found: $DATA_DIR/json_full.json"
    exit 1
fi

module load poetry

# Create data splits if they don't exist
SPLIT_FILE="${PROJECT_DIR}/data_splits.json"
if [ ! -f "$SPLIT_FILE" ]; then
    echo "Creating data splits..."
    cd "$PROJECT_DIR"
    poetry run python3 -c "from htr_essays.data.dataset import create_data_splits; create_data_splits('$DATA_DIR/json_full.json', '$SPLIT_FILE')"
    echo
fi

# Run training
cd "$PROJECT_DIR"
if [ $NUM_GPUS -gt 1 ]; then
    echo "Launching distributed training on $NUM_GPUS GPUs..."
    poetry run python3 -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --use_env \
        -m htr_essays.training.train \
        $DEBUG \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_gpus $NUM_GPUS \
        --output_dir $OUTPUT_DIR \
        --data_dir "$DATA_DIR" \
        --annotations_file "$DATA_DIR/json_full.json" \
        --split_file "$SPLIT_FILE"
else
    echo "Launching single-GPU training..."
    poetry run python3 -m htr_essays.training.train \
        $DEBUG \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR \
        --data_dir "$DATA_DIR" \
        --annotations_file "$DATA_DIR/json_full.json" \
        --split_file "$SPLIT_FILE"
fi

echo
echo "=========================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
