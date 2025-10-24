#!/bin/bash
# Resume training from a checkpoint
# Usage: ./resume_training.sh <checkpoint_path> [config_file]

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [config_file]"
    echo ""
    echo "Example:"
    echo "  $0 results/tiny_gpu_1000/checkpoints/checkpoint_epoch_60.pt configs/tiny_gpu_1000.yaml"
    exit 1
fi

CHECKPOINT_PATH="$1"
CONFIG_FILE="${2:-configs/tiny_gpu_1000.yaml}"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if checkpoint exists
if [ ! -f "$PROJECT_DIR/$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $PROJECT_DIR/$CHECKPOINT_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=== Resume Training ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_FILE"
echo "======================="
echo ""

# Run the training with resume flag
cd "$PROJECT_DIR/.." || exit 1

python3 training_gpu_prefetch/scripts/run_training_gpu.py \
    --model GT+RL \
    --config "$CONFIG_FILE" \
    --resume "training_gpu_prefetch/$CHECKPOINT_PATH"
