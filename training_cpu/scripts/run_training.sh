#!/bin/bash

# Training script for CPU-based CVRP models
# This script activates the environment and runs training

set -e  # Exit on error

echo "=================================================="
echo "🚀 CVRP Model Training (CPU)"
echo "=================================================="

# Check if we're in the training_cpu/scripts directory
if [ ! -f "run_training.py" ]; then
    echo "❌ Error: run_training.py not found!"
    echo "Please run this script from the training_cpu/scripts directory."
    exit 1
fi

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "../../venv" ]; then
        echo "📦 Activating virtual environment..."
        source ../../venv/bin/activate
    else
        echo "⚠️  Virtual environment not found. Please run setup_venv.sh first."
        exit 1
    fi
else
    echo "✅ Virtual environment already activated"
fi

# Parse arguments or use defaults
CONFIG=${1:-"../../configs/default.yaml"}
MODEL=${2:-"GT+RL"}

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Model: $MODEL"
echo ""

# Run training
if [ "$MODEL" == "all" ]; then
    echo "Training all models..."
    python run_training.py --config "$CONFIG" --all
else
    echo "Training $MODEL..."
    python run_training.py --config "$CONFIG" --model "$MODEL"
fi

echo ""
echo "=================================================="
echo "✅ Training complete!"
echo "=================================================="
echo ""
echo "Results saved in: training_cpu/results/"
echo ""
echo "To generate comparison plots, run:"
echo "  python make_comparative_plot.py --config $CONFIG"
