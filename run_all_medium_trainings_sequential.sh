#!/bin/bash

# This script runs all three medium GPU trainings sequentially
# The script itself will run in a screen session

echo "========================================="
echo "Sequential Medium GPU Training Script"
echo "========================================="
echo ""
echo "This will run the following trainings in order:"
echo "1. medium_gpu_annealing"
echo "2. medium_gpu_large_model" 
echo "3. medium_gpu_optimal"
echo ""
echo "Each training will complete before the next one starts."
echo ""

# Function to display current time
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to check GPU memory
check_gpu_memory() {
    echo "GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader | head -1
}

cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL
source venv/bin/activate

echo "Environment activated at $(timestamp)"
check_gpu_memory
echo ""

# Kill any existing training sessions to free up memory
echo "Cleaning up any existing training sessions..."
screen -X -S training_medium_annealing quit 2>/dev/null
screen -X -S training_medium_large_model quit 2>/dev/null
screen -X -S training_medium_optimal quit 2>/dev/null
sleep 5

echo "Starting sequential training at $(timestamp)"
echo ""

# Training 1: Annealing
echo "========================================="
echo "1/3: Starting Medium GPU Annealing Training"
echo "Time: $(timestamp)"
echo "========================================="
python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_annealing.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_annealing_seq.log
echo ""
echo "Annealing training completed at $(timestamp)"
check_gpu_memory
echo ""
sleep 10  # Brief pause between trainings

# Training 2: Large Model
echo "========================================="
echo "2/3: Starting Medium GPU Large Model Training"
echo "Time: $(timestamp)"
echo "========================================="
python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_large_model.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_large_model_seq.log
echo ""
echo "Large model training completed at $(timestamp)"
check_gpu_memory
echo ""
sleep 10  # Brief pause between trainings

# Training 3: Optimal
echo "========================================="
echo "3/3: Starting Medium GPU Optimal Training"
echo "Time: $(timestamp)"
echo "========================================="
python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_optimal.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_optimal_seq.log
echo ""
echo "Optimal training completed at $(timestamp)"
check_gpu_memory
echo ""

echo "========================================="
echo "ALL TRAININGS COMPLETED SUCCESSFULLY!"
echo "Finished at: $(timestamp)"
echo "========================================="
echo ""
echo "Log files created:"
echo "  - training_medium_annealing_seq.log"
echo "  - training_medium_large_model_seq.log"
echo "  - training_medium_optimal_seq.log"
echo ""
echo "This screen session will remain active."
echo "Press Enter to exit or Ctrl+A, D to detach."
read
