#!/bin/bash

echo "========================================="
echo "Script to run remaining medium GPU trainings"
echo "========================================="
echo ""
echo "This script will run training_medium_annealing and training_medium_optimal"
echo "sequentially after training_medium_large_model completes."
echo ""

# Function to check if large model is still running
check_large_model_running() {
    screen -ls | grep -q "training_medium_large_model"
    return $?
}

# Wait for large model to complete
if check_large_model_running; then
    echo "Waiting for training_medium_large_model to complete..."
    echo "You can check its progress with: screen -r training_medium_large_model"
    echo ""
    
    while check_large_model_running; do
        sleep 60  # Check every minute
    done
    
    echo "training_medium_large_model has completed!"
fi

# Now start the remaining trainings sequentially
screen -dmS training_medium_remaining bash -c '
cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL
source venv/bin/activate

echo "========================================="
echo "Starting Medium GPU Annealing Training"
echo "========================================="
python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_annealing.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_annealing_new.log

echo ""
echo "========================================="
echo "Starting Medium GPU Optimal Training"
echo "========================================="
python3 training_gpu/scripts/run_training_gpu.py --config configs/medium_gpu_optimal.yaml --model GT+RL --device cuda:0 --force-retrain 2>&1 | tee training_medium_optimal_new.log

echo ""
echo "========================================="
echo "All remaining training sessions completed!"
echo "========================================="
'

echo "Remaining trainings started in screen session 'training_medium_remaining'"
echo ""
echo "Commands:"
echo "  View progress:     screen -r training_medium_remaining"
echo "  Detach:           Ctrl+A, then D"
echo "  List screens:     screen -ls"
echo "  Kill if needed:   screen -X -S training_medium_remaining quit"
echo ""
echo "Log files:"
echo "  - training_medium_annealing_new.log"
echo "  - training_medium_optimal_new.log"
