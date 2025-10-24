#!/bin/bash

# Script to run three tiny GPU experiments sequentially in screen sessions
# Each experiment will run in its own screen session

set -e  # Exit on any error

# Configuration
VENV_PATH="venv/bin/activate"
SCRIPT_PATH="training_gpu/scripts/run_training_gpu.py"
MODEL="GT+RL"

# Experiment configurations
CONFIG1="configs/tiny_gpu_150.yaml"
CONFIG2="configs/tiny_gpu_500.yaml"
CONFIG3="configs/tiny_gpu_750.yaml"

# Screen session names
SESSION1="tiny_gpu_150_fixed"
SESSION2="tiny_gpu_500_fixed"
SESSION3="tiny_gpu_750_fixed"

echo "Starting sequential tiny GPU experiments (fixed version)..."

# Function to wait for a screen session to finish
wait_for_screen_session() {
    local session_name=$1
    echo "Waiting for screen session '$session_name' to complete..."
    
    while screen -list | grep -q "$session_name"; do
        sleep 60  # Check every 60 seconds
        echo "$(date): Session '$session_name' is still running..."
    done
    
    echo "$(date): Session '$session_name' has completed."
}

# Start first experiment
echo "$(date): Starting first experiment: $CONFIG1"
screen -dmS "$SESSION1" bash -c "
    source $VENV_PATH && 
    python $SCRIPT_PATH --config $CONFIG1 --model $MODEL --force-retrain;
    echo 'First experiment (tiny_gpu_150) completed at $(date)';
    sleep 2
"

echo "First experiment started in screen session '$SESSION1'"
echo "You can attach to it with: screen -r $SESSION1"

# Wait for first experiment to complete
wait_for_screen_session "$SESSION1"

# Start second experiment
echo "$(date): Starting second experiment: $CONFIG2"
screen -dmS "$SESSION2" bash -c "
    source $VENV_PATH && 
    python $SCRIPT_PATH --config $CONFIG2 --model $MODEL --force-retrain;
    echo 'Second experiment (tiny_gpu_500) completed at $(date)';
    sleep 2
"

echo "Second experiment started in screen session '$SESSION2'"
echo "You can attach to it with: screen -r $SESSION2"

# Wait for second experiment to complete
wait_for_screen_session "$SESSION2"

# Start third experiment
echo "$(date): Starting third experiment: $CONFIG3"
screen -dmS "$SESSION3" bash -c "
    source $VENV_PATH && 
    python $SCRIPT_PATH --config $CONFIG3 --model $MODEL --force-retrain;
    echo 'Third experiment (tiny_gpu_750) completed at $(date)';
    sleep 2
"

echo "Third experiment started in screen session '$SESSION3'"
echo "You can attach to it with: screen -r $SESSION3"

# Wait for third experiment to complete
wait_for_screen_session "$SESSION3"

echo "$(date): ALL THREE EXPERIMENTS COMPLETED!"
echo ""
echo "Results are available in:"
echo "  - training_gpu/results/tiny_gpu_150/"
echo "  - training_gpu/results/tiny_gpu_500/"
echo "  - training_gpu/results/tiny_gpu_750/"
echo ""
echo "All experiments ran with full 100 epochs (early stopping disabled)"
