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
SESSION1="tiny_gpu_150"
SESSION2="tiny_gpu_500"
SESSION3="tiny_gpu_750"

echo "Starting sequential tiny GPU experiments..."

# Function to wait for a screen session to finish
wait_for_screen_session() {
    local session_name=$1
    echo "Waiting for screen session '$session_name' to complete..."
    
    while screen -list | grep -q "$session_name"; do
        sleep 30  # Check every 30 seconds
        echo "Session '$session_name' is still running..."
    done
    
    echo "Session '$session_name' has completed."
}

# Start first experiment
echo "Starting first experiment: $CONFIG1"
screen -dmS "$SESSION1" bash -c "
    source $VENV_PATH && 
    python $SCRIPT_PATH --config $CONFIG1 --model $MODEL;
    echo 'First experiment (tiny_gpu_150) completed';
    read -p 'Press Enter to close this screen session...'
"

echo "First experiment started in screen session '$SESSION1'"
echo "You can attach to it with: screen -r $SESSION1"

# Wait for first experiment to complete
wait_for_screen_session "$SESSION1"

# Start second experiment
echo "Starting second experiment: $CONFIG2"
screen -dmS "$SESSION2" bash -c "
    source $VENV_PATH && 
    python $SCRIPT_PATH --config $CONFIG2 --model $MODEL;
    echo 'Second experiment (tiny_gpu_500) completed';
    read -p 'Press Enter to close this screen session...'
"

echo "Second experiment started in screen session '$SESSION2'"
echo "You can attach to it with: screen -r $SESSION2"

# Wait for second experiment to complete
wait_for_screen_session "$SESSION2"

# Start third experiment
echo "Starting third experiment: $CONFIG3"
screen -dmS "$SESSION3" bash -c "
    source $VENV_PATH && 
    python $SCRIPT_PATH --config $CONFIG3 --model $MODEL;
    echo 'Third experiment (tiny_gpu_750) completed';
    read -p 'Press Enter to close this screen session...'
"

echo "Third experiment started in screen session '$SESSION3'"
echo "You can attach to it with: screen -r $SESSION3"

echo "All three experiments have been queued sequentially."
echo "Experiment 1 (150 batches) has completed."
echo "Experiment 2 (500 batches) has completed."
echo "Experiment 3 (750 batches) is now running."
echo ""
echo "To monitor the current experiment:"
echo "  screen -r $SESSION3"
echo ""
echo "To list all screen sessions:"
echo "  screen -list"
echo ""
echo "Experiments will run with cosine LR schedule (1e-4 → 1e-6) and temperature (2.5 → 0.70)"
