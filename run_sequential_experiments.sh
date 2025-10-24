#!/bin/bash

# Script to run two experiments sequentially in screen sessions
# Each experiment will run in its own screen session

set -e  # Exit on any error

# Configuration
VENV_PATH="venv/bin/activate"
SCRIPT_PATH="training_gpu/scripts/run_training_gpu.py"
MODEL="GT+RL"

# Experiment configurations
CONFIG1="configs/experiment_rollout_only.yaml"
CONFIG2="configs/medium_experiment_hybrid_50.yaml"

# Screen session names
SESSION1="experiment_rollout_only"
SESSION2="medium_experiment_hybrid_50"

echo "Starting sequential experiments..."

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
    echo 'First experiment completed';
    read -p 'Press Enter to close this screen session...'
"

echo "First experiment started in screen session '$SESSION1'"
echo "You can attach to it with: screen -r $SESSION1"

# Wait for first experiment to complete
wait_for_screen_session "$SESSION1"

echo "Starting second experiment: $CONFIG2"
screen -dmS "$SESSION2" bash -c "
    source $VENV_PATH && 
    python $SCRIPT_PATH --config $CONFIG2 --model $MODEL;
    echo 'Second experiment completed';
    read -p 'Press Enter to close this screen session...'
"

echo "Second experiment started in screen session '$SESSION2'"
echo "You can attach to it with: screen -r $SESSION2"

echo "Both experiments have been queued sequentially."
echo "First experiment has completed, second experiment is now running."
echo ""
echo "To monitor the current experiment:"
echo "  screen -r $SESSION2"
echo ""
echo "To list all screen sessions:"
echo "  screen -list"

