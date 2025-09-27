#!/bin/bash

# General script to run GPU training configurations sequentially
# Usage: ./run_seq.sh config1.yaml config2.yaml config3.yaml ...
# Example: ./run_seq.sh configs/tiny_gpu_150.yaml configs/tiny_gpu_500.yaml &

set -e  # Exit on any error

# Check if configs are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 config1.yaml config2.yaml [config3.yaml ...]"
    echo "Example: $0 configs/tiny_gpu_150.yaml configs/tiny_gpu_500.yaml"
    exit 1
fi

# Configuration
VENV_PATH="venv/bin/activate"
SCRIPT_PATH="training_gpu/scripts/run_training_gpu.py"
MODEL="GT+RL"
LOG_FILE="sequential_training_$(date +%Y%m%d_%H%M%S).log"

# Create log file
echo "Sequential Training Log - Started at $(date)" > "$LOG_FILE"
echo "Command: $0 $*" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Function to log with timestamp
log_msg() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# Function to run a single experiment
run_experiment() {
    local config_file="$1"
    local experiment_num="$2"
    local total_experiments="$3"
    
    # Extract experiment name from config path
    local exp_name=$(basename "$config_file" .yaml)
    local session_name="seq_${exp_name}_$$"
    
    log_msg "Starting experiment $experiment_num/$total_experiments: $config_file"
    log_msg "Screen session: $session_name"
    
    # Start experiment in screen session
    screen -dmS "$session_name" bash -c "
        cd $(pwd) &&
        source $VENV_PATH && 
        echo 'Starting training for $config_file at \$(date)' &&
        python $SCRIPT_PATH --config $config_file --model $MODEL --force-retrain 2>&1 | tee -a $LOG_FILE;
        echo 'Experiment $exp_name completed at \$(date)' | tee -a $LOG_FILE;
        sleep 2
    "
    
    log_msg "Experiment started in screen session '$session_name'"
    log_msg "Waiting for completion..."
    
    # Wait for screen session to finish
    while screen -list | grep -q "$session_name"; do
        sleep 60  # Check every minute
        log_msg "Experiment $experiment_num ($exp_name) still running..."
    done
    
    log_msg "Experiment $experiment_num ($exp_name) completed!"
    echo "" >> "$LOG_FILE"
}

# Main execution
log_msg "Starting sequential training of $# configurations"
log_msg "Process ID: $$"
log_msg "Log file: $LOG_FILE"

# Run each configuration sequentially
for i in $(seq 1 $#); do
    config_file="${!i}"
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        log_msg "ERROR: Config file not found: $config_file"
        exit 1
    fi
    
    run_experiment "$config_file" "$i" "$#"
done

log_msg "ALL EXPERIMENTS COMPLETED!"
log_msg "Results can be found in training_gpu/results/"
log_msg "Full log available in: $LOG_FILE"

# Summary
echo ""
echo "=========================================="
echo "SEQUENTIAL TRAINING SUMMARY"
echo "=========================================="
echo "Started: $(head -1 "$LOG_FILE" | cut -d'-' -f2-)"
echo "Completed: $(date)"
echo "Configurations processed: $#"
echo "Log file: $LOG_FILE"
echo ""
echo "To view results:"
for i in $(seq 1 $#); do
    config_file="${!i}"
    exp_name=$(basename "$config_file" .yaml | sed 's/configs\///')
    echo "  - training_gpu/results/$exp_name/"
done
