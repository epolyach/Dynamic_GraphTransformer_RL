#!/bin/bash

# Sequential Training Script - Runs configs one after another in screen
# Usage: ./run_sequential_training.sh

CONFIGS=(
    "configs/tiny_gpu_512_optimal.yaml"
    "configs/tiny_gpu_512_optimal_T20.yaml"
)

MODEL="GT+RL"
DEVICE="cuda:0"
BASE_DIR="/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL"

echo "=== Sequential Training Script ==="
echo "Configs to run:"
for i in "${!CONFIGS[@]}"; do
    echo "  $((i+1)). ${CONFIGS[$i]}"
done
echo ""

# Function to wait for screen session to complete
wait_for_completion() {
    local session_name=$1
    local config_name=$2
    
    echo "Waiting for $session_name to complete..."
    
    while screen -list | grep -q "$session_name"; do
        # Check if session is still running
        if screen -S "$session_name" -X select . 2>/dev/null; then
            # Show progress every 60 seconds
            sleep 60
            echo -n "."
        else
            # Session ended
            break
        fi
    done
    
    echo ""
    echo "$config_name training completed!"
}

# Kill any existing training sessions
echo "Cleaning up existing sessions..."
screen -ls | grep "training_seq_" | awk '{print $1}' | while read session; do
    screen -X -S "$session" quit 2>/dev/null
done

# Run each config sequentially
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    session_name="training_seq_$((i+1))"
    config_base=$(basename "$config" .yaml)
    
    echo ""
    echo "=== Starting config $((i+1))/${#CONFIGS[@]}: $config ==="
    echo "Session: $session_name"
    echo "Log: ${config_base}_training.log"
    
    # Start training in screen
    screen -dmS "$session_name" bash -c "
        cd $BASE_DIR
        echo 'Starting training with $config at $(date)'
        python3 training_gpu/scripts/run_training_gpu.py \
            --config '$config' \
            --model '$MODEL' \
            --device '$DEVICE' \
            --force-retrain \
            2>&1 | tee '${config_base}_training.log'
        echo 'Training completed at $(date)'
    "
    
    echo "Training started in screen session: $session_name"
    sleep 5  # Give it a moment to initialize
    
    # Wait for this training to complete before starting the next
    wait_for_completion "$session_name" "$config_base"
done

echo ""
echo "=== All training sessions completed! ==="
echo ""
echo "Check logs:"
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    config_base=$(basename "$config" .yaml)
    echo "  ${config_base}_training.log"
done
