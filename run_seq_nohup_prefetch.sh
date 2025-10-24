#!/bin/bash

# Wrapper to run sequential PREFETCH training with nohup (survives SSH disconnection)
# Usage: ./run_seq_nohup_prefetch.sh config1.yaml config2.yaml config3.yaml ...

if [ $# -eq 0 ]; then
    echo "Usage: $0 config1.yaml config2.yaml [config3.yaml ...]"
    echo "Example: $0 configs/tiny_gpu_150.yaml configs/tiny_gpu_500.yaml configs/tiny_gpu_750.yaml"
    exit 1
fi

# Generate unique log file name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NOHUP_LOG="nohup_sequential_prefetch_${TIMESTAMP}.log"

echo "Starting sequential PREFETCH training with nohup..."
echo "Configs: $*"
echo "Nohup log: $NOHUP_LOG"
echo ""

# Run with nohup to survive SSH disconnection
nohup ./run_seq_prefetch.sh "$@" > "$NOHUP_LOG" 2>&1 &

# Get the process ID
PID=$!

echo "Sequential PREFETCH training started in background!"
echo "Process ID: $PID"
echo "Nohup log file: $NOHUP_LOG"
echo ""
echo "To monitor progress:"
echo "  tail -f $NOHUP_LOG"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo "  screen -ls"
echo ""
echo "To kill if needed:"
echo "  kill $PID"
echo ""
echo "The process will continue even if you close SSH connection."

# Save PID for reference
echo $PID > "sequential_training_prefetch_${TIMESTAMP}.pid"
echo "PID saved to: sequential_training_prefetch_${TIMESTAMP}.pid"
