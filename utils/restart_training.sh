#!/bin/bash

echo "==============================================="
echo "Restarting Training with 16-Worker Config"
echo "==============================================="
echo ""

# Find the old training process
OLD_PID=$(ps aux | grep "run_training_gpu.py" | grep "evgeny.polyachenko" | grep -v grep | awk '{print $2}')

if [ ! -z "$OLD_PID" ]; then
    echo "Found old training process: PID $OLD_PID"
    echo "Stopping old training..."
    kill $OLD_PID
    sleep 2
    
    # Check if it's still running (force kill if needed)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Process still running, force killing..."
        kill -9 $OLD_PID
        sleep 1
    fi
    echo "✓ Old training stopped"
else
    echo "No old training process found"
fi

# Kill any remaining worker processes
echo ""
echo "Cleaning up worker processes..."
pkill -f "multiprocessing.spawn.*evgeny.polyachenko" 2>/dev/null
sleep 1
echo "✓ Workers cleaned up"

# Kill old screen session
echo ""
echo "Cleaning up old screen session..."
screen -S seq_tiny_gpu_1000_1683732 -X quit 2>/dev/null
echo "✓ Screen session cleaned up"

echo ""
echo "==============================================="
echo "Starting NEW training with 16 workers..."
echo "==============================================="
echo ""

# Verify config
echo "Configuration check:"
grep "num_workers:" configs/tiny_gpu_1000.yaml
echo ""

# Start new training
echo "Launching training..."
./run_seq_nohup.sh configs/tiny_gpu_1000.yaml

echo ""
echo "==============================================="
echo "Training restarted!"
echo "==============================================="
echo ""
echo "The new training will use 16 workers (instead of 6)"
echo ""
echo "Monitor with:"
echo "  tail -f nohup_sequential_*.log"
echo "  htop  # Should see 17 Python processes when generating data"
echo "  nvtop # GPU usage"
echo ""

