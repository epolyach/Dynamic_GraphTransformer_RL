#!/bin/bash
echo "GPU Optimal Solver Benchmark Monitor"
echo "====================================="
echo "Started: $(date)"
echo ""

# Check if process is running
PID=$(ps aux | grep -E "python.*benchmark_gpu_truly_optimal" | grep -v grep | grep -v SCREEN | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "Process Status: RUNNING (PID: $PID)"
    
    # Check GPU usage
    echo ""
    echo "GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv | tail -1
    
    # Check screen output
    echo ""
    echo "Latest Output:"
    screen -S gpu_optimal_n10_c30_1000 -X hardcopy /tmp/gpu_screen_monitor.txt 2>/dev/null
    if [ -f /tmp/gpu_screen_monitor.txt ]; then
        tail -5 /tmp/gpu_screen_monitor.txt | grep -v "^$"
    fi
    
    # Estimate completion
    echo ""
    echo "Estimated completion time for 1000 instances:"
    echo "  Based on ~1.5s per instance: ~25 minutes"
    echo "  Current runtime: $(ps -o etime= -p $PID | xargs)"
else
    echo "Process Status: NOT RUNNING"
    echo "The benchmark may have completed or stopped."
fi

echo ""
echo "To attach to screen session: screen -r gpu_optimal_n10_c30_1000"
echo "To detach from screen: Ctrl-A then D"
