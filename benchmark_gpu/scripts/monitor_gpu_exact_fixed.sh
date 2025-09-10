#!/bin/bash
echo "GPU Optimal Exact Solver (FIXED) - Benchmark Monitor"
echo "===================================================="
echo "Started: $(date)"
echo ""

# Check if process is running
PID=$(ps aux | grep -E "python.*benchmark_gpu_truly_optimal_n10.py.*1000.*30" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "Process Status: ✅ RUNNING (PID: $PID)"
    echo "Configuration: N=10, C=30, 1000 instances"
    
    # Check GPU usage
    echo ""
    echo "GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv | tail -1
    
    # Check screen output
    echo ""
    echo "Latest Output:"
    screen -S gpu_exact_n10_c30_1000_fixed -X hardcopy /tmp/gpu_exact_monitor.txt 2>/dev/null
    if [ -f /tmp/gpu_exact_monitor.txt ]; then
        tail -10 /tmp/gpu_exact_monitor.txt | grep -E "(Batch|time|CPC|Vehicles)" | tail -5
    fi
    
    # Estimate completion
    echo ""
    echo "Performance Estimate:"
    echo "  Based on ~0.7s per instance: ~12 minutes total"
    echo "  Current runtime: $(ps -o etime= -p $PID | xargs)"
    
    # Progress estimation
    if [ -f /tmp/gpu_exact_monitor.txt ]; then
        CURRENT_BATCH=$(grep -E "Batch [0-9]+/100" /tmp/gpu_exact_monitor.txt | tail -1 | sed -n 's/.*Batch \([0-9]*\)\/100.*/\1/p')
        if [ -n "$CURRENT_BATCH" ]; then
            PROGRESS=$((CURRENT_BATCH * 10))
            PERCENT=$((CURRENT_BATCH))
            echo "  Progress: ${PROGRESS}/1000 instances (${PERCENT}%)"
            REMAINING=$((100 - CURRENT_BATCH))
            ETA_MIN=$((REMAINING * 7 / 10))
            echo "  ETA: ~${ETA_MIN} minutes remaining"
        fi
    fi
else
    echo "Process Status: ❌ NOT RUNNING"
    echo "The benchmark may have completed or stopped."
    
    # Check for results file
    LATEST_RESULT=$(ls -t gpu_exact_n10_results_*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        echo ""
        echo "Latest Results File: $LATEST_RESULT"
        echo "File size: $(ls -lh $LATEST_RESULT | awk '{print $5}')"
        echo "Lines: $(wc -l < $LATEST_RESULT)"
        echo ""
        echo "Sample results:"
        head -3 $LATEST_RESULT
    fi
fi

echo ""
echo "Commands:"
echo "  Attach to screen: screen -r gpu_exact_n10_c30_1000_fixed"
echo "  Detach from screen: Ctrl-A then D"
echo "  Check logs: tail -f /tmp/gpu_exact_monitor.txt"
