#!/bin/bash

echo "========================================================================"
echo "OR-TOOLS GLS BENCHMARKS STATUS CHECK"
echo "========================================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Function to check process and estimate progress
check_benchmark() {
    local n=$1
    local instances=$2
    local script_pattern=$3
    
    # Check if process is running
    pid=$(pgrep -f "$script_pattern" | head -1)
    
    if [ ! -z "$pid" ]; then
        # Get CPU usage
        cpu=$(ps -p $pid -o %cpu= | tr -d ' ')
        
        # Get runtime
        runtime=$(ps -p $pid -o etime= | tr -d ' ')
        
        echo "N=$n ($instances inst): ⏳ RUNNING | PID: $pid | CPU: ${cpu}% | Time: $runtime"
        
        # Check for result file
        result_file=$(ls -t ortools_gls_n${n}_${instances}inst_*.json 2>/dev/null | head -1)
        if [ ! -z "$result_file" ]; then
            echo "  └─ Result available: $result_file"
        fi
    else
        # Check if completed
        result_file=$(ls -t ortools_gls_n${n}_${instances}inst_*.json 2>/dev/null | head -1)
        if [ ! -z "$result_file" ]; then
            echo "N=$n ($instances inst): ✓ COMPLETED | Result: $result_file"
        else
            echo "N=$n ($instances inst): ⏸ NOT RUNNING"
        fi
    fi
}

echo "BENCHMARK STATUS:"
echo "----------------------------------------"

# Check N=10
check_benchmark 10 10000 "benchmark_ortools_multi_n_fixed.py.*--n 10"

# Check N=20
check_benchmark 20 10000 "benchmark_ortools_multi_n_fixed.py.*--n 20"

# Check N=50
check_benchmark 50 10000 "benchmark_ortools_multi_n_fixed.py.*--n 50"

# Check N=100 (1000 instances)
pid=$(pgrep -f "benchmark_ortools_gls_fixed.py.*--instances 1000" | head -1)
if [ ! -z "$pid" ]; then
    cpu=$(ps -p $pid -o %cpu= | tr -d ' ')
    runtime=$(ps -p $pid -o etime= | tr -d ' ')
    echo "N=100 (1000 inst):  ⏳ RUNNING | PID: $pid | CPU: ${cpu}% | Time: $runtime"
else
    result_file=$(ls -t ortools_gls_n100_1000inst_*.json 2>/dev/null | head -1)
    if [ ! -z "$result_file" ]; then
        echo "N=100 (1000 inst):  ✓ COMPLETED | Result: $result_file"
    else
        echo "N=100 (1000 inst):  ⏸ NOT RUNNING"
    fi
fi

# Check N=100 (10000 instances)
pid=$(pgrep -f "benchmark_ortools_gls_fixed.py.*--instances 10000" | head -1)
if [ ! -z "$pid" ]; then
    cpu=$(ps -p $pid -o %cpu= | tr -d ' ')
    runtime=$(ps -p $pid -o etime= | tr -d ' ')
    echo "N=100 (10000 inst): ⏳ RUNNING | PID: $pid | CPU: ${cpu}% | Time: $runtime"
else
    result_file=$(ls -t ortools_gls_n100_10000inst_*.json 2>/dev/null | head -1)
    if [ ! -z "$result_file" ]; then
        echo "N=100 (10000 inst): ✓ COMPLETED | Result: $result_file"
    else
        echo "N=100 (10000 inst): ⏸ NOT RUNNING"
    fi
fi

echo ""
echo "========================================================================"
echo "ESTIMATED COMPLETION TIMES (based on 2s/instance for 10k runs):"
echo "----------------------------------------"
echo "N=10:  ~333 minutes (~5.5 hours)"
echo "N=20:  ~333 minutes (~5.5 hours)" 
echo "N=50:  ~333 minutes (~5.5 hours)"
echo "N=100: ~333 minutes (~5.5 hours) for 10k instances"
echo "N=100: ~17 minutes for 1k instances"
echo "========================================================================"
