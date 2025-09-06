#!/bin/bash
# Test runner for OR-Tools GLS parallel processing
# Created: 2025-09-06

echo "========================================================================"
echo "OR-Tools GLS Parallel Test Runner"
echo "========================================================================"
echo "Date: $(date)"
echo "Directory: $(pwd)"
echo ""

# Check Python environment
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
    echo "Using virtual environment Python: $PYTHON"
else
    PYTHON="python3"
    echo "Using system Python: $PYTHON"
fi

# Verify Python version
echo "Python version:"
$PYTHON --version
echo ""

# Check if OR-Tools is installed
echo "Checking OR-Tools installation..."
$PYTHON -c "from ortools.constraint_solver import pywrapcp; print('OR-Tools is installed ✓')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: OR-Tools is not installed!"
    echo "Install with: pip install ortools"
    exit 1
fi
echo ""

# Check if required script exists
if [ ! -f "benchmark_gpu/benchmark_cpu/scripts/ortools/benchmarks/benchmark_ortools_gls_fixed.py" ]; then
    echo "ERROR: Required script benchmark_ortools_gls_fixed.py not found!"
    exit 1
fi

# Menu for test selection
echo "Select test type:"
echo "1) Quick test (N=10,20 with 2s timeout, 4 threads total)"
echo "2) Small test (N=10,20,50 with 2s timeout, 6 threads)"
echo "3) Medium test (N=10,20,50,100 with 5s timeout, 8 threads)"
echo "4) Full parallel test (original configuration)"
echo "5) Custom configuration"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Running quick test..."
        $PYTHON test_ortools_parallel.py
        ;;
    2)
        echo "Running small test..."
        cat > small_test.py << 'SMALLTEST'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from test_ortools_parallel import run_thread_instances
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os

configs = [
    (10, 20, 2, 2),   # N=10
    (20, 30, 2, 2),   # N=20
    (50, 40, 2, 2),   # N=50
]

tasks = []
test_output_base = 'benchmark_gpu/results/ortools_test_runs'
os.makedirs(test_output_base, exist_ok=True)

for n, capacity, timeout, num_threads in configs:
    output_dir = f'{test_output_base}/n{n}_t{timeout}s_test'
    for thread_id in range(num_threads):
        tasks.append((n, capacity, timeout, thread_id, num_threads, 3, output_dir))

print(f"Running {len(tasks)} tasks...")
with ProcessPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(run_thread_instances, task) for task in tasks]
    for future in futures:
        future.result()
print("Small test complete!")
SMALLTEST
        $PYTHON small_test.py
        rm small_test.py
        ;;
    3)
        echo "Running medium test..."
        cat > medium_test.py << 'MEDTEST'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from test_ortools_parallel import run_thread_instances
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os
import time

configs = [
    (10, 20, 5, 2),   # N=10
    (20, 30, 5, 2),   # N=20
    (50, 40, 5, 2),   # N=50
    (100, 50, 5, 2),  # N=100
]

tasks = []
test_output_base = 'benchmark_gpu/results/ortools_test_runs'
os.makedirs(test_output_base, exist_ok=True)

start_time = time.time()
for n, capacity, timeout, num_threads in configs:
    output_dir = f'{test_output_base}/n{n}_t{timeout}s_test'
    for thread_id in range(num_threads):
        tasks.append((n, capacity, timeout, thread_id, num_threads, 5, output_dir))

print(f"Running {len(tasks)} tasks with 5 instances each...")
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(run_thread_instances, task) for task in tasks]
    for i, future in enumerate(futures):
        result = future.result()
        print(f"Task {i+1}/{len(tasks)} completed")

elapsed = time.time() - start_time
print(f"Medium test complete in {elapsed:.2f} seconds!")
MEDTEST
        $PYTHON medium_test.py
        rm medium_test.py
        ;;
    4)
        echo "Running full parallel test..."
        if [ -f "benchmark_gpu/scripts/run_ortools_gls.py" ]; then
            cd benchmark_gpu
            $PYTHON scripts/run_ortools_gls.py
            cd ..
        else
            echo "ERROR: Full test script not found!"
            exit 1
        fi
        ;;
    5)
        echo "Custom configuration"
        read -p "Enter N (number of nodes): " n
        read -p "Enter capacity: " capacity
        read -p "Enter timeout (seconds): " timeout
        read -p "Enter number of instances: " instances
        read -p "Enter number of threads: " threads
        
        echo "Running custom test: N=$n, C=$capacity, T=$timeout, I=$instances, Threads=$threads"
        
        cat > custom_test.py << CUSTOMTEST
#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from test_ortools_parallel import run_thread_instances
from concurrent.futures import ProcessPoolExecutor
import os

test_output_base = 'benchmark_gpu/results/ortools_test_runs'
os.makedirs(test_output_base, exist_ok=True)
output_dir = f'{test_output_base}/custom_n${n}_t${timeout}s'

tasks = []
for thread_id in range($threads):
    tasks.append(($n, $capacity, $timeout, thread_id, $threads, $instances, output_dir))

print(f"Running custom configuration with $threads threads...")
with ProcessPoolExecutor(max_workers=$threads) as executor:
    futures = [executor.submit(run_thread_instances, task) for task in tasks]
    for future in futures:
        future.result()
print("Custom test complete!")
CUSTOMTEST
        $PYTHON custom_test.py
        rm custom_test.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Test complete! Check results in:"
echo "  benchmark_gpu/results/ortools_test_runs/"
echo ""
echo "To view results:"
echo "  ls -la benchmark_gpu/results/ortools_test_runs/"
echo "========================================================================"
