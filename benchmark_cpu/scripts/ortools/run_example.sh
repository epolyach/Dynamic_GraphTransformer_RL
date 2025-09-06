#!/bin/bash
# Example usage of run_ortools_gls.py with CLI arguments

# Quick test - N=10, 20 instances, 2 threads, 5s timeout
echo "Running quick test..."
python3 production/run_ortools_gls.py \
    --subfolder "test_n10_quick" \
    --n 10 \
    --instances 20 \
    --timeout 5 \
    --threads 2

# Medium test - N=20, 100 instances, 4 threads, 10s timeout
echo "Running medium test..."
python3 production/run_ortools_gls.py \
    --subfolder "test_n20_medium" \
    --n 20 \
    --instances 100 \
    --timeout 10 \
    --threads 4

# Production run - N=50, 1000 instances, 8 threads, 30s timeout
echo "Running production benchmark..."
python3 production/run_ortools_gls.py \
    --subfolder "production_n50" \
    --n 50 \
    --instances 1000 \
    --timeout 30 \
    --threads 8 \
    --capacity 40  # Optional: specify capacity

echo "All benchmarks complete!"
echo "Check results in benchmark_cpu/results/"
