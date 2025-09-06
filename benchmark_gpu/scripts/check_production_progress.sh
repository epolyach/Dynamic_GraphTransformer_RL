#!/bin/bash

echo "======================================================================="
echo "OR-Tools GLS Production Benchmark Progress"
echo "Time: $(date)"
echo "======================================================================="
echo

# Check running processes
echo "Running processes:"
ps aux | grep -E "run_ortools_gls_production|benchmark_ortools" | grep -v grep | wc -l
echo

# Check completed batches in each directory
echo "Progress by configuration:"
echo

for timeout in 2 5; do
    dir="results/ortools_gls_${timeout}s_production"
    if [ -d "$dir" ]; then
        echo "Timeout ${timeout}s:"
        for n in 10 20 50 100; do
            checkpoint="$dir/checkpoint_n${n}_timeout${timeout}.json"
            if [ -f "$checkpoint" ]; then
                completed=$(grep -o '"completed_batches": \[[^]]*\]' "$checkpoint" | grep -o '[0-9]' | wc -l)
                instances=$((completed * 100))
                percent=$((instances * 100 / 10000))
                echo "  N=$n: $instances/10000 ($percent%)"
            else
                echo "  N=$n: Not started"
            fi
        done
        echo
    fi
done

# Check latest log entries
echo "Latest log entries:"
tail -5 results/ortools_production_log.txt
echo

# Estimate time remaining (rough)
echo "Note: Each batch of 100 instances takes approximately:"
echo "  - 2s timeout: ~3.5 minutes"
echo "  - 5s timeout: ~8.5 minutes"
echo "Total estimated time: 10-15 hours with 8 parallel threads"
