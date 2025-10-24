#!/bin/bash

# Inventory all results files in the project

echo "=== PROJECT RESULTS INVENTORY ==="
echo "Generated: $(date)"
echo ""

echo "## Training GPU Results"
echo "### training_gpu/results/"
find training_gpu/results -name "*.csv" -o -name "*.json" | sort
echo ""

echo "### training_gpu_prefetch/results/"
find training_gpu_prefetch/results -name "*.csv" -o -name "*.json" | sort
echo ""

echo "## Training CPU Results"
echo "### training_cpu/results/"
find training_cpu/results -name "*.csv" -o -name "*.json" 2>/dev/null | sort
echo ""

echo "## Root-level Results"
echo "### results/"
find results -name "*.csv" -o -name "*.json" 2>/dev/null | sort
echo ""

echo "## Benchmark CPU Results"
echo "### benchmark_cpu/results/"
find benchmark_cpu/results -name "*.csv" -o -name "*.json" 2>/dev/null | head -30
echo ""

echo "## Benchmark GPU Results"
echo "### benchmark_gpu/results/"
find benchmark_gpu/results -name "*.csv" -o -name "*.json" 2>/dev/null | head -30
echo ""

echo "## Total Counts"
echo "CSV files: $(find . -name "*.csv" -not -path "./venv/*" | wc -l)"
echo "JSON files: $(find . -name "*.json" -not -path "./venv/*" -not -path "./.git/*" | wc -l)"
echo "PyTorch models: $(find . -name "*.pth" -not -path "./venv/*" | wc -l)"
echo ""

echo "## Directory Sizes"
du -sh training_gpu/results 2>/dev/null
du -sh training_gpu_prefetch/results 2>/dev/null
du -sh training_cpu/results 2>/dev/null
du -sh benchmark_cpu/results 2>/dev/null
du -sh benchmark_gpu/results 2>/dev/null
du -sh results 2>/dev/null

