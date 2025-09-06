#!/bin/bash
# Script to organize results directories
# Created: 2025-09-06

echo "Creating organized directory structure..."

# Create structure for benchmark_gpu
mkdir -p benchmark_gpu/results/plots
mkdir -p benchmark_gpu/results/tables
mkdir -p benchmark_gpu/results/data
mkdir -p benchmark_gpu/results/logs
mkdir -p benchmark_gpu/scripts/plotting
mkdir -p benchmark_gpu/scripts/table_generation
mkdir -p benchmark_gpu/scripts/archive

# Create structure for training_cpu
mkdir -p training_cpu/results/plots
mkdir -p training_cpu/results/tables
mkdir -p training_cpu/results/data
mkdir -p training_cpu/results/logs
mkdir -p training_cpu/scripts/ortools
mkdir -p training_cpu/scripts/ortools/production
mkdir -p training_cpu/scripts/ortools/benchmarks
mkdir -p training_cpu/scripts/ortools/monitoring
mkdir -p training_cpu/scripts/plotting
mkdir -p training_cpu/scripts/table_generation

echo "Moving files to organized locations..."

# Move plot files from benchmark_gpu/scripts
echo "Moving plot files..."
if ls benchmark_gpu/scripts/*.png 2>/dev/null; then
    mv benchmark_gpu/scripts/*.png benchmark_gpu/results/plots/
fi
if ls benchmark_gpu/scripts/*.eps 2>/dev/null; then
    mv benchmark_gpu/scripts/*.eps benchmark_gpu/results/plots/
fi

# Move tex files
echo "Moving LaTeX tables..."
if ls benchmark_gpu/scripts/*.tex 2>/dev/null; then
    mv benchmark_gpu/scripts/*.tex benchmark_gpu/results/tables/
fi
if ls benchmark_gpu/results/*.tex 2>/dev/null; then
    mv benchmark_gpu/results/*.tex benchmark_gpu/results/tables/
fi

# Move data files
echo "Moving data files..."
if ls benchmark_gpu/scripts/*.csv 2>/dev/null; then
    mv benchmark_gpu/scripts/*.csv benchmark_gpu/results/data/
fi
if ls benchmark_gpu/scripts/*.txt 2>/dev/null; then
    mv benchmark_gpu/scripts/*.txt benchmark_gpu/results/logs/
fi
if ls benchmark_gpu/scripts/*.out 2>/dev/null; then
    mv benchmark_gpu/scripts/*.out benchmark_gpu/results/logs/
fi
if ls benchmark_gpu/results/*.txt 2>/dev/null; then
    mv benchmark_gpu/results/*.txt benchmark_gpu/results/logs/
fi
if ls benchmark_gpu/results/*.json 2>/dev/null; then
    mv benchmark_gpu/results/*.json benchmark_gpu/results/data/
fi

# Move backup files to archive
echo "Archiving backup files..."
if ls benchmark_gpu/scripts/*.backup 2>/dev/null; then
    mv benchmark_gpu/scripts/*.backup benchmark_gpu/scripts/archive/
fi
if ls benchmark_gpu/scripts/*_backup.py 2>/dev/null; then
    mv benchmark_gpu/scripts/*_backup.py benchmark_gpu/scripts/archive/
fi

# Move plotting scripts
echo "Organizing plotting scripts..."
for script in plot_*.py make_*.py compare_*.py; do
    if [ -f "benchmark_gpu/scripts/$script" ]; then
        cp "benchmark_gpu/scripts/$script" "benchmark_gpu/scripts/plotting/"
    fi
done

# Move table generation scripts
echo "Organizing table generation scripts..."
for script in generate_*.py; do
    if [ -f "benchmark_gpu/scripts/$script" ]; then
        cp "benchmark_gpu/scripts/$script" "benchmark_gpu/scripts/table_generation/"
    fi
done

echo "Organization complete!"
echo ""
echo "Summary of new structure:"
echo "├── benchmark_gpu/"
echo "│   ├── results/"
echo "│   │   ├── plots/       (*.png, *.eps)"
echo "│   │   ├── tables/      (*.tex)"
echo "│   │   ├── data/        (*.csv, *.json)"
echo "│   │   └── logs/        (*.txt, *.out)"
echo "│   └── scripts/"
echo "│       ├── plotting/"
echo "│       ├── table_generation/"
echo "│       └── archive/"
echo "└── training_cpu/"
echo "    ├── results/"
echo "    │   ├── plots/"
echo "    │   ├── tables/"
echo "    │   ├── data/"
echo "    │   └── logs/"
echo "    └── scripts/"
echo "        └── ortools/"
echo "            ├── production/"
echo "            ├── benchmarks/"
echo "            └── monitoring/"
