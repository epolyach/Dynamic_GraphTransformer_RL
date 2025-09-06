#!/bin/bash
# Migration Plan for CPU Scripts to training_cpu/
# Generated: 2025-09-06
# WARNING: Review before executing!

# Create backup first
echo "Creating backup of scripts directory..."
tar -czf benchmark_gpu_scripts_backup_$(date +%Y%m%d_%H%M%S).tar.gz benchmark_gpu/scripts/

# Create target directory structure
echo "Creating target directories..."
mkdir -p training_cpu/scripts/ortools
mkdir -p training_cpu/scripts/ortools/production
mkdir -p training_cpu/scripts/ortools/monitoring
mkdir -p training_cpu/scripts/ortools/benchmarks

# Move OR-Tools production scripts
echo "Moving OR-Tools production scripts..."
mv benchmark_gpu/scripts/run_ortools_gls_parallel_test.py training_cpu/scripts/ortools/production/
mv benchmark_gpu/scripts/run_ortools_gls_production.py training_cpu/scripts/ortools/production/
mv benchmark_gpu/scripts/run_ortools_gls_extras.py training_cpu/scripts/ortools/production/
mv benchmark_gpu/scripts/run_ortools_gls_test.py training_cpu/scripts/ortools/production/

# Move OR-Tools benchmark scripts
echo "Moving OR-Tools benchmark scripts..."
mv benchmark_gpu/scripts/benchmark_ortools_gls.py training_cpu/scripts/ortools/benchmarks/
mv benchmark_gpu/scripts/benchmark_ortools_gls_fixed.py training_cpu/scripts/ortools/benchmarks/
mv benchmark_gpu/scripts/benchmark_ortools_multi_n.py training_cpu/scripts/ortools/benchmarks/
mv benchmark_gpu/scripts/benchmark_ortools_multi_n_fixed.py training_cpu/scripts/ortools/benchmarks/

# Move OR-Tools monitoring scripts
echo "Moving OR-Tools monitoring scripts..."
mv benchmark_gpu/scripts/monitor_all_ortools.py training_cpu/scripts/ortools/monitoring/
mv benchmark_gpu/scripts/monitor_all_ortools_backup.py training_cpu/scripts/ortools/monitoring/
mv benchmark_gpu/scripts/monitor_ortools_fixed_n100.py training_cpu/scripts/ortools/monitoring/
mv benchmark_gpu/scripts/continuous_monitor_ortools.py training_cpu/scripts/ortools/monitoring/
mv benchmark_gpu/scripts/continuous_monitor_ortools_fixed.py training_cpu/scripts/ortools/monitoring/

# Move OR-Tools table generation
echo "Moving OR-Tools table generation script..."
mv benchmark_gpu/scripts/generate_ortools_timeout_table.py training_cpu/scripts/ortools/

# Move OR-Tools output files (not Python scripts but related)
echo "Moving OR-Tools output files..."
mv benchmark_gpu/scripts/ortools_*.out training_cpu/scripts/ortools/
mv benchmark_gpu/scripts/ortools_*.tex training_cpu/scripts/ortools/

# Create symlinks for backward compatibility (optional)
echo "Creating compatibility symlinks..."
cd benchmark_gpu/scripts/
ln -s ../../training_cpu/scripts/ortools/production/run_ortools_gls_parallel_test.py .
ln -s ../../training_cpu/scripts/ortools/production/run_ortools_gls_production.py .
cd ../..

echo "Migration complete!"
echo ""
echo "IMPORTANT: After migration, you need to:"
echo "1. Update import paths in scripts that reference moved files"
echo "2. Update any shell scripts or documentation that reference old paths"
echo "3. Test the moved scripts to ensure they work from new location"
echo ""
echo "Files that may need import updates:"
echo "- Any script importing from benchmark_gpu.scripts that references OR-Tools"
echo "- Shell scripts in benchmark_gpu/scripts/*.sh"
echo "- Documentation files (README.md, etc.)"
