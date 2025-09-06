# Cleanup Recommendations for benchmark_gpu/scripts/
Generated: 2025-09-06

## Identified Issues

### 1. Duplicate/Obsolete Scripts
- `monitor_all_ortools_backup.py` - Backup of monitor_all_ortools.py
- `benchmark_gpu_dp_exact_fixed.py.backup` - Backup file
- `test_cpc_lognormal.py.backup` - Backup file
- Multiple "fixed" versions alongside originals:
  - `benchmark_ortools_gls.py` vs `benchmark_ortools_gls_fixed.py`
  - `benchmark_ortools_multi_n.py` vs `benchmark_ortools_multi_n_fixed.py`
  - `continuous_monitor_ortools.py` vs `continuous_monitor_ortools_fixed.py`
  - `generate_table_with_normality.py` vs `generate_table_with_normality_fixed.py`

### 2. Test Scripts Mixed with Production
- 8 test_*.py files mixed with production scripts
- Should be in a separate test/ directory

### 3. Data Files Mixed with Code
- `.tex` files (LaTeX tables)
- `.out` files (output logs)
- `.txt` summary files
- `.png` and `.eps` figure files
- `.csv` data files

## Recommended Directory Structure

```
benchmark_gpu/
├── scripts/
│   ├── benchmarks/        # Main benchmark runners
│   ├── tests/            # Test scripts
│   ├── visualization/    # Plotting and figure generation
│   ├── monitoring/       # Progress monitoring
│   └── utils/           # Utility scripts
├── data/                # Generated data files
│   ├── results/         # CSV, JSON results
│   └── logs/           # .out, .txt log files
└── figures/            # Generated plots (.png, .eps)

training_cpu/
├── scripts/
│   └── ortools/        # All OR-Tools CPU scripts
│       ├── production/
│       ├── benchmarks/
│       └── monitoring/
```

## Cleanup Actions

### Immediate (Safe to do now):
```bash
# 1. Remove backup files
rm benchmark_gpu/scripts/*.backup
rm benchmark_gpu/scripts/*_backup.py

# 2. Move data files out of scripts directory
mkdir -p benchmark_gpu/data/summaries
mv benchmark_gpu/scripts/*.txt benchmark_gpu/data/summaries/
mv benchmark_gpu/scripts/*.csv benchmark_gpu/data/

mkdir -p benchmark_gpu/figures
mv benchmark_gpu/scripts/*.png benchmark_gpu/figures/
mv benchmark_gpu/scripts/*.eps benchmark_gpu/figures/

mkdir -p benchmark_gpu/tables
mv benchmark_gpu/scripts/*.tex benchmark_gpu/tables/
```

### After Review:
```bash
# 3. Archive obsolete scripts
mkdir -p benchmark_gpu/scripts/archive
mv benchmark_gpu/scripts/*_fixed.py benchmark_gpu/scripts/archive/

# 4. Organize test scripts
mkdir -p benchmark_gpu/scripts/tests
mv benchmark_gpu/scripts/test_*.py benchmark_gpu/scripts/tests/

# 5. Move example scripts
mkdir -p benchmark_gpu/scripts/examples
mv benchmark_gpu/scripts/*_example*.py benchmark_gpu/scripts/examples/
```

## Scripts to Keep Active

### Core Production Scripts:
1. **GPU Benchmarks:**
   - `benchmark_gpu_multi_n.py`
   - `benchmark_gpu_adaptive_n.py`
   - `benchmark_gpu_heuristic_gls_advanced.py`

2. **CPU/OR-Tools (Latest):**
   - `run_ortools_gls_parallel_test.py` ⭐ (with thread support)
   - `src/benchmarking/solvers/cpu/ortools_gls.py`

3. **Visualization:**
   - `plot_cpu_gpu_comparison.py`
   - `make_log_norm_figure_cli.py` (latest)

4. **Table Generation:**
   - `generate_final_latex_tables.py`
   - `generate_ortools_timeout_table.py`

## Scripts to Archive/Remove

### Can be archived:
- All `*_backup.py` files
- All `*_fixed.py` files (when original exists)
- Older versions of scripts with newer replacements

### Can be removed after verification:
- Output files (.out, .txt logs)
- Generated figures that can be regenerated
- Temporary CSV files

## Consolidation Opportunities

1. **Merge similar monitoring scripts:**
   - Combine all monitor_ortools_*.py into one configurable script

2. **Unify table generation:**
   - Consolidate generate_*_table*.py scripts into one flexible generator

3. **Combine test scripts:**
   - Merge test_heuristic.py and test_heuristic2.py
   - Combine all test_cpc_*.py scripts

## Backup Strategy

Before any changes:
```bash
# Full backup
tar -czf benchmark_gpu_scripts_full_backup_$(date +%Y%m%d).tar.gz benchmark_gpu/scripts/

# Create git branch for changes
git checkout -b cleanup-scripts-$(date +%Y%m%d)
git add benchmark_gpu/scripts/scripts_inventory.md
git add benchmark_gpu/scripts/cleanup_recommendations.md
git add benchmark_gpu/scripts/migration_plan.sh
git commit -m "Document script inventory and cleanup plan"
```

## Priority Actions

1. **High Priority:**
   - Move OR-Tools scripts to training_cpu/
   - Remove .backup files
   - Move data files out of scripts/

2. **Medium Priority:**
   - Organize test scripts
   - Archive fixed versions
   - Consolidate monitoring scripts

3. **Low Priority:**
   - Merge similar scripts
   - Update documentation
   - Clean up imports

## Notes on Latest Implementations

### OR-Tools GLS with Thread Support
The latest implementation is `run_ortools_gls_parallel_test.py` which uses:
- ProcessPoolExecutor for true parallelism
- Configurable thread counts
- Striped instance allocation
- Individual JSON output per thread
- Proper timeout management

This should be preserved as the primary OR-Tools GLS implementation going forward.
