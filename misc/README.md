# Misc Folder Contents

This folder contains auxiliary files, tests, analysis reports, and old versions that may be useful for reference but are not part of the main project execution.

## Directory Structure

### `/analysis`
Analysis reports and documentation generated during development:
- `ARCHITECTURE_MISMATCH_ANALYSIS.md` - Analysis of architecture differences
- `GAT_ANALYSIS_REPORT.md` - GAT model analysis
- `GT_RL_ANALYSIS.md` - GT+RL model analysis  
- `LEGACY_MODELS_INVENTORY.md` - Documentation of legacy model versions
- `POINTER_ANALYSIS_REPORT.md` - Pointer network analysis
- `POINTER_LEGACY_COMPARISON.md` - Comparison with legacy pointer implementation

### `/benchmarks`
Benchmarking and baseline scripts:
- `add_gt_greedy_baseline.py` - Script to add GT-Greedy baseline
- `benchmark_exact_cpu_modified.py` - CPU benchmarking script

### `/matlab`
MATLAB validation scripts:
- `validate_cvrp_solution.m` - CVRP solution validation
- `validate_cvrp_solution.asv` - MATLAB autosave file

### `/old_files`
Backup of old/replaced files:
- `README_old.md` - Previous version of README
- `run_enhanced_training.old` - Previous training script
- `benchmark_4_solvers_old.py` - Old benchmarking script
- Various `.old` model files (DGT variants, enhanced models, etc.)

### `/tests`
Test scripts for model validation:
- `test_advanced_gt.py` - Tests for advanced GT model
- `test_dgt.py` - Tests for unified DGT model
- `test_gt_simple.py` - Tests for simple GT model
- `test_legacy_gat.py` - Tests for legacy GAT model
- `test_minimal_gt.py` - Minimal GT testing
- `test_rollout_minimal.py` - Minimal rollout testing

### Root of `/misc`
Utility scripts:
- `erase_run.py` - Script to clean up run results
- `make_test_instance.py` - Generate test CVRP instances
- `environment_info.yaml` - Environment configuration info
