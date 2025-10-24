# Results Data Consolidation Plan

## Objective
Collect and organize ALL training and benchmark results in the repository with clear documentation, WITHOUT disrupting the running training.

## Current Running Training
**Screen Session:** 1940710.training
**Script:** training_gpu_prefetch/scripts/run_training_gpu.py
**Config:** configs/normal_gpu_1000.yaml
**Status:** ⚠️ DO NOT TOUCH - Keep running

## Code Version Confirmation
✅ **training_gpu_prefetch/** has the MOST RECENT code:
- advanced_trainer_gpu.py: Oct 2, 2025 (vs Sep 30 in training_gpu)
- rollout_baseline_gpu_fixed.py: Oct 2, 2025 (vs Sep 26 in training_gpu)
- batch_prefetcher.py: Only exists in prefetch version

**Conclusion:** training_gpu_prefetch is the active, most recent codebase.

## Results Consolidation Strategy

### Phase 1: Create Results Archive Structure
```
results_consolidated/
├── training/
│   ├── gpu/
│   │   ├── tiny_gpu_150/
│   │   ├── tiny_gpu_500/
│   │   ├── tiny_gpu_750/
│   │   ├── tiny_gpu_1000/
│   │   ├── tiny_gpu_512_variants/
│   │   └── experiment_5_curriculum/
│   ├── cpu/
│   │   ├── tiny/
│   │   └── tiny_1/
│   └── legacy_root_results/
│       ├── default/
│       ├── test_512/
│       ├── test_csv_resume/
│       └── medium/
├── benchmarks/
│   ├── cpu/
│   │   ├── ortools_gls/
│   │   ├── ortools_greedy/
│   │   └── metrics_tables/
│   └── gpu/
│       ├── exact_dp/
│       └── heuristic/
└── RESULTS_INDEX.md
```

### Phase 2: Copy (Not Move) Results
**Important:** We'll COPY results, not move them, to preserve the original structure while the training runs.

```bash
# Create consolidated directory
mkdir -p results_consolidated/{training/{gpu,cpu,legacy_root_results},benchmarks/{cpu,gpu}}

# Copy training_gpu results
rsync -av training_gpu/results/ results_consolidated/training/gpu/from_training_gpu/

# Copy training_gpu_prefetch results  
rsync -av training_gpu_prefetch/results/ results_consolidated/training/gpu/from_prefetch/

# Copy training_cpu results
rsync -av training_cpu/results/ results_consolidated/training/cpu/

# Copy root-level results
rsync -av results/ results_consolidated/training/legacy_root_results/

# Copy benchmark results
rsync -av benchmark_cpu/results/ results_consolidated/benchmarks/cpu/
rsync -av benchmark_gpu/results/ results_consolidated/benchmarks/gpu/
```

### Phase 3: Create RESULTS_INDEX.md
Generate comprehensive documentation of all results:
- Experiment name
- Configuration used
- Training metrics (best/final CPC)
- Date and duration
- Original location
- File sizes
- Status (complete/incomplete)

### Phase 4: Deduplicate and Organize
Identify duplicate results between training_gpu and training_gpu_prefetch:
- Compare by experiment name
- Keep newer version (from prefetch)
- Document which was kept in INDEX

### Phase 5: Create Analysis Scripts
Generate scripts to:
- Load and compare all training runs
- Generate consolidated plots
- Create LaTeX tables of all results
- Extract best models and configurations

## Execution Steps (Safe for Running Training)

1. ✅ Create results_consolidated/ structure
2. ✅ Copy all results (read-only operations)
3. ✅ Generate RESULTS_INDEX.md
4. ✅ Create comparison and analysis scripts
5. ✅ Verify data integrity
6. Document findings

**Note:** We will NOT delete or move anything from training_gpu_prefetch/ until training completes.

## Ready to Execute?
This is completely safe - all operations are read-only copies that won't affect your running training.

Shall I proceed with results consolidation?
