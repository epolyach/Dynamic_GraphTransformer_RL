# Project Cleanup and Reorganization Plan

## Current State Analysis (Node: sedan-gpu2)

### Directory Structure
```
Current:
- benchmark_cpu/         ✅ Keep
- benchmark_gpu/         ✅ Keep
- training_cpu/          ✅ Keep
- training_gpu/          ❌ Remove (superseded by prefetch)
- training_gpu_prefetch/ ✅ Keep → Rename to training_gpu/
- src/                   ✅ Keep (core implementation)
- configs/               ✅ Keep
- results/               ⚠️  Merge into appropriate training dirs
- remove/                ❌ Delete
- scripts/               ⚠️  Review and consolidate
- MD/                    ⚠️  Review documentation
- !_paper/               ✅ Keep (paper figures/code)
- venv/                  ✅ Keep
```

### Results to Consolidate

#### Training Results (CSV/JSON)
**training_gpu/** (to be removed):
- tiny_gpu_150/
- tiny_gpu_500/
- tiny_gpu_750/
- tiny_gpu_512/
- tiny_gpu_512_fixed_temp/
- tiny_gpu_512_optimal/
- tiny_gpu_512_optimal_T20/
- experiment_5_curriculum/

**training_gpu_prefetch/** (to become training_gpu/):
- tiny_gpu_150/ ✅
- tiny_gpu_500/ ✅
- tiny_gpu_750/ ✅
- tiny_gpu_1000/ ✅
- tiny_gpu_512/ ✅
- tiny_gpu_512_fixed_temp/ ✅
- tiny_gpu_512_optimal/ ✅
- tiny_gpu_512_optimal_T20/ ✅
- experiment_5_curriculum/ ✅

**training_cpu/**:
- tiny/
- tiny_1/

**results/** (root level - to be moved):
- default/
- test_512/
- test_csv_resume/
- medium/

#### Benchmark Results

**benchmark_cpu/results/**:
- CSV files (various experiments)
- OR-Tools GLS results (multiple N values)
- OR-Tools Greedy results
- Metrics tables

**benchmark_gpu/results/**:
- Exact DP results (N=6, 8, 10)
- Heuristic GLS results
- CSV comparison files

## Cleanup Strategy

### Phase 1: Backup Current State
```bash
# Create backup of entire project
cd /home/evgeny.polyachenko/CVRP
tar -czf Dynamic_GraphTransformer_RL_backup_$(date +%Y%m%d_%H%M%S).tar.gz Dynamic_GraphTransformer_RL/
```

### Phase 2: Results Consolidation

#### Step 2.1: Merge training_gpu results into training_gpu_prefetch
```bash
# Copy unique results from training_gpu to training_gpu_prefetch
# (Skip duplicates, prefetch versions are newer)
```

#### Step 2.2: Move root-level results to appropriate locations
```bash
# Move results/default → training_gpu_prefetch/results/
# Move results/medium → training_gpu_prefetch/results/
# Move results/test_* → training_gpu_prefetch/results/ (or archive)
```

#### Step 2.3: Create consolidated results documentation
```bash
# Create RESULTS_INDEX.md documenting all available results
```

### Phase 3: Code Consolidation

#### Step 3.1: Compare and verify no functionality loss
```bash
# training_gpu_prefetch has:
# - batch_prefetcher.py (NEW)
# - advanced_trainer_gpu.py (105 more lines, enhanced)
# - rollout_baseline_gpu_fixed.py (improved version)
# - All other files identical
```

#### Step 3.2: Rename training_gpu_prefetch → training_gpu
```bash
# After removing old training_gpu
```

### Phase 4: Documentation Cleanup

#### Keep (Essential):
- README.md (updated)
- PREFETCH_TRAINING_GUIDE.md → Merge into README
- GPU_TRAINING_GUIDE.md → Merge into README
- benchmark_cpu/scripts/ortools/README.md
- training_gpu_prefetch/README.md → becomes training_gpu/README.md

#### Archive (Historical/Debug):
- CLEANUP_AND_NEXT_STEPS.md
- CPU_OPTIMIZATION_SUMMARY.md
- PARALLEL_DATA_GENERATION.md
- PREFETCH_BUG_FOUND.md
- SEED_VERIFICATION.md
- TRAINING_ISSUES_ANALYSIS.md
- WHY_SEQUENTIAL_BATCHES.md
- OPTIMIZATION_SUMMARY.txt
- All other debug/analysis .md files

→ Move to `archive/documentation/`

### Phase 5: Scripts Cleanup

#### Root-level scripts to keep:
- run_seq_nohup_prefetch.sh → rename to run_seq_nohup.sh
- setup_venv.sh
- requirements.txt

#### Scripts to remove/archive:
- run_seq.sh (old version)
- run_seq_nohup.sh (old version)
- run_seq_prefetch.sh (redundant)
- run_tiny_gpu_experiments*.sh (specific experiments)
- restart_training.sh (specific to old setup)
- gpu_cluster_monitor.sh (if not used)

### Phase 6: Remove Directories
```bash
- remove/                (already marked for removal)
- training_gpu/          (after merging results)
- scripts/               (if empty or consolidated)
```

## Multi-Node Synchronization

Since you mentioned 4 nodes, we need to:

1. **Identify which nodes have the repository**
2. **Check for node-specific results**
3. **Consolidate all unique results**
4. **Synchronize cleaned version across all nodes**

### Commands to check other nodes:
```bash
# Check if other nodes accessible
for node in sedan-gpu1 sedan-gpu3 sedan-gpu4; do
  echo "=== $node ==="
  ssh $node "cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL && pwd && ls -la" 2>&1
done
```

## Expected Clean Structure

```
Dynamic_GraphTransformer_RL/
├── README.md                    # Comprehensive, updated
├── requirements.txt
├── setup_venv.sh
├── run_seq_nohup.sh            # Renamed from prefetch version
├── configs/                     # All YAML configs
├── src/                         # Core implementation
│   ├── generator/
│   ├── models/
│   ├── benchmarking/
│   ├── metrics/
│   └── ...
├── benchmark_cpu/
│   ├── scripts/
│   ├── results/
│   │   ├── RESULTS_INDEX.md   # Documents all results
│   │   ├── ortools_gls_*/     # Organized by experiment
│   │   └── csv/
│   └── README.md
├── benchmark_gpu/
│   ├── scripts/
│   ├── results/
│   │   ├── RESULTS_INDEX.md
│   │   ├── exact_dp_*/
│   │   └── csv/
│   └── README.md
├── training_cpu/
│   ├── scripts/
│   ├── lib/
│   ├── results/
│   │   ├── RESULTS_INDEX.md
│   │   └── tiny*/
│   └── README.md
├── training_gpu/                # Renamed from training_gpu_prefetch
│   ├── scripts/
│   ├── lib/
│   │   ├── batch_prefetcher.py
│   │   ├── advanced_trainer_gpu.py
│   │   └── ...
│   ├── results/
│   │   ├── RESULTS_INDEX.md
│   │   └── all consolidated results
│   └── README.md
├── archive/
│   ├── documentation/          # Historical .md files
│   ├── old_scripts/            # Deprecated scripts
│   └── old_results/            # Superseded experiments
├── venv/
└── paper/                      # Renamed from !_paper
    ├── figures/
    └── python/
```

## Key Benefits

1. **Clear Structure**: Only 4 main directories (benchmark_cpu, benchmark_gpu, training_cpu, training_gpu)
2. **No Duplication**: Single source of truth for each component
3. **Better Documentation**: Consolidated README with clear examples
4. **Preserved History**: Archive directory maintains historical context
5. **Multi-node Ready**: Easy to sync across all 4 nodes

## Next Steps

Would you like me to:
1. Check what's on the other 3 nodes?
2. Create detailed file-by-file migration script?
3. Start with Phase 1 (backup)?
4. Generate RESULTS_INDEX.md documenting all experiments?
