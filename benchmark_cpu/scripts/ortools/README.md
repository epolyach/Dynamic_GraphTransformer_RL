# OR-Tools GLS Benchmark Runner

## Main Script: `production/run_ortools_gls.py`

Production-ready OR-Tools GLS benchmark runner with ProcessPoolExecutor parallel processing.

### Features
- ✅ Full CLI argument support
- ✅ Outputs to `benchmark_cpu/results/[subfolder]/`
- ✅ Creates NT JSON files (one per thread)
- ✅ Real-time logging to both console and `benchmark_log.txt`
- ✅ Thread-based parallel processing with optimal load balancing
- ✅ Auto-calculates vehicle capacity based on problem size

### Usage

```bash
python3 benchmark_cpu/scripts/ortools/production/run_ortools_gls.py \
    --subfolder SUBFOLDER \
    --n N \
    --instances NI \
    --timeout TIMEOUT \
    --threads NT \
    [--capacity CAPACITY]
```

### Arguments
- `--subfolder`: Name of subfolder in `benchmark_cpu/results/` where results will be saved
- `--n`: Problem size (number of customer nodes)
- `--instances`: Total number of instances to generate and solve (NI)
- `--timeout`: Timeout in seconds per instance
- `--threads`: Number of parallel threads (NT)
- `--capacity`: (Optional) Vehicle capacity. Auto-calculated if not provided

### Output Structure
```
benchmark_cpu/results/
└── [subfolder]/
    ├── benchmark_log.txt              # Real-time monitoring log
    ├── thread_00_n[N]_[timestamp].json  # Thread 0 results
    ├── thread_01_n[N]_[timestamp].json  # Thread 1 results
    └── ...                            # NT total JSON files
```

### Examples

#### Quick Test (N=10, 20 instances, 2 threads)
```bash
python3 production/run_ortools_gls.py \
    --subfolder "test_n10" \
    --n 10 \
    --instances 20 \
    --timeout 5 \
    --threads 2
```

#### Production Run (N=50, 1000 instances, 8 threads)
```bash
python3 production/run_ortools_gls.py \
    --subfolder "production_n50" \
    --n 50 \
    --instances 1000 \
    --timeout 30 \
    --threads 8
```

### Monitoring

The `benchmark_log.txt` file in the output subfolder provides:
- Real-time progress updates
- Thread status monitoring
- Instance distribution information
- Execution timing
- Success/failure tracking
- JSON file creation confirmation

### Example Scripts
- `run_example.sh`: Contains example benchmark configurations

### Directory Structure
```
benchmark_cpu/scripts/ortools/
├── production/
│   └── run_ortools_gls.py        # Main runner with CLI
├── benchmarks/
│   └── benchmark_ortools_gls_fixed.py  # Core benchmark implementation
├── monitoring/                    # Monitoring scripts
└── run_example.sh                # Usage examples
```
