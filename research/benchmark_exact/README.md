# CVRP Exact Solver Benchmark Suite

This folder contains a self-contained benchmarking suite for evaluating CVRP exact solvers. The suite compares Dynamic Programming (DP) and OR-Tools solvers on systematically generated CVRP instances.

## Contents

- `benchmark_cli.py` - Main CLI benchmark script with configurable parameters
- `plot_benchmark.py` - Publication-quality plotting script for benchmark results
- `exact_solver.py` - Core solver implementation (DP + OR-Tools)
- `enhanced_generator.py` - CVRP instance generator for consistent test problems
- `requirements.txt` - Python dependencies
- `README.md` - This documentation
- Various CSV data files and PNG/PDF plots from benchmark runs

## Installation

Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running Benchmarks

#### Basic Usage

```bash
# Run comparison benchmark (DP vs OR-Tools) - default settings
python benchmark_cli.py

# Run DP-only benchmark (faster, matches original baseline)
python benchmark_cli.py --mode dp-only
```

#### Advanced Configuration

```bash
# Custom benchmark with specific parameters
python benchmark_cli.py \
    --mode compare \
    --instances 50 \
    --timeout 30 \
    --n-start 5 \
    --n-end 20 \
    --capacity 25 \
    --demand-min 1 \
    --demand-max 8 \
    --output my_results.csv
```

#### Key Parameters

- `--mode`: Choose `dp-only` (fast) or `compare` (DP vs OR-Tools)
- `--instances`: Number of instances to solve per problem size N (default: 100)
- `--timeout`: Maximum time in seconds per problem size N (default: 60.0)
- `--n-start`, `--n-end`: Problem size range (default: 5 to 50 customers)
- `--capacity`: Vehicle capacity (default: 30)
- `--demand-min`, `--demand-max`: Customer demand range (default: [1, 10])
- `--output`: Output CSV filename (default: benchmark_results.csv)

### Generating Plots

#### Basic Plotting

```bash
# Generate plots from benchmark results
python plot_benchmark.py benchmark_results.csv

# Custom output prefix
python plot_benchmark.py results.csv --output my_analysis
```

This creates:
- `cvrp_benchmark.png` - High-resolution plot (300 DPI)

#### Plot Features

- **Panel 1**: Execution time vs problem size (log scale)
- **Panel 2**: Solution quality (cost per customer) with error bars
- Automatic detection of single-solver vs comparison mode
- Clean, publication-quality formatting with 2 vertical panels
- Error bars showing standard deviation across instances

## Important Notes

### Solver Behavior

- **Pure Solutions Only**: Both DP and OR-Tools now raise exceptions on timeout/failure instead of using fallback heuristics
- **Statistics**: Only successfully solved instances contribute to averages and standard deviations
- **Sample Sizes**: The benchmark reports actual number of solved instances per problem size

### Performance Expectations

- **DP Solver**: Exponential complexity O(n²·2ⁿ), optimal for N ≤ 12-15
- **OR-Tools**: Polynomial scaling, practical for larger instances
- **Crossover Point**: DP becomes slower than OR-Tools around N = 15-20

### File Structure

The suite is completely self-contained with local imports:
```python
from exact_solver import ExactCVRPSolver
from enhanced_generator import EnhancedCVRPGenerator
```

## Examples

### Quick DP Benchmark

```bash
# Fast DP-only benchmark for N=5 to N=15
python benchmark_cli.py --mode dp-only --n-start 5 --n-end 15 --instances 20
python plot_benchmark.py benchmark_results.csv
```

### Comprehensive Comparison

```bash
# Full comparison with extended timeout
python benchmark_cli.py --mode compare --timeout 120 --instances 50
python plot_benchmark.py benchmark_results.csv --output detailed_comparison
```

### Small-Scale Analysis

```bash
# Quick test run
python benchmark_cli.py --n-start 5 --n-end 10 --instances 10 --timeout 10
python plot_benchmark.py benchmark_results.csv --output quick_test
```

## Output Interpretation

### CSV Columns (Comparison Mode)

- `N`: Number of customers
- `time_or`: Average OR-Tools solve time (seconds)
- `cpc_or`: Average OR-Tools cost per customer
- `std_or`: Standard deviation of OR-Tools costs
- `time_dp`: Average DP solve time (seconds)
- `cpc_dp`: Average DP cost per customer
- `std_dp`: Standard deviation of DP costs

### CSV Columns (DP-Only Mode)

- `N`: Number of customers
- `time_s`: Average DP solve time (seconds)
- `cost_per_customer`: Average DP cost per customer
- `std`: Standard deviation of DP costs

### Performance Metrics

- **Cost per Customer**: Normalized solution quality metric
- **Solve Time**: Average time for successful solves only
- **Standard Deviation**: Variability across successfully solved instances
- **Sample Size**: Displayed in terminal output (solved/attempted)

## Troubleshooting

If you encounter import errors, ensure you're running from within the `benchmark_exact` directory:
```bash
cd benchmark_exact
python benchmark_cli.py --help
```

For plotting issues, ensure matplotlib is properly installed:
```bash
pip install matplotlib numpy
```

## 📊 Key Findings from Existing Data

The existing benchmark data shows:

- **Data Range**: N = 5 to 16 customers
- **Crossover Point**: DP becomes slower than OR-Tools around N = 12-15
- **Speed**: DP significantly faster for small instances (N ≤ 12)
- **Quality**: DP produces exact solutions vs OR-Tools near-optimal
- **Growth**: DP shows exponential growth vs OR-Tools polynomial scaling
- **Recommendation**: Use DP for N ≤ 12, OR-Tools for larger instances

## Dependencies

Required Python packages (see `requirements.txt`):
- numpy
- matplotlib
- ortools (for OR-Tools solver)
- gurobipy (optional, for Gurobi solver)

## Contact

This is a self-contained benchmarking tool for CVRP exact solvers research.
