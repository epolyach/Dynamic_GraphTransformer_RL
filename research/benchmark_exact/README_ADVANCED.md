# Advanced CVRP Solver Benchmark System

A comprehensive benchmark suite for comparing **exact** and **near-optimal** CVRP algorithms at scale. Supports problem sizes from N=5 to N=500+ customers with state-of-the-art algorithms.

## 🚀 **Key Features**

### **🎯 Intelligent Algorithm Selection**
- **N ≤ 12**: Dynamic Programming (exact, very fast)
- **N ≤ 20**: Branch-and-Cut with CVRP cuts (exact, fast)  
- **N ≤ 50**: Enhanced OR-Tools (exact/near-exact, moderate)
- **N ≤ 200**: HGS-CVRP (near-optimal, very fast)
- **N > 200**: ALNS (heuristic, fast and robust)

### **🏆 State-of-the-Art Algorithms**

#### **Exact Algorithms**
- **Dynamic Programming**: Bitmasking with state space reduction
- **Branch-and-Cut**: Gurobi with CVRP-specific cutting planes
- **Enhanced OR-Tools**: Advanced constraint programming configuration

#### **Near-Optimal Heuristics**
- **HGS-CVRP**: Hybrid Genetic Search (world-champion algorithm, 0.1-0.5% gaps)
- **ALNS**: Adaptive Large Neighborhood Search (robust, 1-3% gaps)
- **Multi-start Local Search**: Advanced operators for large instances

### **📊 Multiple Benchmark Modes**
- **`exact-only`**: Pure exact algorithms for rigorous optimality studies
- **`compare`**: Exact vs Heuristic comparison for algorithm evaluation
- **`heuristic-only`**: Large-scale heuristic performance analysis

### **📈 Publication-Quality Visualization**
- Automatic benchmark mode detection
- Time complexity analysis with log-scale plots
- Solution quality comparison with error bars
- Optimality gap analysis
- Algorithm usage tracking

## 🛠️ **Quick Start**

### **Installation**

```bash
# Basic setup (DP + ALNS only)
pip install numpy matplotlib

# Recommended setup (with OR-Tools)
pip install numpy matplotlib ortools

# Best setup (with HGS-CVRP - world's best heuristic)
pip install numpy matplotlib ortools pyvrp

# Full setup (with Gurobi - requires license)
pip install numpy matplotlib ortools pyvrp gurobipy
```

### **Basic Usage**

```bash
# Small-scale exact benchmark (N=5-20)
python benchmark_advanced_cli.py --n-start 5 --n-end 20 --mode exact-only

# Medium-scale comparison (N=5-50)  
python benchmark_advanced_cli.py --n-start 5 --n-end 50 --mode compare

# Large-scale heuristic benchmark (N=10-200)
python benchmark_advanced_cli.py --n-start 10 --n-end 200 --mode heuristic-only

# Visualize results
python plot_advanced_benchmark.py benchmark_advanced_results.csv
```

## 📋 **Command Line Options**

### **benchmark_advanced_cli.py**

```bash
python benchmark_advanced_cli.py [OPTIONS]

Required Options:
  --mode {exact-only,compare,heuristic-only}  Benchmark mode

Problem Configuration:
  --n-start N              Start N (default: 5)
  --n-end N                End N inclusive (default: 200)
  --capacity N             Vehicle capacity (default: 30)
  --demand-min N           Min demand (default: 1)
  --demand-max N           Max demand (default: 10)

Execution Control:
  --instances N            Instances per N (default: 50)
  --timeout SECONDS        Timeout per N (default: 120.0)
  --output FILE            Output CSV file
  --verbose                Verbose output

Examples:
  # Quick test
  python benchmark_advanced_cli.py --n-start 5 --n-end 15 --instances 10

  # Research study  
  python benchmark_advanced_cli.py --n-start 5 --n-end 100 --instances 100 --timeout 300

  # Large-scale heuristic analysis
  python benchmark_advanced_cli.py --n-start 20 --n-end 500 --mode heuristic-only
```

### **plot_advanced_benchmark.py**

```bash
python plot_advanced_benchmark.py CSV_FILE [OPTIONS]

Options:
  --output PREFIX          Output file prefix (default: auto-generate)
  --title "TITLE"          Custom plot title

Examples:
  python plot_advanced_benchmark.py results.csv
  python plot_advanced_benchmark.py results.csv --title "CVRP Algorithm Comparison"
```

## 🔬 **Benchmark Modes**

### **1. Exact-Only Mode (`--mode exact-only`)**

**Purpose**: Rigorous optimality studies with proven optimal solutions.

**Algorithms Used**:
- N ≤ 12: Dynamic Programming  
- N ≤ 20: Branch-and-Cut (Gurobi) or Enhanced OR-Tools
- N ≤ 50: Enhanced OR-Tools

**Output**: Time complexity, solution quality, algorithm usage

**Use Cases**:
- Algorithm validation
- Optimality proofs
- Time complexity analysis
- Small to medium instance studies

```bash
python benchmark_advanced_cli.py --mode exact-only --n-start 5 --n-end 25
```

### **2. Compare Mode (`--mode compare`)**

**Purpose**: Head-to-head comparison of exact vs heuristic algorithms.

**What it does**:
- Runs both exact and heuristic algorithms on same instances
- Computes optimality gaps
- Analyzes speed vs quality tradeoffs

**Output**: Comparative performance, optimality gaps, algorithm selection analysis

**Use Cases**:
- Algorithm benchmarking
- Research comparisons
- Method validation
- Performance analysis

```bash
python benchmark_advanced_cli.py --mode compare --n-start 5 --n-end 50
```

### **3. Heuristic-Only Mode (`--mode heuristic-only`)**

**Purpose**: Large-scale performance analysis with near-optimal algorithms.

**Algorithms Used**:
- N ≤ 200: HGS-CVRP (if available)
- N > 200: ALNS
- Fallback: Greedy + Local Search

**Output**: Scalability analysis, estimated gaps, algorithm robustness

**Use Cases**:
- Large-scale studies
- Industrial applications
- Scalability analysis
- Real-world performance

```bash
python benchmark_advanced_cli.py --mode heuristic-only --n-start 20 --n-end 500
```

## 📊 **Expected Performance**

### **Exact Algorithms**

| Algorithm | Size Range | Solve Time | Optimality | Best For |
|-----------|------------|------------|-------------|----------|
| **DP** | N ≤ 12 | < 1s | Guaranteed | Small instances |
| **Branch-Cut** | N ≤ 20 | 1-30s | Guaranteed | Medium instances |
| **OR-Tools** | N ≤ 50 | 5-300s | Guaranteed* | Larger instances |

*OR-Tools may use heuristics for very large instances

### **Heuristic Algorithms**

| Algorithm | Size Range | Solve Time | Typical Gap | Best For |
|-----------|------------|------------|-------------|----------|
| **HGS-CVRP** | N ≤ 200 | < 1s | 0.1-0.5% | Near-optimal, fast |
| **ALNS** | N ≤ 500+ | 1-10s | 1-3% | Large instances |
| **Greedy** | Any | < 0.1s | 5-15% | Fallback only |

## 🎯 **Example Use Cases**

### **1. Research Validation Study**
```bash
# Compare exact vs heuristic for research paper
python benchmark_advanced_cli.py \
  --mode compare \
  --n-start 5 --n-end 30 \
  --instances 100 --timeout 600 \
  --output research_validation.csv

python plot_advanced_benchmark.py research_validation.csv \
  --title "Exact vs Heuristic CVRP Algorithm Comparison"
```

### **2. Industrial Scalability Analysis**
```bash
# Large-scale heuristic performance for industry
python benchmark_advanced_cli.py \
  --mode heuristic-only \
  --n-start 50 --n-end 500 \
  --instances 50 --timeout 120 \
  --output industrial_scale.csv

python plot_advanced_benchmark.py industrial_scale.csv \
  --title "Industrial CVRP Solver Scalability Analysis"
```

### **3. Algorithm Development Baseline**
```bash
# Establish exact baselines for new algorithm comparison
python benchmark_advanced_cli.py \
  --mode exact-only \
  --n-start 5 --n-end 20 \
  --instances 200 --timeout 1800 \
  --output exact_baseline.csv

python plot_advanced_benchmark.py exact_baseline.csv \
  --title "Exact CVRP Baseline for Algorithm Development"
```

## 🔧 **Algorithm Availability**

The benchmark automatically detects available algorithms:

- **✅ Always Available**: Dynamic Programming, ALNS, Greedy
- **✅ OR-Tools**: Install with `pip install ortools`
- **✅ HGS-CVRP**: Install with `pip install pyvrp` (highly recommended)
- **⭐ Gurobi**: Install with `pip install gurobipy` (requires license)

**Recommendation**: Install PyVRP for HGS-CVRP - it's the world's best CVRP heuristic.

## 📁 **File Structure**

```
benchmark_exact/
├── advanced_solver.py              # Advanced algorithm implementations
├── benchmark_advanced_cli.py       # Main benchmark CLI
├── plot_advanced_benchmark.py      # Enhanced plotting system
├── enhanced_generator.py           # CVRP instance generator
├── requirements_advanced.txt       # Dependency specifications
├── README_ADVANCED.md              # This file
└── [results]
    ├── benchmark_advanced_results.csv  # Benchmark data
    └── plot_*.png                      # Generated plots
```

## 🤝 **Compatibility**

- **Backward Compatible**: Works with original `plot_benchmark.py` for DP/OR-Tools data
- **Forward Compatible**: Enhanced plotting supports all benchmark modes
- **Cross-Platform**: Windows, macOS, Linux
- **Python**: 3.8+

## 🔬 **Scientific Applications**

### **Research Papers**
- Algorithm comparison studies
- Optimality gap analysis  
- Time complexity validation
- Scalability benchmarking

### **Industrial Applications**
- Solver selection for production systems
- Performance validation
- Scalability planning
- Cost-benefit analysis

### **Algorithm Development**
- Baseline establishment
- Performance validation
- Comparative evaluation
- Improvement measurement

## 🎉 **Ready to Scale!**

This advanced benchmark system is designed to handle:

- ✅ **Small instances** (N ≤ 20): Exact algorithms with optimality proofs
- ✅ **Medium instances** (N ≤ 50): Mixed exact/heuristic comparison  
- ✅ **Large instances** (N ≤ 200): State-of-the-art heuristics
- ✅ **Very large instances** (N > 200): Robust scalable algorithms

Start with a small test, then scale up to your research or industrial needs!

```bash
# Start small
python benchmark_advanced_cli.py --n-start 5 --n-end 15 --instances 10

# Scale up!
python benchmark_advanced_cli.py --n-start 10 --n-end 200 --instances 100 --mode compare
```

Happy benchmarking! 🚀
