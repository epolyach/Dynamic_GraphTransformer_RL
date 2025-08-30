# CVRP Benchmark Usage Guide

## Configuration-Driven Benchmarks

Both CPU and GPU benchmarks now use **identical parameters** loaded from `config.json` to ensure fair comparisons.

### Configuration File (`config.json`)

```json
{
  "instance_generation": {
    "capacity": 30,
    "demand_min": 1,
    "demand_max": 10,
    "coord_range": 100,
    "coordinates_normalized": true
  }
}
```

**Fixed Parameters (DO NOT CHANGE):**
- **Capacity**: 30
- **Demand**: Integer range [1, 10]
- **Coordinates**: Generated as integers [0, 100], normalized to [0, 1]

## Usage

### CPU Benchmark
```bash
python3 benchmark_exact_cpu.py --n-start 5 --n-end 10 --instances-min 10 --instances-max 10 --timeout 120
```

### GPU Benchmark  
```bash
python3 benchmark_exact_gpu.py --n-start 5 --n-end 10 --instances 10 --timeout 120
```

### Key Features

✅ **Unified Instance Generation**: Both benchmarks generate identical instances  
✅ **Config-Driven**: Parameters loaded from `config.json`  
✅ **Identical Results**: Same instances produce identical optimal costs  
✅ **Fair Comparison**: Performance differences reflect computational efficiency, not instance variations

### Verification

Both benchmarks will display config validation on startup:
```
📋 Loading configuration from config.json...
✅ Config validation passed
   - Capacity: 30
   - Demand range: [1, 10]
   - Coordinate range: [0, 100] normalized to [0, 1]
🔧 Using parameters: capacity=30, demand=[1,10], coord_range=100
```

### Files

- `benchmark_exact_cpu.py`: Main CPU benchmark (config-driven)
- `benchmark_exact_gpu.py`: Main GPU benchmark (config-driven, unified instance generation)
- `config.json`: Fixed parameter configuration
- `config_loader.py`: Configuration loader utility
- `verify_config_consistency.py`: Verification script
