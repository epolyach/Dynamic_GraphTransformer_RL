# Dynamic Graph Transformer + Exact Benchmark Suite (Unified)

This repository unifies CPU training and CPU/GPU exact benchmarking under a single, consistent problem specification and shared code.

Key goals:
- One strict and unique CVRP instance generator used everywhere (training and benchmarks)
- Clear separation of concerns between training and benchmarking
- Only four models remain for training: GT-Greedy, GAT+RL, GT+RL, DGT+RL

## Project Structure

```
.
├── configs/                 # YAML configs (single source of truth)
│   ├── default.yaml
│   ├── tiny.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── production.yaml
├── src/
│   ├── generator/           # Canonical instance generator (used by all components)
│   │   └── generator.py
│   ├── models/              # Model implementations (only 4 kept)
│   ├── training/
│   ├── eval/
│   └── utils/
├── benchmark_cpu/           # CPU exact/heuristic benchmarks (uses src/generator)
├── benchmark_gpu/           # GPU exact benchmarks (uses src/generator)
├── run_training.py          # Train models (supports --all)
└── README.md
```

## Strict Problem Specification (single source of truth)
- Coordinates: integer grid [0, coord_range] normalized to [0,1]
- Demands: uniform integer in [demand_range.min, demand_range.max]
- Capacity: integer vehicle_capacity
- No augmentation in training/benchmarking by default
- Seed discipline: training/validation seeds do not overlap; benchmarks accept explicit seed(s)

These rules are implemented in src/generator/generator.py and read entirely from configs/*.yaml via src/utils/config.py.

## Training (CPU)
Supported models only: GAT+RL, GT+RL, DGT+RL, GT-Greedy

Examples:
```bash
python3 run_training.py --model GT+RL --config configs/small.yaml
python3 run_training.py --all --config configs/tiny.yaml
```

## Benchmarks
Benchmarks reuse the same generator and configs.

- CPU: benchmark_cpu/
- GPU: benchmark_gpu/

Examples:
```bash
# CPU exact/heuristics
python3 benchmark_cpu/run_exact.py --config configs/small.yaml --seed 42

# GPU exact
python3 benchmark_gpu/run_exact.py --config configs/small.yaml --seed 42
```

## Notes
- All defaults live in configs/default.yaml; no hidden defaults are allowed in code.
- Results, plots and virtualenvs are ignored by .gitignore.
- See docs/problem_spec.md for full details (to be expanded).

## License
Research code for academic use.
