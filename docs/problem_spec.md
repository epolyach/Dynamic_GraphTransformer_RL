# CVRP Problem Specification (Unified)

This document defines the single, strict CVRP instance generation protocol used across training and benchmarks.

- Nodes: 1 depot + N customers
- Coordinates: Each node i has coordinates (x_i, y_i)
  - x_i, y_i are sampled uniformly from integer grid [0, coord_range], normalized to [0, 1]
- Demands: d_0 = 0 (depot), and for i >= 1, d_i ~ UniformInteger[demand_range.min, demand_range.max]
- Vehicle capacity: C = vehicle_capacity (integer)
- Distances: Euclidean distance matrix on normalized coordinates
- Augmentation: disabled by default in both training and benchmarks
- Random seeds:
  - Training: seeds derived from epoch and batch to avoid overlaps
  - Validation: distinct seed space (offset) to ensure no train/val leakage
  - Benchmarks: explicit seed provided via CLI for reproducibility

Configuration source of truth: configs/default.yaml (overridden by specific scale YAMLs).

Any code generating instances must import from src/generator/generator.py to ensure consistency.

