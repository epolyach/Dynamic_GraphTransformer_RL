#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CPU_CSV = PROJECT_ROOT / 'training_cpu' / 'results' / 'tiny_1' / 'csv' / 'history_gt_rl.csv'
GPU_V2_CSV = PROJECT_ROOT / 'training_gpu_v2' / 'results' / 'tiny_1' / 'csv' / 'history_gt_rl_v2.csv'


def main():
    if not CPU_CSV.exists():
        print(f"CPU CSV not found: {CPU_CSV}")
        sys.exit(1)
    if not GPU_V2_CSV.exists():
        print(f"GPU v2 CSV not found: {GPU_V2_CSV}")
        sys.exit(1)

    cpu = pd.read_csv(CPU_CSV)
    gpu = pd.read_csv(GPU_V2_CSV)

    print("=== Tiny_1 CPU vs GPU v2 Comparison ===")
    print(f"CPU epochs: {len(cpu)}; GPU v2 epochs: {len(gpu)}")

    # Compare first overlapping epoch if lengths differ
    n = min(len(cpu), len(gpu))
    if n == 0:
        print("No overlapping epochs to compare")
        sys.exit(0)

    cpu_subset = cpu.iloc[:n]
    gpu_subset = gpu.iloc[:n]

    # Report basic stats
    print("\nPer-epoch summary (first few epochs):")
    for i in range(min(5, n)):
        ce = cpu_subset.iloc[i]
        ge = gpu_subset.iloc[i]
        print(
            f"epoch {i}: CPU(train={ce['train_cost_geometric']:.4f}, val={ce['val_cost_geometric']:.4f}), "
            f"GPUv2(train={ge['train_cost_geometric']:.4f}, val={ge['val_cost_geometric']:.4f}, time={ge.get('time_per_epoch', float('nan')):.2f}s)"
        )

    print("\nAverages over compared epochs:")
    print(f"CPU train avg: {cpu_subset['train_cost_geometric'].mean():.4f}")
    print(f"CPU val   avg: {cpu_subset['val_cost_geometric'].mean():.4f}")
    print(f"GPU train avg: {gpu_subset['train_cost_geometric'].mean():.4f}")
    print(f"GPU val   avg: {gpu_subset['val_cost_geometric'].mean():.4f}")
    if 'time_per_epoch' in gpu_subset.columns:
        print(f"GPU time/epoch avg: {gpu_subset['time_per_epoch'].mean():.2f}s")


if __name__ == '__main__':
    main()

