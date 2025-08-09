#!/usr/bin/env python3
from pathlib import Path
import sys
import shutil
import pandas as pd
import numpy as np
import torch

# Ensure project root on path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.experiments.run_comparative_study_cpu import run_experiment, generate_plots


def main():
    device = torch.device("cpu")
    # 20 customers => 21 nodes including depot
    problem_sizes = [21]
    instances = 400
    runs = 1
    seed = 0
    capacity = 3.0

    print("Running CPU comparative study: Customers=20, Instances=400 ...")
    df = run_experiment(device, problem_sizes, instances, runs, seed, capacity)

    out_dir = project_root / "results" / "cpu_C20_I400"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparative_study_cpu.csv"
    df.to_csv(csv_path, index=False)

    # Generate plots to results dir
    generate_plots(df, out_dir, problem_sizes)

    # Copy main plot to utils/plots with the conventional name
    src_plot = out_dir / "comparative_study_cpu.png"
    dst_plot = project_root / "utils" / "plots" / "comparative_study_results_05.png"
    try:
        shutil.copy2(src_plot, dst_plot)
        print(f"Saved plot copy to {dst_plot}")
    except Exception as e:
        print(f"Warning: failed to copy plot: {e}")

    # Build markdown-like table
    # Filter for our problem size
    size = problem_sizes[0]
    sdf = df[df["problem_size"] == size].copy()

    # Map variants to desired display order and names
    order = [
        ("greedy_baseline", "greedy_baseline"),
        ("pointer_rl", "pointer_rl"),
        ("static_rl", "static_rl"),
        ("dynamic_gt_rl", "dynamic_gt_rl"),
        ("naive_baseline", "naive_baseline"),
    ]

    rows = []
    for key, label in order:
        row = sdf[sdf["variant"] == key]
        if row.empty:
            cost = ""
            t = ""
        else:
            cost_val = float(row["avg_solution_cost_per_customer"].iloc[0])
            t_val = row["avg_computation_time"].iloc[0]
            cost = f"{cost_val:.3f}"
            t = f"{float(t_val):.3f}" if np.isfinite(t_val) else "—"
        rows.append((label, cost, t))

    # Print table
    header = ["Model", "CPU Cost/Cust", "CPU Time (s)"]
    w = [max(len(str(x[i])) for x in ([header] + rows)) for i in range(3)]
    fmt = lambda r: " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r))
    print()
    print(fmt(header))
    print("-|-".join("-"*wi for wi in w))
    for r in rows:
        print(fmt(r))

if __name__ == "__main__":
    main()
