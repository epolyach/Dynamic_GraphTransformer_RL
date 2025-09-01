#!/usr/bin/env python3
import argparse
import time
from typing import List

from src.utils.config import load_config
from src.generator.generator import create_data_generator

from src.benchmarking.solvers.gpu import exact_gpu_dp
from src.benchmarking.solvers.gpu import exact_gpu_improved


def main():
    parser = argparse.ArgumentParser(description="GPU benchmark (exact+heuristic) using unified generator")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()

    config = load_config(args.config)
    gen = create_data_generator(config)

    instances: List[dict] = gen(batch_size=args.instances, epoch=1, seed=args.seed)
    n = config['problem']['num_customers']

    # GPU exact DP for small N (<=16)
    if n <= 16:
        t0 = time.time()
        results = exact_gpu_dp.solve_batch(instances, verbose=False)
        dt = time.time() - t0
        avg_cpc = sum(r.cost for r in results) / len(results) / n
        print(f"GPU-Exact-DP: N={n}, instances={len(instances)}, time={dt:.3f}s, avg CPC={avg_cpc:.4f}")

    # GPU improved heuristic (or exact for tiny N)
    t0 = time.time()
    results = exact_gpu_improved.solve_batch(instances, time_limit=args.time_limit, verbose=False)
    dt = time.time() - t0
    avg_cpc = sum(r.cost for r in results) / len(results) / n
    print(f"GPU-Improved: N={n}, instances={len(instances)}, time={dt:.3f}s, avg CPC={avg_cpc:.4f}")


if __name__ == "__main__":
    main()

