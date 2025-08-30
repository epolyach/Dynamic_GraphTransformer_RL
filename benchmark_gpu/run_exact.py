#!/usr/bin/env python3
import argparse
from typing import List
from src.utils.config import load_config
from src.generator.generator import create_data_generator

# Placeholder GPU exact benchmark entrypoint that reuses canonical generator.
# TODO: integrate with GPU solvers in solvers/ (exact_gpu*, etc.)

def main():
    parser = argparse.ArgumentParser(description="GPU exact benchmark using canonical CVRP generator")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    gen = create_data_generator(config)
    instances: List[dict] = []
    instances = gen(batch_size=args.instances, epoch=1, seed=args.seed)
    print(f"Generated {len(instances)} instances (N={config['problem']['num_customers']}) for GPU benchmark.")
    print("TODO: integrate with GPU exact solvers and compute metrics.")

if __name__ == "__main__":
    main()

