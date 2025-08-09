#!/usr/bin/env python3
"""
GPU comparative study skeleton for 4 models.
This will be filled as models are GPU-optimized. Excludes legacy GAT+RL.
"""
import argparse
import torch

# TODO: import/create the 4 GPU-ready model factory entries
# from src.models.model_factory import make_model  # example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="*", default=["pointer_rl", "transformer_rl", "dyn_gt_small", "ablation_ref"], help="4 models to compare")
    parser.add_argument("--instances", type=int, default=1000)
    parser.add_argument("--n_nodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("GPU comparative skeleton")
    print("Device:", args.device)
    print("Models:", args.models)
    print("Instances:", args.instances, "n_nodes:", args.n_nodes, "batch:", args.batch_size)

    # Placeholder
    print("TODO: implement loading data, running 4 models, computing naive baseline, and plotting 3+1 figures.")


if __name__ == "__main__":
    main()

