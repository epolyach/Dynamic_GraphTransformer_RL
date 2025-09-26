#!/usr/bin/env python3
"""Analysis of overengineering in advanced_trainer_gpu.py"""

import re

# Read the file
with open('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL/training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()
    lines = content.split('\n')

# Analysis results
issues = []
suggestions = []

# 1. n_customers_tensor analysis
print("=" * 70)
print("ANALYSIS OF OVERENGINEERING IN advanced_trainer_gpu.py")
print("=" * 70)

print("\n1. n_customers_tensor Issue (Lines ~395-396):")
print("-" * 50)
print("CURRENT CODE:")
print("  n_customers_list = [len(instances[b]['coords']) - 1 for b in range(len(instances))]")
print("  n_customers_tensor = torch.tensor(n_customers_list, device=gpu_manager.device, dtype=torch.float32)")
print("\nISSUE:")
print("  - Creating a tensor for n_customers is unnecessary since:")
print("    a) It's used for element-wise division with rcosts (which is a tensor)")
print("    b) PyTorch automatically broadcasts scalars/lists during operations")
print("\nSUGGESTED SIMPLIFICATION:")
print("  # Option 1: Use the list directly (PyTorch will broadcast)")
print("  n_customers = [len(inst['coords']) - 1 for inst in instances]")
print("  cpc_vals = rcosts / torch.tensor(n_customers, device=rcosts.device)")
print("\n  # Option 2: If all instances have same size, use scalar")
print("  # n_customers = len(instances[0]['coords']) - 1  # if uniform")
print("  # cpc_vals = rcosts / n_customers")

# 2. Duplicate batch_cost calculations
print("\n2. Duplicate batch_cost Calculations (Lines ~399-403, 410-412):")
print("-" * 50)
print("ISSUE:")
print("  - batch_cost is calculated twice with identical logic")
print("  - Lines 399-403: First calculation")
print("  - Lines 410-412: Redundant recalculation (now commented out)")
print("\nSUGGESTED FIX:")
print("  - Already fixed in previous correction")
print("  - Keep only the first calculation")

# 3. Unnecessary tensor conversions
print("\n3. Unnecessary Tensor Conversions (Lines ~406, 415):")
print("-" * 50)
print("CURRENT CODE:")
print("  Line 406: rcosts = [rcosts[i] for i in range(len(rcosts))]")
print("  Line 415: costs_tensor = torch.stack(rcosts).to(dtype=torch.float32)")
print("\nISSUE:")
print("  - Converting tensor to list then back to tensor is inefficient")
print("  - rcosts is already a tensor from compute_route_cost_vectorized")
print("\nSUGGESTED SIMPLIFICATION:")
print("  # Remove line 406 entirely")
print("  # Line 415 becomes:")
print("  costs_tensor = rcosts.to(dtype=torch.float32)  # Already a tensor")

# 4. Repeated CPC calculation logic
print("\n4. Repeated CPC Calculation Logic:")
print("-" * 50)
print("ISSUE:")
print("  - CPC calculation logic is duplicated in training and validation")
print("  - Lines 399-403 (training) and lines 507-527 (validation)")
print("\nSUGGESTED SIMPLIFICATION:")
def_text = """
def compute_cpc(costs, n_customers, use_geometric_mean=False, device=None):
    \"\"\"Compute Cost Per Customer (CPC) with specified aggregation.\"\"\"
    if isinstance(n_customers, (list, tuple)):
        n_customers = torch.tensor(n_customers, device=device or costs.device, dtype=torch.float32)
    
    if use_geometric_mean:
        cpc_logs = torch.log(costs + 1e-10) - torch.log(n_customers)
        return torch.exp(cpc_logs.mean())
    else:
        return (costs / n_customers).mean()
"""
print(def_text)

# 5. Complex temperature scheduling
print("\n5. Overly Complex AdaptiveTemperatureScheduler (Lines ~42-72):")
print("-" * 50)
print("ISSUE:")
print("  - Complex performance tracking with windows")
print("  - Many parameters that might not significantly impact training")
print("\nSUGGESTED SIMPLIFICATION:")
simpler_temp = """
class SimpleTemperatureScheduler:
    def __init__(self, start=2.0, end=0.5, epochs=100):
        self.start = start
        self.end = end
        self.epochs = epochs
    
    def get_temp(self, epoch):
        # Linear or exponential decay
        alpha = epoch / self.epochs
        return self.start * (1 - alpha) + self.end * alpha
"""
print(simpler_temp)

# 6. Early stopping complexity
print("\n6. EarlyStopping Class Complexity (Lines ~74-102):")
print("-" * 50)
print("ISSUE:")
print("  - Deep copying entire model state dict on every improvement")
print("  - Memory intensive for large models")
print("\nSUGGESTED SIMPLIFICATION:")
print("  - Save only the checkpoint path, not the weights in memory")
print("  - Or use PyTorch's built-in checkpoint utilities")

# Summary
print("\n" + "=" * 70)
print("SUMMARY OF SIMPLIFICATION OPPORTUNITIES")
print("=" * 70)

simplifications = [
    "1. Remove unnecessary tensor creation for n_customers",
    "2. Eliminate duplicate batch_cost calculations (already fixed)",
    "3. Remove tensor->list->tensor conversions",
    "4. Extract CPC calculation into reusable function",
    "5. Simplify temperature scheduling logic",
    "6. Optimize early stopping memory usage",
    "7. Consider removing unused abstractions (DataLoaderGPU if redundant)",
    "8. Consolidate validation metric calculations"
]

print("\nKey Simplifications:")
for i, s in enumerate(simplifications, 1):
    print(f"  {i}. {s}")

print("\nEstimated Impact:")
print("  - Code reduction: ~100-150 lines")
print("  - Memory usage: Reduced by avoiding unnecessary tensor copies")
print("  - Performance: Slight improvement from fewer conversions")
print("  - Maintainability: Significantly improved")

print("\n" + "=" * 70)
