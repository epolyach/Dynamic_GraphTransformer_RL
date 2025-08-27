#!/usr/bin/env python3
"""Test the update_state function directly."""

import torch
import sys
sys.path.append('.')

from src.models.legacy_gat import GAT_Decoder

# Create decoder
decoder = GAT_Decoder(128, 128)

# Test data
demands = torch.tensor([[0., 18., 17., 16.]])  # depot=0, customers with demands
dynamic_capacity = torch.tensor([[12.]])  # Current capacity
max_capacity = torch.tensor([[30.]])  # Max capacity

print("Test 1: Visit depot (should reset capacity)")
print(f"  Before: dynamic_capacity={dynamic_capacity}")
index = torch.tensor([0])  # Visit depot
result = decoder.update_state(demands, dynamic_capacity.clone(), index, max_capacity)
print(f"  After visiting depot (node 0): {result}")
print(f"  Expected: [[30.]]")

print("\nTest 2: Visit customer")
dynamic_capacity = torch.tensor([[30.]])
print(f"  Before: dynamic_capacity={dynamic_capacity}")
index = torch.tensor([1])  # Visit customer 1 with demand 18
result = decoder.update_state(demands, dynamic_capacity.clone(), index, max_capacity)
print(f"  After visiting customer 1 (demand=18): {result}")
print(f"  Expected: [[12.]]")
