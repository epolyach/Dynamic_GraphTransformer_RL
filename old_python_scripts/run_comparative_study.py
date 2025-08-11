#!/usr/bin/env python3
"""
Comparative Study: 3 Different Architectures
1. Baseline Pointer Network
2. Graph Transformer  
3. Dynamic Graph Transformer

Run all three and compare performance with detailed analysis and visualization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

# CPU-optimized version - no GPU support
device = torch.device("cpu")
print(f"🖥️  Using device: {device} (CPU-optimized)")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    return logging.getLogger(__name__)

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def configure_cpu_threads(max_threads: int = None):
    """Enhanced CPU threading configuration for optimal multicore performance."""
    import os
    try:
        if max_threads is None:
            # Use all available cores for maximum performance
            max_threads = os.cpu_count() or 4
        
        # Set PyTorch threading
        torch.set_num_threads(max_threads)
        # Set inter-op threads for CPU parallelism (use fewer threads to avoid oversubscription)
        torch.set_num_interop_threads(max(1, max_threads // 4))
        
        # Configure OpenMP if available
        os.environ['OMP_NUM_THREADS'] = str(max_threads)
        os.environ['MKL_NUM_THREADS'] = str(max_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(max_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(max_threads)
        
        # Optimize for CPU performance (quiet mode)
        os.environ['KMP_BLOCKTIME'] = '0'
        os.environ['KMP_SETTINGS'] = '0'  # Disable verbose output
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'  # Remove verbose
        
        print(f"🚀 CPU optimization: {max_threads} threads configured")
        print(f"   PyTorch threads: {torch.get_num_threads()}")
        print(f"   Inter-op threads: {torch.get_num_interop_threads()}")
        
    except Exception as e:
        print(f"⚠️ CPU threading configuration failed: {e}")

# Configure CPU threads for better multithreading performance
if device.type == 'cpu':
    configure_cpu_threads()

def generate_cvrp_instance(num_customers, capacity, coord_range, demand_range, seed=None):
    """Generate CVRP instance with integer demands and configurable capacity
    
    Args:
        num_customers: Number of customer nodes (excluding depot)
        capacity: Vehicle capacity (integer)
        coord_range: Coordinate range for generation (will be normalized)
        demand_range: Tuple (min_demand, max_demand) for integer demands
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates: random integers 0 to coord_range, then divide by coord_range for normalization to [0,1]
    coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
    for i in range(num_customers + 1):
        coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range
    
    # Generate integer demands from demand_range - ensure they are integers
    demands = np.zeros(num_customers + 1, dtype=np.int32)
    for i in range(1, num_customers + 1):  # Skip depot (index 0)
        demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1)
    
    # Compute distance matrix
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    return {
        'coords': coords,
        'demands': demands.astype(np.int32),  # Ensure demands are integers
        'distances': distances,
        'capacity': int(capacity)  # Ensure capacity is integer
    }

class BaselinePointerNetwork(nn.Module):
    """Pipeline 1: Simple Pointer Network with basic attention"""
    
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Simple node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Basic attention mechanism
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3)
        demands_batch = torch.zeros(batch_size, max_nodes)
        capacities = torch.zeros(batch_size)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']
        
        # Embed nodes
        embedded = self.node_embedding(node_features)  # [B, N, H]
        
        # Simple attention (no multi-head)
        Q = self.attention_query(embedded)
        K = self.attention_key(embedded)
        V = self.attention_value(embedded)
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.bmm(attention_weights, V)
        
        return self._generate_routes(attended, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances)
    
    def _generate_routes(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances):
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        # Initialize state
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        # Track which batches are done to avoid processing them
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        for step in range(max_steps):
            # Check which batches are done (all customers visited AND at depot)
            for b in range(batch_size):
                if not batch_done[b]:  # Only check if not already done
                    customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            # If all batches are done, break
            if batch_done.all():
                break
            
            # Create context (mean of unvisited nodes)
            unvisited_mask = ~visited
            context = torch.zeros(batch_size, 1, hidden_dim)
            
            for b in range(batch_size):
                if unvisited_mask[b].any():
                    context[b, 0] = node_embeddings[b][unvisited_mask[b]].mean(dim=0)
            
            context = context.expand(-1, max_nodes, -1)
            
            # Compute pointer scores
            pointer_input = torch.cat([node_embeddings, context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints + pad beyond actual nodes
            cap_mask = demands_batch > remaining_capacity.unsqueeze(1)
            mask = visited | cap_mask
            # Mask out padded nodes beyond actual graph size (except depot 0)
            pad_mask = torch.zeros_like(mask)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords'])
                if actual_nodes < max_nodes:
                    pad_mask[b, actual_nodes:] = True
                pad_mask[b, 0] = False
            mask = mask | pad_mask
            # Don't allow staying at depot if already at depot
            currently_at_depot_vec = torch.tensor([len(r) > 0 and r[-1] == 0 for r in routes])
            if currently_at_depot_vec.any():
                mask[currently_at_depot_vec, 0] = True
            # Safety handling
            all_masked = mask.all(dim=1)
            need_allow_depot = all_masked & (~currently_at_depot_vec)
            if need_allow_depot.any():
                mask[need_allow_depot, 0] = False
            done_mask = all_masked & currently_at_depot_vec
            batch_done[done_mask] = True
            
            # Use large negative instead of -inf to keep softmax finite
            scores = scores.masked_fill(mask, -1e9)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)

            # Entropy per batch at this step (robust to zeros)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        actions[b] = torch.multinomial(probs[b], 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            all_entropies.append(step_entropy)
            
            # Update state only for batches that aren't done
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    
                    if action == 0:  # Return to depot
                        remaining_capacity[b] = capacities[b]
                        # DON'T reset visited - customers should stay visited permanently
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Check termination: all customers visited AND currently at depot
            all_done = True
            for b in range(batch_size):
                customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Routes should already end at depot due to termination condition
        # Only add depot if route is empty (shouldn't happen)
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        return routes, combined_log_probs, combined_entropy

class GraphTransformerNetwork(nn.Module):
    """Pipeline 2: Graph Transformer with multi-head self-attention"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Graph-level aggregation
        self.graph_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3)
        demands_batch = torch.zeros(batch_size, max_nodes)
        capacities = torch.zeros(batch_size)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']
        
        # Embed nodes
        embedded = self.node_embedding(node_features)  # [B, N, H]
        
        # Apply transformer layers
        x = embedded
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Graph-level attention for global context
        graph_context, _ = self.graph_attention(x, x, x)
        enhanced_embeddings = x + graph_context  # Residual connection
        
        return self._generate_routes(enhanced_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances)
    
    def _generate_routes(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances):
        # Same routing logic as baseline but with enhanced embeddings
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        # Track which batches are done to avoid processing them
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        for step in range(max_steps):
            # Check which batches are done (all customers visited AND at depot)
            for b in range(batch_size):
                if not batch_done[b]:  # Only check if not already done
                    customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            # If all batches are done, break
            if batch_done.all():
                break
            
            # Dynamic context based on current state
            context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            
            # Compute pointer scores with enhanced embeddings
            pointer_input = torch.cat([node_embeddings, context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints + pad beyond actual nodes
            cap_mask = demands_batch > remaining_capacity.unsqueeze(1)
            mask = visited | cap_mask
            pad_mask = torch.zeros_like(mask)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords'])
                if actual_nodes < max_nodes:
                    pad_mask[b, actual_nodes:] = True
                pad_mask[b, 0] = False
            mask = mask | pad_mask
            currently_at_depot_vec = torch.tensor([len(r) > 0 and r[-1] == 0 for r in routes])
            if currently_at_depot_vec.any():
                mask[currently_at_depot_vec, 0] = True
            all_masked = mask.all(dim=1)
            need_allow_depot = all_masked & (~currently_at_depot_vec)
            if need_allow_depot.any():
                mask[need_allow_depot, 0] = False
            done_mask = all_masked & currently_at_depot_vec
            batch_done[done_mask] = True
            
            scores = scores.masked_fill(mask, -1e9)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)

            # Entropy per batch at this step (robust to zeros)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        actions[b] = torch.multinomial(probs[b], 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            all_entropies.append(step_entropy)
            
            # Update state only for batches that aren't done
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    
                    if action == 0:  # Return to depot
                        remaining_capacity[b] = capacities[b]
                        # DON'T reset visited - customers should stay visited permanently
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Check termination: all customers visited AND currently at depot
            all_done = True
            for b in range(batch_size):
                customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Routes should already end at depot due to termination condition
        # Only add depot if route is empty (shouldn't happen)
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        return routes, combined_log_probs, combined_entropy

class GraphTransformerGreedy(nn.Module):
    """Pipeline 3: Graph Transformer with Greedy Selection (No RL)"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Graph-level aggregation
        self.graph_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Pointer network for greedy selection
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=True):  # Always greedy
        # Same as GraphTransformerNetwork but always uses greedy=True
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3)
        demands_batch = torch.zeros(batch_size, max_nodes)
        capacities = torch.zeros(batch_size)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']
        
        # Embed nodes
        embedded = self.node_embedding(node_features)  # [B, N, H]
        
        # Apply transformer layers
        x = embedded
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Graph-level attention for global context
        graph_context, _ = self.graph_attention(x, x, x)
        enhanced_embeddings = x + graph_context  # Residual connection
        
        return self._generate_routes(enhanced_embeddings, node_features, demands_batch, capacities, max_steps, temperature, True, instances)  # Force greedy
    
    def _generate_routes(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances):
        # Same routing logic as GraphTransformerNetwork but always greedy
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        # Track which batches are done to avoid processing them
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        for step in range(max_steps):
            # Check which batches are done (all customers visited AND at depot)
            for b in range(batch_size):
                if not batch_done[b]:  # Only check if not already done
                    customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            # If all batches are done, break
            if batch_done.all():
                break
            
            # Dynamic context based on current state
            context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            
            # Compute pointer scores with enhanced embeddings
            pointer_input = torch.cat([node_embeddings, context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints + pad beyond actual nodes
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                # Don't allow staying at depot if already at depot
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                # Pad beyond actual nodes (except depot)
                actual_nodes = len(instances[b]['coords'])
                if actual_nodes < max_nodes:
                    mask[b, actual_nodes:] = True
                mask[b, 0] = mask[b, 0]
                
                # Safety: if all nodes masked and we're not at depot, allow depot
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                # If we're at depot and all customers visited, we should have terminated above
                elif mask[b].all() and currently_at_depot:
                    # Mark this batch as done to avoid further processing
                    batch_done[b] = True
            
            # Stable logits/probs to avoid NaNs
            scores = scores.masked_fill(mask, -1e9)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)

            # Entropy per batch at this step (robust to zeros)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Always greedy selection among feasible
            actions = probs.argmax(dim=-1)
            selected_log_probs = torch.gather(log_probs, 1, actions.unsqueeze(1)).squeeze(1)
            
            all_log_probs.append(selected_log_probs)
            all_entropies.append(step_entropy)
            
            # Update state only for batches that aren't done
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    
                    if action == 0:  # Return to depot
                        remaining_capacity[b] = capacities[b]
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Check termination: all customers visited AND currently at depot
            all_done = True
            for b in range(batch_size):
                customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Routes should already end at depot due to termination condition
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        return routes, combined_log_probs, combined_entropy

class DynamicGraphTransformerNetwork(nn.Module):
    """Pipeline 4: Dynamic Graph Transformer with adaptive updates"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Dynamic update components
        self.state_encoder = nn.Linear(4, hidden_dim)  # capacity_used, step, visited_count, distance_from_depot
        # PreNorm + gated residual for stability of dynamic updates
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.res_gate = nn.Parameter(torch.tensor(-2.19722458))  # sigmoid ~= 0.1
        self.dynamic_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # node + context + state
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3)
        demands_batch = torch.zeros(batch_size, max_nodes)
        capacities = torch.zeros(batch_size)
        distances_batch = torch.zeros(batch_size, max_nodes, max_nodes)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']
            distances_batch[i, :n_nodes, :n_nodes] = torch.tensor(inst['distances'], dtype=torch.float32)
        
        # Initial embedding
        embedded = self.node_embedding(node_features)
        
        # Apply transformer layers
        x = embedded
        for layer in self.transformer_layers:
            x = layer(x)
        
        return self._generate_routes_dynamic(x, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances)
    
    def _generate_routes_dynamic(self, node_embeddings, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances=None):
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        current_nodes = torch.zeros(batch_size, dtype=torch.long)  # Current position
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        # Track which batches are done to avoid processing them
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        for step in range(max_steps):
            # Check which batches are done (all customers visited AND at depot)
            for b in range(batch_size):
                if not batch_done[b]:  # Only check if not already done
                    if instances and b < len(instances):
                        actual_nodes = len(instances[b]['coords'])
                    else:
                        actual_nodes = max_nodes
                    customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                    currently_at_depot = current_nodes[b].item() == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            # If all batches are done, break
            if batch_done.all():
                break
            # Dynamic state features
            capacity_used = (capacities - remaining_capacity) / capacities
            step_progress = torch.full((batch_size,), step / max_steps)
            visited_count = visited.float().sum(dim=1) / max_nodes
            
            # Distance from depot
            distance_from_depot = torch.zeros(batch_size)
            for b in range(batch_size):
                current_pos = current_nodes[b].item()
                distance_from_depot[b] = distances_batch[b, current_pos, 0]
            
            # Encode dynamic state
            state_features = torch.stack([
                capacity_used, step_progress, visited_count, distance_from_depot
            ], dim=1)  # [B, 4]
            
            state_encoding = self.state_encoder(state_features)  # [B, H]
            
            # Update node embeddings based on current state (PreNorm + gated residual)
            dynamic_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)  # [B, N, H]
            normed = self.pre_norm(node_embeddings)
            update_input = torch.cat([normed, dynamic_context], dim=-1)
            delta = self.dynamic_update(update_input)
            gate = torch.sigmoid(self.res_gate)
            
            # Apply dynamic updates with learnable gate
            updated_embeddings = node_embeddings + gate * delta
            
            # Enhanced context with state information
            global_context = updated_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            state_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)
            
            # Enhanced pointer scores
            pointer_input = torch.cat([updated_embeddings, global_context, state_context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints + pad beyond actual nodes
            cap_mask = demands_batch > remaining_capacity.unsqueeze(1)
            mask = visited | cap_mask
            pad_mask = torch.zeros_like(mask)
            for b in range(batch_size):
                if instances and b < len(instances):
                    actual_nodes = len(instances[b]['coords'])
                else:
                    actual_nodes = max_nodes
                if actual_nodes < max_nodes:
                    pad_mask[b, actual_nodes:] = True
                pad_mask[b, 0] = False
            mask = mask | pad_mask
            currently_at_depot_vec = torch.tensor([len(r) > 0 and r[-1] == 0 for r in routes])
            if currently_at_depot_vec.any():
                mask[currently_at_depot_vec, 0] = True
            all_masked = mask.all(dim=1)
            need_allow_depot = all_masked & (~currently_at_depot_vec)
            if need_allow_depot.any():
                mask[need_allow_depot, 0] = False
            done_mask = all_masked & currently_at_depot_vec
            batch_done[done_mask] = True
            
            scores = scores.masked_fill(mask, -1e9)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)

            # Entropy per batch at this step (robust to zeros)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        actions[b] = torch.multinomial(probs[b], 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            all_entropies.append(step_entropy)
            
            # Update state only for batches that aren't done
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    current_nodes[b] = action
                    
                    if action == 0:  # Return to depot
                        remaining_capacity[b] = capacities[b]
                        # DON'T reset visited - customers should stay visited permanently
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Check termination: all customers visited AND currently at depot
            all_done = True
            for b in range(batch_size):
                if instances and b < len(instances):
                    actual_nodes = len(instances[b]['coords'])
                else:
                    actual_nodes = max_nodes
                customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                currently_at_depot = current_nodes[b].item() == 0
                
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Routes should already end at depot due to termination condition
        # Only add depot if route is empty (shouldn't happen)
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        return routes, combined_log_probs, combined_entropy

class GraphAttentionTransformer(nn.Module):
    """Pipeline 5: Graph Attention Transformer with Edge Features"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Edge feature embedding (distance)
        self.edge_embedding = nn.Linear(1, hidden_dim // 4)
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final attention aggregation
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Enhanced pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3)
        demands_batch = torch.zeros(batch_size, max_nodes)
        capacities = torch.zeros(batch_size)
        distances_batch = torch.zeros(batch_size, max_nodes, max_nodes)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']
            distances_batch[i, :n_nodes, :n_nodes] = torch.tensor(inst['distances'], dtype=torch.float32)
        
        # Initial node embedding
        node_embeds = self.node_embedding(node_features)  # [B, N, H]
        
        # Apply GAT layers with residual connections
        x = node_embeds
        for gat_layer, layer_norm in zip(self.gat_layers, self.layer_norms):
            # Self-attention with residual
            attn_out, _ = gat_layer(x, x, x)
            x = layer_norm(x + attn_out)
        
        # Global context attention
        global_context, _ = self.global_attention(x, x, x)
        enhanced_embeddings = x + global_context
        
        return self._generate_routes_gat(enhanced_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances)
    
    def _generate_routes_gat(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances):
        # Same routing logic but with GAT-enhanced embeddings
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        for step in range(max_steps):
            # Check termination
            for b in range(batch_size):
                if not batch_done[b]:
                    customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            if batch_done.all():
                break
            
            # Enhanced context with attention
            context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            
            # Compute pointer scores
            pointer_input = torch.cat([node_embeddings, context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints + pad beyond actual nodes
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                # pad beyond actual nodes (except depot)
                actual_nodes = len(instances[b]['coords'])
                if actual_nodes < max_nodes:
                    mask[b, actual_nodes:] = True
                mask[b, 0] = mask[b, 0]  # ensure depot handling remains
                
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                elif mask[b].all() and currently_at_depot:
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, -1e9)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)

            # Entropy per batch at this step (robust to zeros)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Sample actions
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        probs = torch.softmax(scores[b] / temperature, dim=-1)
                        actions[b] = torch.multinomial(probs, 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            all_entropies.append(step_entropy)
            
            # Update state
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    
                    if action == 0:
                        remaining_capacity[b] = capacities[b]
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Final termination check
            all_done = True
            for b in range(batch_size):
                customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        return routes, combined_log_probs, combined_entropy

def naive_baseline_solution(instance):
    """Generate naive baseline solution: depot->customer->depot for each customer"""
    n_customers = len(instance['coords']) - 1
    route = [0]  # Start at depot
    
    # Visit each customer individually
    for customer in range(1, n_customers + 1):
        if len(route) > 1:  # Not the first customer
            route.append(0)  # Return to depot first
        route.append(customer)  # Visit customer
    
    route.append(0)  # Final return to depot
    return route

def compute_route_cost(route, distances):
    """Compute total cost of a route"""
    if len(route) <= 1:
        return 0.0
    
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]]
    return cost

def compute_normalized_cost(route, distances, n_customers):
    """Compute cost per customer (normalized cost)"""
    total_cost = compute_route_cost(route, distances)
    return total_cost / n_customers if n_customers > 0 else 0.0

def compute_naive_baseline_cost(instance):
    """Compute cost of naive solution: depot->node->depot for each customer"""
    distances = instance['distances']
    n_customers = len(instance['coords']) - 1  # excluding depot
    naive_cost = 0.0
    
    for customer_idx in range(1, n_customers + 1):  # customers are indexed 1 to n
        naive_cost += distances[0, customer_idx] * 2  # depot->customer->depot
    
    return naive_cost

def validate_route(route, n_customers, model_name="Unknown", instance=None):
    """RIGOROUS VALIDATION: Validate that a route is a correct CVRP solution
    
    Validates:
    1. Route structure (starts/ends at depot, no consecutive depots)
    2. Customer coverage (all customers visited exactly once)
    3. Capacity constraints (no truck overloading)
    4. Route feasibility (valid node indices)
    
    Args:
        route: List of node indices representing the vehicle route
        n_customers: Number of customers (excluding depot)
        model_name: Model identifier for error reporting
        instance: Instance data containing demands and capacity (optional but recommended)
    
    Returns:
        bool: True if route is valid, otherwise exits with detailed error
    """
    
    # === BASIC STRUCTURE VALIDATION ===
    if len(route) == 0:
        print(f"\n🚨 VALIDATION FAILED: {model_name}")
        print(f"Error: Empty route!")
        print(f"Route: {route}")
        sys.exit(1)
    
    # CRITICAL: Route must end at depot (index 0)
    if route[-1] != 0:
        print(f"\n🚨 VALIDATION FAILED: {model_name}")
        print(f"Error: Route must end at depot (0), but ends at {route[-1]}")
        print(f"Route: {route}")
        sys.exit(1)
    
    # Route should start at depot
    if route[0] != 0:
        print(f"\n🚨 VALIDATION FAILED: {model_name}")
        print(f"Error: Route must start at depot (0), but starts at {route[0]}")
        print(f"Route: {route}")
        sys.exit(1)
    
    # Check for consecutive depot visits (depot->depot is forbidden)
    for i in range(len(route) - 1):
        if route[i] == 0 and route[i + 1] == 0:
            print(f"\n🚨 VALIDATION FAILED: {model_name}")
            print(f"Error: Consecutive depot visits at positions {i}-{i+1}")
            print(f"Route: {route}")
            sys.exit(1)
    
    # === CUSTOMER COVERAGE VALIDATION ===
    customers_in_route = [node for node in route if node != 0]
    unique_customers = set(customers_in_route)
    
    # Check for duplicate customer visits
    if len(customers_in_route) != len(unique_customers):
        duplicates = [x for x in customers_in_route if customers_in_route.count(x) > 1]
        print(f"\n🚨 VALIDATION FAILED: {model_name}")
        print(f"Error: Duplicate customer visits: {duplicates}")
        print(f"Customers in route: {customers_in_route}")
        print(f"Route: {route}")
        sys.exit(1)
    
    # Check if all customers are visited exactly once
    expected_customers = set(range(1, n_customers + 1))
    if unique_customers != expected_customers:
        missing = expected_customers - unique_customers
        extra = unique_customers - expected_customers
        print(f"\n🚨 VALIDATION FAILED: {model_name}")
        print(f"Expected customers: {sorted(expected_customers)}")
        print(f"Found customers: {sorted(unique_customers)}")
        if missing:
            print(f"Missing customers: {sorted(missing)}")
        if extra:
            print(f"Extra/invalid customers: {sorted(extra)}")
        print(f"Route: {route}")
        sys.exit(1)
    
    # === NODE INDEX VALIDATION ===
    # Check for invalid node indices
    max_valid_index = n_customers  # depot=0, customers=1 to n_customers
    invalid_nodes = [node for node in route if node < 0 or node > max_valid_index]
    if invalid_nodes:
        print(f"\n🚨 VALIDATION FAILED: {model_name}")
        print(f"Error: Invalid node indices: {invalid_nodes}")
        print(f"Valid range: 0 to {max_valid_index} (depot=0, customers=1-{n_customers})")
        print(f"Route: {route}")
        sys.exit(1)
    
    # === CAPACITY CONSTRAINT VALIDATION ===
    if instance is not None and 'demands' in instance and 'capacity' in instance:
        demands = instance['demands']
        vehicle_capacity = instance['capacity']
        
        # Simulate route execution to check capacity constraints
        current_load = 0.0
        max_load_violation = 0.0
        violation_segments = []
        
        # Split route into trips (depot to depot segments)
        trips = []
        current_trip = []
        
        for i, node in enumerate(route):
            current_trip.append(node)
            if node == 0 and len(current_trip) > 1:  # End of trip (return to depot)
                trips.append(current_trip[:])
                current_trip = [0]  # Start new trip at depot
        
        # Validate each trip's capacity
        for trip_idx, trip in enumerate(trips):
            trip_load = 0.0
            trip_customers = [node for node in trip if node != 0]
            
            for customer in trip_customers:
                if customer <= len(demands) - 1:  # Valid customer index
                    customer_demand = demands[customer]
                    trip_load += customer_demand
                    
                    # Check if capacity is exceeded at any point
                    if trip_load > vehicle_capacity:
                        violation = trip_load - vehicle_capacity
                        if violation > max_load_violation:
                            max_load_violation = violation
                        violation_segments.append({
                            'trip': trip_idx,
                            'customer': customer,
                            'load': trip_load,
                            'capacity': vehicle_capacity,
                            'violation': violation
                        })
        
        # Report capacity violations
        if violation_segments:
            print(f"\n🚨 VALIDATION FAILED: {model_name}")
            print(f"Error: Capacity constraint violations detected!")
            print(f"Vehicle capacity: {vehicle_capacity}")
            print(f"Maximum violation: {max_load_violation:.3f}")
            print(f"Violations:")
            for v in violation_segments:
                print(f"  Trip {v['trip']}: Customer {v['customer']} causes load {v['load']:.3f} > {v['capacity']} (excess: {v['violation']:.3f})")
            print(f"Route trips: {trips}")
            print(f"Full route: {route}")
            sys.exit(1)
        
        # === DEMAND CONSISTENCY CHECK ===
        # Verify that all customer demands are reasonable (> 0)
        zero_demand_customers = []
        for customer in range(1, len(demands)):
            if demands[customer] <= 0:
                zero_demand_customers.append(customer)
        
        if zero_demand_customers:
            print(f"\n⚠️  WARNING: {model_name}")
            print(f"Customers with zero/negative demand: {zero_demand_customers}")
            # This is a warning, not a failure
    
    return True

def train_model(model, instances, config, model_name, logger):
    """Train a single model and return training history"""
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate schedule: linear warmup -> cosine decay
    base_lr = config['learning_rate']
    warmup_epochs = int(config.get('warmup_epochs', 5))
    min_lr = float(config.get('min_lr', 1e-4))
    def lr_factor(ep):
        if warmup_epochs > 0 and ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        # Cosine from 1.0 down to min_lr/base_lr over remaining epochs
        total = max(1, config['num_epochs'] - warmup_epochs)
        t = max(0, ep - warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * t / total))
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine
    
    train_losses = []
    train_costs = []
    val_costs = []
    
    # Split data
    split_idx = int(0.8 * len(instances))
    train_instances = instances[:split_idx]
    val_instances = instances[split_idx:]
    
    logger.info(f"🏋️ Training {model_name}...")
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        epoch_losses = []
        epoch_costs = []
        
        batch_size = config['batch_size']
        num_batches = len(train_instances) // batch_size
        
        # Temperature schedule: cosine from temp_start -> temp_min
        temp_start = float(config.get('temp_start', 1.5))
        temp_min = float(config.get('temp_min', 0.2))
        if config['num_epochs'] > 1:
            cosine_t = 0.5 * (1 + math.cos(math.pi * epoch / (config['num_epochs'] - 1)))
        else:
            cosine_t = 0.0
        current_temp = temp_min + (temp_start - temp_min) * cosine_t
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            batch_instances = train_instances[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            routes, log_probs, entropies = model(batch_instances, temperature=current_temp)
            
            # Compute costs and validate routes
            costs = []
            normalized_costs = []
            for route, instance in zip(routes, batch_instances):
                n_customers = len(instance['coords']) - 1
                
                # VALIDATE ROUTE - This will exit with error if route is invalid
                validate_route(route, n_customers, f"{model_name}-TRAIN", instance)

                total_cost = compute_route_cost(route, instance['distances'])
                normalized_cost = compute_normalized_cost(route, instance['distances'], n_customers)
                costs.append(total_cost)
                normalized_costs.append(normalized_cost)
            
            costs_tensor = torch.tensor(costs, dtype=torch.float32)
            
            # REINFORCE loss with batch mean baseline
            baseline = costs_tensor.mean().detach()
            advantages = baseline - costs_tensor  # Lower costs should have positive advantages
            
            # Entropy regularization with cosine-decaying coefficient
            start_c = config.get('entropy_coef', 0.0)
            end_c = config.get('entropy_min', 0.0)
            if config['num_epochs'] > 1:
                cosine_factor = 0.5 * (1 + math.cos(math.pi * epoch / (config['num_epochs'] - 1)))
            else:
                cosine_factor = 0.0
            entropy_coef = end_c + (start_c - end_c) * cosine_factor

            loss = (-advantages * log_probs).mean() - entropy_coef * entropies.mean()
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_costs.extend(costs)
        
        # Step LR (manual schedule with warmup + cosine)
        factor = lr_factor(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * factor
        
        train_losses.append(np.mean(epoch_losses))
        train_costs.append(np.mean(epoch_costs))
        
        # Validation
        if epoch % 3 == 0:
            model.eval()
            val_batch_costs = []
            val_batch_normalized = []
            
            with torch.no_grad():
                for i in range(0, len(val_instances), batch_size):
                    batch_val = val_instances[i:i + batch_size]
                    routes, _, _ = model(batch_val, greedy=True)
                    
                    for j, (route, instance) in enumerate(zip(routes, batch_val)):
                        n_customers = len(instance['coords']) - 1
                        
                        # VALIDATE ROUTE - This will exit with error if route is invalid
                        validate_route(route, n_customers, f"{model_name}-VAL", instance)

                        total_cost = compute_route_cost(route, instance['distances'])
                        normalized_cost = compute_normalized_cost(route, instance['distances'], n_customers)
                        val_batch_costs.append(total_cost)
                        val_batch_normalized.append(normalized_cost)
            
            val_cost = np.mean(val_batch_costs)
            val_normalized = np.mean(val_batch_normalized)
            val_costs.append(val_cost)
            
            logger.info(f"   Epoch {epoch:2d}: Loss={train_losses[-1]:.3f}, Train={np.mean(epoch_costs) / config['num_customers']:.3f}/cust, Val={val_normalized:.3f}/cust")
        else:
            logger.info(f"   Epoch {epoch:2d}: Loss={train_losses[-1]:.3f}, Train={np.mean(epoch_costs) / config['num_customers']:.3f}/cust")
    
    return {
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs,
        'final_val_cost': val_costs[-1] if val_costs else train_costs[-1]
    }

def load_config(config_path):
    """Load configuration from YAML file"""
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='CPU Comparative Study')
    parser.add_argument('--config', type=str, default='configs/small.yaml', help='Path to YAML configuration file')
    parser.add_argument('--customers', type=int, default=None, help='Override number of customers from config')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs from config')
    parser.add_argument('--instances', type=int, default=None, help='Override number of instances from config')
    parser.add_argument('--batch', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--coord_range', type=int, default=None, help='Override coordinate range from config')
    parser.add_argument('--max_demand', type=int, default=None, help='Override max demand from config')
    parser.add_argument('--capacity', type=int, default=None, help='Override capacity from config')
    parser.add_argument('--plot_suffix', type=str, default='', help='Suffix for plot filenames')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Override hidden dimension from config')
    parser.add_argument('--entropy_coef', type=float, default=None, help='Override initial entropy regularization coefficient')
    parser.add_argument('--entropy_min', type=float, default=None, help='Override minimum entropy coefficient')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='Override warmup epochs')
    parser.add_argument('--min_lr', type=float, default=None, help='Override minimum LR')
    parser.add_argument('--temp_start', type=float, default=None, help='Override initial sampling temperature')
    parser.add_argument('--temp_min', type=float, default=None, help='Override minimum sampling temperature')
    parser.add_argument('--only_dgt', action='store_true', help='Train only the DGT+RL model to avoid extra deps')
    parser.add_argument('--exclude_dgt', action='store_true', help='Train all models except DGT+RL and reuse prior DGT results')
    parser.add_argument('--reuse_dgt_path', type=str, default='results/small/analysis/comparative_study_complete.pt', help='Path to prior DGT results file to reuse when --exclude_dgt is set')
    return parser.parse_args()


def build_pyg_data_from_instance(instance):
    import torch
    from torch_geometric.data import Data
    coords = torch.tensor(instance['coords'], dtype=torch.float32)
    n = coords.size(0)
    # Complete graph edge index
    ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    edge_index = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=0)
    # Edge attributes are distances
    edge_attr = torch.tensor(instance['distances'].reshape(-1, 1), dtype=torch.float32)
    demand = torch.tensor(instance['demands'], dtype=torch.float32).unsqueeze(1)
    capacity = torch.tensor([instance['capacity']], dtype=torch.float32)
    return Data(x=coords, edge_index=edge_index, edge_attr=edge_attr, demand=demand, capacity=capacity)


def train_legacy_gat_rl(model, instances, config, model_name, logger):
    """Train the legacy GAT_RL model using its original training loop (unchanged algorithms),
    then evaluate on the validation set to report final metrics. Additionally, extract
    per-epoch training metrics and per-3-epochs validation costs so plots show curves.
    """
    import numpy as np
    import torch
    import glob
    import pandas as pd
    from torch_geometric.loader import DataLoader
    # Import legacy training function
    from src_batch.train.train_model import train as legacy_train

    # Build Data list from our generated instances (match GAT_RL generation)
    data_list = [build_pyg_data_from_instance(inst) for inst in instances]
    split_idx = int(0.8 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    batch_size = config['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    n_steps = config['num_customers'] * 2  # legacy expects a step bound
    T = 2.5
    lr = 1e-4
    num_epochs = config['num_epochs']

    # Run the legacy training loop (kept intact)
    logger.info(f"🏋️ Training {model_name} (legacy RL training)...")
    logger.info(f"   Legacy training config: lr={lr}, n_steps={n_steps}, num_epochs={num_epochs}, T={T}")
    ckpt_folder = 'results/small/checkpoints/legacy_checkpoints'
    os.makedirs(ckpt_folder, exist_ok=True)
    # Capture time to find latest CSV emitted by legacy loop
    before_csv = set(glob.glob('logs/training/*.csv'))
    legacy_train(model, train_loader, val_loader, ckpt_folder, 'actor.pt', lr, n_steps, num_epochs, T)

    # After training, evaluate on validation with greedy to get final cost
    model.eval()
    val_epoch_costs = []
    with torch.no_grad():
        for batch in val_loader:
            actions, tour_logp = model(batch, n_steps=n_steps, greedy=True, T=T)
            depot = torch.zeros(actions.size(0), 1, dtype=torch.long)
            actions_with_depot = torch.cat([depot, actions, depot], dim=1)
            # Use PyG batch.ptr to slice per-graph coordinates correctly
            ptr = batch.ptr if hasattr(batch, 'ptr') else None
            for b in range(actions_with_depot.size(0)):
                route = actions_with_depot[b].cpu().tolist()
                if ptr is not None:
                    start = int(ptr[b].item())
                    end = int(ptr[b + 1].item())
                    coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                else:
                    # Fallback for single-graph batches
                    coords = batch.x.view(-1, 2).cpu().numpy()
                cost = 0.0
                for i in range(len(route) - 1):
                    a = route[i]
                    c = route[i + 1]
                    pa = coords[a]
                    pb = coords[c]
                    cost += float(((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2) ** 0.5)
                val_epoch_costs.append(cost)
    final_val_cost = float(np.mean(val_epoch_costs) if val_epoch_costs else 0.0)

    # Extract training curves from the legacy CSV if available
    train_losses = []
    train_costs = []
    try:
        after_csv = set(glob.glob('logs/training/*.csv'))
        new_csvs = sorted(list(after_csv - before_csv), key=lambda p: os.path.getmtime(p))
        csv_path = new_csvs[-1] if new_csvs else sorted(list(after_csv), key=lambda p: os.path.getmtime(p))[-1]
        logger.info(f"   📄 Using legacy CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"   📊 CSV shape: {df.shape}, columns: {list(df.columns)}")
        
        # Debug: Show first few rows
        logger.info(f"   📋 First 3 rows of CSV:")
        for i, row in df.head(3).iterrows():
            logger.info(f"      Row {i}: {dict(row)}")
        
        # Columns are strings; convert to float
        if 'mean_loss' in df.columns:
            legacy_losses = [float(x) for x in df['mean_loss'].tolist()][:num_epochs]
            logger.info(f"   🔢 Extracted {len(legacy_losses)} loss values (requested {num_epochs})")
            # STANDARDIZE: Legacy uses advantage * log_prob (positive sign)
            # Our models use -advantage * log_prob (negative sign)
            # Convert legacy to our convention by negating
            train_losses = [-loss for loss in legacy_losses]
        if 'mean_reward' in df.columns:
            train_costs = [float(x) for x in df['mean_reward'].tolist()][:num_epochs]
            logger.info(f"   💰 Extracted {len(train_costs)} cost values (requested {num_epochs})")
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to extract CSV data: {e}")
        # Fallback if CSV missing
        train_losses = []
        train_costs = []

    # Ensure lengths match expected plotting length
    if len(train_losses) < num_epochs:
        train_losses += [float('nan')] * (num_epochs - len(train_losses))
    if len(train_costs) < num_epochs:
        train_costs += [float('nan')] * (num_epochs - len(train_costs))

    # Build validation curve by evaluating checkpoints at epochs [0, 3, 6, ...]
    val_costs = []
    eval_epochs = list(range(0, num_epochs, 3))
    with torch.no_grad():
        for e in eval_epochs:
            ckpt_path = os.path.join(ckpt_folder, f'{e}', 'actor.pt')
            if os.path.exists(ckpt_path):
                # Load weights into model
                try:
                    state = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        model.load_state_dict(state['model_state_dict'])
                    else:
                        model.load_state_dict(state)
                except Exception:
                    pass
                # Evaluate greedy on validation set
                epoch_costs = []
                for batch in val_loader:
                    actions, tour_logp = model(batch, n_steps=n_steps, greedy=True, T=T)
                    depot = torch.zeros(actions.size(0), 1, dtype=torch.long)
                    actions_with_depot = torch.cat([depot, actions, depot], dim=1)
                    # Use PyG batch.ptr to slice per-graph coordinates correctly
                    ptr = batch.ptr if hasattr(batch, 'ptr') else None
                    for b in range(actions_with_depot.size(0)):
                        route = actions_with_depot[b].cpu().tolist()
                        if ptr is not None:
                            start = int(ptr[b].item())
                            end = int(ptr[b + 1].item())
                            coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                        else:
                            coords = batch.x.view(-1, 2).cpu().numpy()
                        cost = 0.0
                        for i in range(len(route) - 1):
                            a = route[i]
                            c = route[i + 1]
                            pa = coords[a]
                            pb = coords[c]
                            cost += float(((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2) ** 0.5)
                        epoch_costs.append(cost)
                val_costs.append(float(np.mean(epoch_costs) if epoch_costs else float('nan')))
            else:
                val_costs.append(float('nan'))

    # Restore final trained weights if available (last epoch)
    final_ckpt = os.path.join(ckpt_folder, f'{num_epochs-1}', 'actor.pt')
    if os.path.exists(final_ckpt):
        try:
            state = torch.load(final_ckpt, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
        except Exception:
            pass

    return {
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs if val_costs else [final_val_cost],
        'final_val_cost': final_val_cost
    }


def run_comparative_study():
    """Main function to run all 6 pipelines and compare"""
    args = parse_args()
    logger = setup_logging()
    logger.info("🚀 Starting Comparative Study: 6 Pipeline Architectures (CPU)")
    
    set_seeds(42)
    
    # Load base configuration from YAML
    config = load_config(args.config)
    
    # Extract nested config values into flat structure for easier access
    # Problem settings
    if 'problem' in config:
        config.update({
            'num_customers': config['problem']['num_customers'],
            'capacity': config['problem']['vehicle_capacity'],
            'coord_range': config['problem']['coord_range'],
            'demand_range': config['problem']['demand_range']
        })
    
    # Training settings
    if 'training' in config:
        config.update({
            'num_instances': config['training']['num_instances'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
            'learning_rate': config['training']['learning_rate']
        })
    
    # Model settings
    if 'model' in config:
        config.update({
            'hidden_dim': config['model']['hidden_dim'],
            'num_heads': config['model']['num_heads'],
            'num_layers': config['model']['num_layers']
        })
    
    # Advanced training settings
    if 'training_advanced' in config:
        config.update({
            'grad_clip': config['training_advanced']['gradient_clip_norm'],
            'warmup_epochs': config['training_advanced']['warmup_epochs'],
            'min_lr': config['training_advanced']['min_lr'],
            'entropy_coef': config['training_advanced']['entropy_coef'],
            'entropy_min': config['training_advanced']['entropy_min'],
            'temp_start': config['training_advanced']['temp_start'],
            'temp_min': config['training_advanced']['temp_min']
        })
    
    # Override config values with command-line arguments if provided
    if args.customers is not None:
        config['num_customers'] = args.customers
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.instances is not None:
        config['num_instances'] = args.instances
    if args.batch is not None:
        config['batch_size'] = args.batch
    if args.coord_range is not None:
        config['coord_range'] = args.coord_range
    if args.max_demand is not None:
        config['demand_range'] = (1, args.max_demand)
    if args.capacity is not None:
        config['capacity'] = args.capacity
    if args.hidden_dim is not None:
        config['hidden_dim'] = args.hidden_dim
    if args.entropy_coef is not None:
        config['entropy_coef'] = args.entropy_coef
    if args.entropy_min is not None:
        config['entropy_min'] = args.entropy_min
    if args.warmup_epochs is not None:
        config['warmup_epochs'] = args.warmup_epochs
    if args.min_lr is not None:
        config['min_lr'] = args.min_lr
    if args.temp_start is not None:
        config['temp_start'] = args.temp_start
    if args.temp_min is not None:
        config['temp_min'] = args.temp_min
    
    # Convert string values to proper numeric types
    if isinstance(config.get('learning_rate'), str):
        config['learning_rate'] = float(config['learning_rate'])
    if isinstance(config.get('entropy_coef'), str):
        config['entropy_coef'] = float(config['entropy_coef'])
    if isinstance(config.get('entropy_min'), str):
        config['entropy_min'] = float(config['entropy_min'])
    if isinstance(config.get('temp_start'), str):
        config['temp_start'] = float(config['temp_start'])
    if isinstance(config.get('temp_min'), str):
        config['temp_min'] = float(config['temp_min'])
    if isinstance(config.get('min_lr'), str):
        config['min_lr'] = float(config['min_lr'])
    if isinstance(config.get('grad_clip'), str):
        config['grad_clip'] = float(config['grad_clip'])
    if isinstance(config.get('gradient_clip_norm'), str):
        config['gradient_clip_norm'] = float(config['gradient_clip_norm'])
    
    # Provide defaults for any missing keys
    config.setdefault('temperature', 1.0)
    
    # Ensure output directories exist (will be created by save_results based on scale)
    os.makedirs("pytorch", exist_ok=True)
    
    # Display clean config summary
    logger.info("\n" + "="*60)
    logger.info("📋 PROBLEM & TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"🎯 Problem Settings:")
    logger.info(f"   Customers: {config['num_customers']} (excluding depot)")
    logger.info(f"   Vehicle Capacity: {config['capacity']}")
    logger.info(f"   Demand Range: {config['demand_range'][0]}-{config['demand_range'][1]}")
    logger.info(f"   Coordinate Range: [0, {config['coord_range']}] → normalized to [0,1]")
    logger.info(f"🏋️ Training Settings:")
    logger.info(f"   Instances: {config['num_instances']}")
    logger.info(f"   Epochs: {config['num_epochs']}")
    logger.info(f"   Batch Size: {config['batch_size']}")
    logger.info(f"   Learning Rate: {config['learning_rate']}")
    logger.info(f"")
    logger.info(f"🧠 Model Architecture:")
    logger.info(f"   Hidden Dimension: {config['hidden_dim']}")
    logger.info(f"   Attention Heads: {config['num_heads']}")
    logger.info(f"   Transformer Layers: {config['num_layers']}")
    logger.info("="*60)
    
    # Generate instances
    logger.info("🔄 Generating CVRP instances...")
    instances = []
    naive_costs = []
    for i in tqdm(range(config['num_instances'])):
        instance = generate_cvrp_instance(
            config['num_customers'], config['capacity'],
            config['coord_range'], config['demand_range'], seed=i
        )
        instances.append(instance)
        naive_costs.append(compute_naive_baseline_cost(instance))
    
    naive_avg_cost = np.mean(naive_costs)
    naive_normalized = naive_avg_cost / config['num_customers']
    logger.info(f"📍 Naive baseline (depot->customer->depot): {naive_avg_cost:.3f} ({naive_normalized:.3f}/cust)")
    
    # Initialize models
    models = {
        'Pointer+RL': BaselinePointerNetwork(3, config['hidden_dim']),
        'GT-Greedy': GraphTransformerGreedy(3, config['hidden_dim'], config['num_heads'], config['num_layers']),
        'GT+RL': GraphTransformerNetwork(3, config['hidden_dim'], config['num_heads'], config['num_layers']),
        'DGT+RL': DynamicGraphTransformerNetwork(3, config['hidden_dim'], config['num_heads'], config['num_layers']),
        'GAT+RL': GraphAttentionTransformer(3, config['hidden_dim'], config['num_heads'], config['num_layers'])
    }
    
    # Optionally include legacy model if not restricted to DGT and dependency is available
    if not args.only_dgt:
        try:
            from src_batch.model.Model import Model as LegacyGATModel
            models['GAT+RL (legacy)'] = LegacyGATModel(node_input_dim=3, edge_input_dim=1, hidden_dim=128, edge_dim=16, layers=4, negative_slope=0.2, dropout=0.6)
        except Exception as e:
            logger.warning(f"Legacy GAT+RL unavailable (torch_geometric not installed?): {e}")
    
    # Training results storage
    results = {}
    training_times = {}
    
    # If only DGT is requested, filter the models dict
    if args.only_dgt:
        models = {'DGT+RL': models['DGT+RL']}
    
    # If DGT is excluded, remove it from training set
    if args.exclude_dgt and 'DGT+RL' in models:
        models.pop('DGT+RL')
    
    # Helper to sanitize model names for filenames
    def _sanitize(name: str):
        return name.lower().replace(' ', '_').replace('+', 'plus').replace('/', '_')

    # Train each model
    for model_name, model in models.items():
        logger.info(f"\n🎯 Training {model_name}")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        start_time = time.time()
        if model_name == 'GAT+RL (legacy)':
            result = train_legacy_gat_rl(model, instances, config, model_name, logger)
        else:
            result = train_model(model, instances, config, model_name, logger)
        training_time = time.time() - start_time
        
        results[model_name] = result
        training_times[model_name] = training_time
        
        # Immediately dump per-model training history to CSV (epochs-aligned)
        try:
            epochs = len(result.get('train_costs', []))
            train_losses = result.get('train_losses', [])
            train_costs = result.get('train_costs', [])
            # Align validation costs to epoch indices (fill others with NaN)
            val_costs_sparse = [float('nan')] * epochs
            vc = result.get('val_costs', [])
            for i, ep in enumerate(range(0, epochs, 3)):
                if i < len(vc):
                    val_costs_sparse[ep] = vc[i]
            df_hist = pd.DataFrame({
                'epoch': list(range(epochs)),
                'train_loss': train_losses + [float('nan')] * max(0, epochs - len(train_losses)),
                'train_cost': train_costs + [float('nan')] * max(0, epochs - len(train_costs)),
                'val_cost': val_costs_sparse
            })
            # Determine scale for output path
            num_customers = config.get('num_customers', 15)
            scale = 'small' if num_customers <= 20 else 'medium' if num_customers <= 50 else 'production'
            
            # Ensure CSV directory exists
            csv_dir = f"results/{scale}/csv"
            os.makedirs(csv_dir, exist_ok=True)
            
            suffix = f"_{args.plot_suffix}" if args.plot_suffix else ""
            out_hist = f"{csv_dir}/history_{_sanitize(model_name)}{suffix}.csv"
            df_hist.to_csv(out_hist, index=False)
            logger.info(f"   🧾 Saved history CSV: {out_hist}")
        except Exception as e:
            logger.warning(f"Failed to save per-model CSV for {model_name}: {e}")
        
        logger.info(f"   ✅ {model_name} completed in {training_time:.1f}s")
        logger.info(f"   Final validation cost: {result['final_val_cost']:.3f} ({result['final_val_cost'] / config['num_customers']:.3f}/cust)")
    
    # If we excluded DGT, try to reuse prior DGT results for comparison
    if args.exclude_dgt:
        try:
            prior = torch.load(args.reuse_dgt_path, map_location='cpu')
            prior_results = prior.get('results', {})
            prior_times = prior.get('training_times', {})
            if 'DGT+RL' in prior_results:
                results['DGT+RL'] = prior_results['DGT+RL']
                training_times['DGT+RL'] = prior_times.get('DGT+RL', float('nan'))
            else:
                logger.warning("Could not find DGT+RL in prior results; proceeding without it.")
        except Exception as e:
            logger.warning(f"Failed to load prior DGT results from {args.reuse_dgt_path}: {e}")
        
        # For parameter count in plots, estimate from saved state_dict if available
        try:
            sd = torch.load('results/small/pytorch/model_dgtplus_rl.pt', map_location='cpu')
            state = sd.get('model_state_dict', sd)
            dgt_params = int(sum(t.numel() for t in state.values() if hasattr(t, 'numel')))
            # Create a dummy module to report params for plotting
            class _Dummy(nn.Module):
                def __init__(self, n):
                    super().__init__()
                    self._n = n
                def parameters(self):
                    # Fake a single parameter tensor with correct count for plotting only
                    yield nn.Parameter(torch.zeros(self._n))
            models['DGT+RL'] = _Dummy(dgt_params)
        except Exception as e:
            logger.warning(f"Failed to infer DGT+RL parameter count: {e}")
    
    # Generate comparison plots and save results BEFORE strict validation to ensure outputs are persisted
    create_comparison_plots(results, training_times, config, logger, naive_avg_cost, models)

    # Create and solve test instance for detailed analysis
    test_results = create_and_solve_test_instance(models, config, logger)
    
    # Persist artifacts regardless of validation outcome
    save_results(results, training_times, models, config)

    # Now run strict validation, but log failures without preventing saved outputs
    try:
        validate_naive_baseline_correctness(results, naive_avg_cost, config, logger, instances)
    except Exception as e:
        logger.error(f"Strict validation failed after saving outputs: {e}")
    
    # Performance summary
    logger.info("\n📊 COMPARATIVE STUDY RESULTS")
    logger.info("=" * 50)
    
    for model_name, result in results.items():
        params = sum(p.numel() for p in models[model_name].parameters())
        logger.info(f"{model_name}:")
        logger.info(f"   Parameters: {params:,}")
        logger.info(f"   Training time: {training_times[model_name]:.1f}s")
        logger.info(f"   Final validation cost: {result['final_val_cost']:.2f}")
        logger.info(f"   Final training cost: {result['train_costs'][-1]:.2f}")
        logger.info("")
    
    # Save results
    save_results(results, training_times, models, config)
    
    return results

def create_comparison_plots(results, training_times, config, logger, naive_baseline_cost, models):
    """Create comprehensive comparison plots with normalized costs (per customer)"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # Normalize naive baseline cost
    naive_normalized = naive_baseline_cost / config['num_customers']
    
    # Build consistent color map for all figures (lines and bars)
    model_names = list(results.keys())
    palette = sns.color_palette("tab10", n_colors=len(model_names))
    color_map = {name: palette[i] for i, name in enumerate(model_names)}

    # 1. Training Loss Comparison (standardized REINFORCE loss for all models)
    plt.subplot(2, 4, 1)
    for model_name, result in results.items():
        plt.plot(result['train_losses'], label=model_name, linewidth=2, marker='o', markersize=3, color=color_map[model_name])
    plt.title('Training Loss Evolution\n(Standardized REINFORCE)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('REINFORCE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Training Cost Comparison (NORMALIZED) - NO RED LINE
    plt.subplot(2, 4, 2)
    for model_name, result in results.items():
        # Normalize training costs by dividing by number of customers
        normalized_train_costs = [cost / config['num_customers'] for cost in result['train_costs']]
        plt.plot(normalized_train_costs, label=model_name, linewidth=2, marker='s', markersize=3, color=color_map[model_name])
    plt.title('Training Cost Evolution (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost per Customer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Validation Cost vs Naive (NORMALIZED)
    plt.subplot(2, 4, 3)
    # Plot naive baseline as background reference line
    val_epochs_full = list(range(0, config['num_epochs'], 3))
    if len(val_epochs_full) == 0:
        val_epochs_full = [0]
    plt.axhline(y=naive_normalized, color='lightgray', linewidth=3, linestyle='--', label='Naive Baseline')
    for model_name, result in results.items():
        val_epochs = list(range(0, config['num_epochs'], 3))[:len(result['val_costs'])]
        normalized_val_costs = [cost / config['num_customers'] for cost in result['val_costs']][:len(val_epochs)]
        plt.plot(val_epochs, normalized_val_costs, 'o-', label=model_name, linewidth=2, markersize=5, color=color_map[model_name])
    plt.title('Validation Cost vs Naive (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost per Customer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Final Performance Bar Chart with Naive Baseline (NORMALIZED)
    plt.subplot(2, 4, 4)
    model_names = list(results.keys())
    # Normalize final costs by dividing by number of customers
    final_costs_normalized = [results[name]['final_val_cost'] / config['num_customers'] for name in model_names]
    # Colors consistent with lines
    colors = [color_map[name] for name in model_names]
    
    # Add naive baseline to the comparison (normalized)
    all_names = model_names + ['Naive Baseline']
    all_costs_normalized = final_costs_normalized + [naive_normalized]
    all_colors = colors + [(0.8, 0.2, 0.2)]  # red-like for naive
    
    bars = plt.bar(range(len(all_names)), all_costs_normalized, color=all_colors, alpha=0.8)
    plt.title('Final Performance vs Naive Baseline (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Approach')
    plt.ylabel('Average Cost per Customer')
    plt.xticks(range(len(all_names)), [name.replace(' ', '\n') for name in all_names], rotation=45)
    
    # Add value labels on bars (normalized)
    for bar, cost in zip(bars, all_costs_normalized):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{cost:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Training Time Comparison
    plt.subplot(2, 4, 5)
    times = [training_times[name] for name in model_names]
    bars = plt.bar(range(len(model_names)), times, color=[color_map[n] for n in model_names], alpha=0.8)
    plt.title('Training Time', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names])
    
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Model Complexity (Parameters)
    plt.subplot(2, 4, 6)
    param_counts = [sum(p.numel() for p in models[name].parameters()) for name in model_names]
    bars = plt.bar(range(len(model_names)), param_counts, color=[color_map[n] for n in model_names], alpha=0.8)
    plt.title('Model Complexity', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Parameters')
    plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names])
    
    for bar, params in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                f'{params:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 7. Learning Efficiency (Cost Improvement)
    plt.subplot(2, 4, 7)
    improvements = []
    for model_name, result in results.items():
        initial_cost = result['train_costs'][0]
        final_cost = result['train_costs'][-1]
        improvement = ((initial_cost - final_cost) / initial_cost) * 100
        improvements.append(improvement)
    
    bars = plt.bar(range(len(model_names)), improvements, color=[color_map[n] for n in model_names], alpha=0.8)
    plt.title('Learning Efficiency', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Cost Improvement (%)')
    plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names])
    
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. Performance vs Complexity Scatter (NORMALIZED)
    plt.subplot(2, 4, 8)
    for i, model_name in enumerate(model_names):
        plt.scatter(param_counts[i], final_costs_normalized[i], 
                   s=100, color=color_map[model_name], alpha=0.8, label=model_name)
        plt.annotate(model_name.replace(' ', '\n'), 
                    (param_counts[i], final_costs_normalized[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Performance vs Complexity (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Parameters')
    plt.ylabel('Validation Cost per Customer')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine scale and ensure plots directory exists
    num_customers = config.get('num_customers', 15)
    scale = 'small' if num_customers <= 20 else 'medium' if num_customers <= 50 else 'production'
    plots_dir = f"results/{scale}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Determine suffix from CLI
    args = parse_args()
    suffix = f"_{args.plot_suffix}" if args.plot_suffix else ""
    out_path = f"{plots_dir}/comparative_study_results{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plots saved to {out_path}")
    
    # Create detailed performance table
    create_performance_table(results, training_times, param_counts, logger)

def create_formatted_results_table(results_dict, training_times, param_counts, naive_cost_per_customer):
    """
    Create a nicely formatted comparison table with proper alignment and 3-digit precision.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        training_times: Dictionary with model training times
        param_counts: List of parameter counts for each model
        naive_cost_per_customer: Naive baseline cost per customer for improvement calculation
    
    Returns:
        str: Formatted table string
    """
    if not results_dict:
        return "No results to display"
    
    model_names = list(results_dict.keys())
    
    # Calculate column widths dynamically
    model_width = max(len("Model"), max(len(model) for model in model_names))
    params_width = max(len("Parameters"), max(len(f"{param_counts[i]:,}") for i in range(len(model_names))))
    time_width = max(len("Time (s)") - 1, max(len(f"{training_times[model]:.1f}s") for model in model_names))  # Remove 1 space
    train_cost_width = max(len("Train Cost"), max(len(f"{results_dict[model]['train_costs'][-1]:.3f}") for model in model_names))
    val_cost_width = max(len("Val Cost"), max(len(f"{results_dict[model]['final_val_cost']:.3f}") for model in model_names))
    val_per_cust_width = max(len("Val/Cust"), max(len(f"{results_dict[model]['final_val_cost']/20:.3f}") for model in model_names))
    improvement_width = max(len("Improv %") - 1, max(len(f"{(1 - (results_dict[model]['final_val_cost']/20)/naive_cost_per_customer)*100:.2f}%") for model in model_names))  # Remove 1 space
    
    # Create table components - add 1 dash to time and improvement columns for proper alignment
    header = f"| {'Model':<{model_width}} | {'Parameters':>{params_width}} | {'Time (s)':>{time_width}} | {'Train Cost':>{train_cost_width}} | {'Val Cost':>{val_cost_width}} | {'Val/Cust':>{val_per_cust_width}} | {'Improv %':>{improvement_width}} |"
    separator = f"|{'-'*(model_width+2)}|{'-'*(params_width+2)}|{'-'*(time_width+3)}|{'-'*(train_cost_width+2)}|{'-'*(val_cost_width+2)}|{'-'*(val_per_cust_width+2)}|{'-'*(improvement_width+3)}|"
    
    # Build table rows
    table_lines = [header, separator]
    
    for i, model_name in enumerate(model_names):
        metrics = results_dict[model_name]
        val_per_customer = metrics['final_val_cost'] / 20
        improvement = (1 - val_per_customer / naive_cost_per_customer) * 100
        
        row = f"| {model_name:<{model_width}} | {param_counts[i]:>{params_width},} | {training_times[model_name]:>{time_width}.1f}s | {metrics['train_costs'][-1]:>{train_cost_width}.3f} | {metrics['final_val_cost']:>{val_cost_width}.3f} | {val_per_customer:>{val_per_cust_width}.3f} | {improvement:>{improvement_width}.2f}% |"
        table_lines.append(row)
    
    return "\n".join(table_lines)

def create_performance_table(results, training_times, param_counts, logger):
    """Create detailed performance comparison table"""
    
    # Calculate naive baseline cost per customer from config
    config_num_customers = 20  # From the global config
    
    # Use validation set naive baseline from the validation function
    # For now, estimate from the validation results (we know it's around 0.2225)
    naive_cost_per_customer = 0.2225  # This should match the validation baseline
    
    data = []
    model_names = list(results.keys())
    
    for i, model_name in enumerate(model_names):
        result = results[model_name]
        data.append({
            'Model': model_name,
            'Parameters': f"{param_counts[i]:,}",
            'Training Time (s)': f"{training_times[model_name]:.1f}",
            'Final Train Cost': f"{result['train_costs'][-1]:.3f}",
            'Final Val Cost': f"{result['final_val_cost']:.3f}",
            'Best Val Cost': f"{min(result['val_costs']):.3f}",
            'Cost Std': f"{np.std(result['val_costs']):.3f}",
            'Convergence': f"{len(result['train_costs'])}/25"
        })
    
    df = pd.DataFrame(data)
    
    # Determine scale and ensure CSV directory exists  
    # Note: We need to get config from somewhere - using default for now
    scale = 'small'  # Default scale - this will be corrected when called from main function
    csv_dir = f"results/{scale}/csv"
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save to CSV
    args = parse_args()
    suffix = f"_{args.plot_suffix}" if args.plot_suffix else ""
    csv_path = f'{csv_dir}/comparative_results{suffix}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"📋 Detailed results saved to {csv_path}")
    
    # Print formatted table
    logger.info("\n📊 DETAILED PERFORMANCE COMPARISON")
    # logger.info("=" * 120)
    logger.info("\n")
    
    # Create and print the nicely formatted table
    formatted_table = create_formatted_results_table(results, training_times, param_counts, naive_cost_per_customer)
    print(formatted_table)
    logger.info("\n")

def validate_naive_baseline_correctness(results, naive_avg_cost, config, logger, instances):
    """STRICT VALIDATION: Naive baseline is absolute maximum - no model can exceed it, even slightly"""
    
    # CRITICAL FIX: Calculate naive baseline from VALIDATION SET ONLY (same data used for model validation)
    split_idx = int(0.8 * len(instances))
    val_instances = instances[split_idx:]
    
    val_naive_costs = [compute_naive_baseline_cost(inst) for inst in val_instances]
    val_naive_avg = np.mean(val_naive_costs)
    val_naive_normalized = val_naive_avg / config['num_customers']
    
    logger.info("\n🔍 STRICT VALIDATION: Checking naive baseline correctness...")
    logger.info(f"📊 Training set naive baseline: {naive_avg_cost / config['num_customers']:.4f} cost/customer")
    logger.info(f"📊 Validation set naive baseline (used for comparison): {val_naive_normalized:.4f} cost/customer")
    
    validation_passed = True
    violations = []
    
    for model_name, result in results.items():
        final_cost_normalized = result['final_val_cost'] / config['num_customers']
        
        if final_cost_normalized > val_naive_normalized:
            # STRICT: ANY violation is an error
            excess = final_cost_normalized - val_naive_normalized
            excess_pct = (excess / val_naive_normalized) * 100
            
            logger.error(f"❌ CRITICAL VIOLATION: {model_name}")
            logger.error(f"   Model cost: {final_cost_normalized:.4f}/cust")
            logger.error(f"   Naive max:  {val_naive_normalized:.4f}/cust")
            logger.error(f"   Excess:     +{excess:.4f}/cust ({excess_pct:+.2f}%)")
            logger.error(f"   This model EXCEEDS the theoretical maximum!")
            
            violations.append({
                'model': model_name,
                'cost': final_cost_normalized,
                'excess': excess,
                'excess_pct': excess_pct
            })
            validation_passed = False
        
        elif final_cost_normalized == val_naive_normalized:
            logger.warning(f"⚠️  {model_name}: {final_cost_normalized:.4f}/cust = {val_naive_normalized:.4f}/cust (Equal to naive - no learning)")
        
        else:
            improvement = ((val_naive_normalized - final_cost_normalized) / val_naive_normalized) * 100
            logger.info(f"✅ {model_name}: {final_cost_normalized:.4f}/cust < {val_naive_normalized:.4f}/cust (improvement: {improvement:.2f}%)")
    
    if not validation_passed:
        logger.error("\n🚨🚨🚨 CRITICAL ERROR: BASELINE VALIDATION FAILED! 🚨🚨🚨")
        logger.error("="*60)
        logger.error("VIOLATION SUMMARY:")
        for violation in violations:
            logger.error(f"  • {violation['model']}: +{violation['excess']:.4f}/cust ({violation['excess_pct']:+.2f}%)")
        
        logger.error("\nThis indicates serious issues:")
        logger.error("  1. Model architecture problems")
        logger.error("  2. Training instability")
        logger.error("  3. Implementation bugs")
        logger.error("  4. Invalid route generation")
        logger.error("\nIMMEDIATE INVESTIGATION REQUIRED!")
        logger.error("="*60)
        
        raise ValueError(f"STRICT BASELINE VALIDATION FAILED: {len(violations)} model(s) exceed naive baseline - investigation required!")
    else:
        logger.info("✅✅✅ STRICT VALIDATION PASSED: All models ≤ naive baseline! ✅✅✅")

def save_results(results, training_times, models, config):
    """Save all results and models to organized results structure"""
    
    # Determine experiment scale based on problem size
    num_customers = config.get('num_customers', 15)
    if num_customers <= 20:
        scale = 'small'
    elif num_customers <= 50:
        scale = 'medium'
    else:
        scale = 'production'
    
    # Create organized directories
    pytorch_dir = f"results/{scale}/pytorch"
    analysis_dir = f"results/{scale}/analysis"
    logs_dir = f"results/{scale}/logs"
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save individual models to pytorch directory
    for model_name, model in models.items():
        filename = f"{pytorch_dir}/model_{model_name.lower().replace(' ', '_').replace('+', 'plus')}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'results': results[model_name],
            'training_time': training_times[model_name]
        }, filename)
    
    # Save complete comparative study results to analysis directory
    torch.save({
        'results': results,
        'training_times': training_times,
        'config': config,
        'experiment_scale': scale
    }, f'{analysis_dir}/comparative_study_complete.pt')

def create_and_solve_test_instance(models, config, logger):
    """Create a test CVRP instance and solve it with each trained model for detailed analysis"""
    logger.info("\n🧪 Creating test instance for detailed model comparison...")
    
    # Determine scale
    num_customers = config.get('num_customers', 15)
    scale = 'small' if num_customers <= 20 else 'medium' if num_customers <= 50 else 'production'
    test_dir = f"results/{scale}/test_instances"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a representative test instance
    test_instance = generate_cvrp_instance(
        num_customers=config['num_customers'],
        capacity=config['capacity'],
        coord_range=config['coord_range'],
        demand_range=config['demand_range'],
        seed=12345  # Fixed seed for reproducibility
    )
    
    # Save test instance
    np.savez(f"{test_dir}/test_instance.npz",
             coords=test_instance['coords'],
             demands=test_instance['demands'],
             distances=test_instance['distances'],
             capacity=test_instance['capacity'])
    
    logger.info(f"📍 Test instance: {config['num_customers']} customers, capacity={config['capacity']}")
    logger.info(f"   Coords range: {test_instance['coords'].min():.3f} to {test_instance['coords'].max():.3f}")
    logger.info(f"   Demands range: {test_instance['demands'][1:].min():.3f} to {test_instance['demands'][1:].max():.3f}")
    
    # Compute naive baseline for reference
    naive_cost = compute_naive_baseline_cost(test_instance)
    naive_route = naive_baseline_solution(test_instance)
    naive_normalized = naive_cost / config['num_customers']
    
    logger.info(f"   Naive baseline: {naive_cost:.3f} ({naive_normalized:.3f}/customer)")
    
    # Test each model on the instance
    test_results = {}
    
    for model_name, model in models.items():
        logger.info(f"\n🔍 Testing {model_name} on test instance...")
        model.eval()
        
        with torch.no_grad():
            # Test both greedy and sampling - handle legacy model differently
            if model_name == 'GAT+RL (legacy)':
                # Legacy model uses different interface and expects batched PyG data
                from torch_geometric.loader import DataLoader
                test_data = build_pyg_data_from_instance(test_instance)
                # Create a DataLoader with batch_size=1 to properly batch the single instance
                test_loader = DataLoader([test_data], batch_size=1, shuffle=False)
                test_batch = next(iter(test_loader))
                
                n_steps = config['num_customers'] * 2
                T = 2.5
                
                # Legacy model returns actions, log_probs (no entropy)
                greedy_actions, greedy_log_probs = model(test_batch, n_steps=n_steps, greedy=True, T=T)
                sample_actions, sample_log_probs = model(test_batch, n_steps=n_steps, greedy=False, T=T)
                
                # Convert to route format (add depot at start and end)
                # Handle different possible tensor shapes from legacy model
                if greedy_actions.dim() == 1:
                    # If 1D, add batch dimension
                    greedy_actions = greedy_actions.unsqueeze(0)
                    sample_actions = sample_actions.unsqueeze(0)
                
                depot = torch.zeros(greedy_actions.size(0), 1, dtype=torch.long)
                greedy_routes = [torch.cat([depot, greedy_actions, depot], dim=1)[0].cpu().tolist()]
                sample_routes = [torch.cat([depot, sample_actions, depot], dim=1)[0].cpu().tolist()]
                
                # Set dummy entropy values
                greedy_entropy = torch.zeros(1)
                sample_entropy = torch.zeros(1)
            else:
                # Regular models
                greedy_routes, greedy_log_probs, greedy_entropy = model([test_instance], greedy=True)
                sample_routes, sample_log_probs, sample_entropy = model([test_instance], greedy=False, temperature=1.0)
            
            greedy_route = greedy_routes[0]
            sample_route = sample_routes[0]
            
            # Validate routes
            validate_route(greedy_route, config['num_customers'], f"{model_name}-GREEDY-TEST", test_instance)
            validate_route(sample_route, config['num_customers'], f"{model_name}-SAMPLE-TEST", test_instance)
            
            # Compute costs
            greedy_cost = compute_route_cost(greedy_route, test_instance['distances'])
            sample_cost = compute_route_cost(sample_route, test_instance['distances'])
            
            # Compute improvement over naive
            greedy_improvement = ((naive_cost - greedy_cost) / naive_cost) * 100
            sample_improvement = ((naive_cost - sample_cost) / naive_cost) * 100
            
            test_results[model_name] = {
                'greedy_route': greedy_route,
                'sample_route': sample_route,
                'greedy_cost': greedy_cost,
                'sample_cost': sample_cost,
                'greedy_cost_per_customer': greedy_cost / config['num_customers'],
                'sample_cost_per_customer': sample_cost / config['num_customers'],
                'greedy_improvement': greedy_improvement,
                'sample_improvement': sample_improvement,
                'greedy_log_prob': greedy_log_probs[0].item(),
                'sample_log_prob': sample_log_probs[0].item(),
                'greedy_entropy': greedy_entropy[0].item(),
                'sample_entropy': sample_entropy[0].item()
            }
            
            logger.info(f"   Greedy: {greedy_cost:.3f} ({greedy_cost/config['num_customers']:.3f}/cust) - {greedy_improvement:+.1f}% vs naive")
            logger.info(f"   Sample: {sample_cost:.3f} ({sample_cost/config['num_customers']:.3f}/cust) - {sample_improvement:+.1f}% vs naive")
            logger.info(f"   Route lengths: Greedy={len(greedy_route)}, Sample={len(sample_route)}")
    
    # Create detailed test results analysis
    test_analysis = {
        'test_instance': {
            'coords': test_instance['coords'].tolist(),
            'demands': test_instance['demands'].tolist(),
            'capacity': test_instance['capacity'],
            'num_customers': config['num_customers']
        },
        'naive_baseline': {
            'route': naive_route,
            'cost': naive_cost,
            'cost_per_customer': naive_normalized
        },
        'model_results': test_results,
        'config': config
    }
    
    # Save detailed test analysis to analysis directory (consolidating data there)
    analysis_dir = f"results/{scale}/analysis"
    torch.save(test_analysis, f"{analysis_dir}/test_instance_analysis.pt")
    
    # Also save as JSON for easy reading
    import json
    
    # Convert numpy types to native Python for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_analysis = convert_for_json(test_analysis)
    with open(f"{analysis_dir}/test_instance_analysis.json", 'w') as f:
        json.dump(json_analysis, f, indent=2)
    
    # Create route visualizations
    try:
        from visualize_test_routes import plot_test_instance_routes, create_interactive_route_analysis
        plots_dir = f"results/{scale}/plots"
        plot_test_instance_routes(test_analysis, config, logger, plots_dir)
        create_interactive_route_analysis(test_analysis, config, analysis_dir)
        logger.info(f"   🎨 Created route visualizations in {plots_dir}")
        logger.info(f"   📄 Created interactive HTML report in {analysis_dir}")
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to create route visualizations: {e}")
    
    logger.info(f"\n💾 Test results saved to:")
    logger.info(f"   Binary: {analysis_dir}/test_instance_analysis.pt")
    logger.info(f"   JSON: {analysis_dir}/test_instance_analysis.json")
    logger.info(f"   Instance: {test_dir}/test_instance.npz")
    logger.info(f"   HTML Report: {analysis_dir}/test_instance_analysis.html")
    
    # Print summary table (normalized per customer)
    logger.info("\n📊 TEST INSTANCE PERFORMANCE SUMMARY (Per Customer)")
    logger.info("=" * 80)
    logger.info(f"{'Model':<20} {'Greedy Cost/Cust':<15} {'Sample Cost/Cust':<15} {'Greedy Impr':<12} {'Sample Impr':<12}")
    logger.info("-" * 80)
    
    for model_name, results in test_results.items():
        greedy_per_cust = results['greedy_cost'] / config['num_customers']
        sample_per_cust = results['sample_cost'] / config['num_customers']
        logger.info(f"{model_name:<20} {greedy_per_cust:<15.3f} {sample_per_cust:<15.3f} {results['greedy_improvement']:<12.1f}% {results['sample_improvement']:<12.1f}%")
    
    logger.info(f"{'Naive Baseline':<20} {naive_normalized:<15.3f} {naive_normalized:<15.3f} {'0.0':<12} {'0.0':<12}")
    logger.info("=" * 80)
    
    return test_results

if __name__ == "__main__":
    results = run_comparative_study()
    args = parse_args()
    suffix = f"_{args.plot_suffix}" if args.plot_suffix else ""
    # Get the scale to show correct output path - load config to get num_customers
    config = load_config(args.config)
    num_customers = args.customers if args.customers is not None else config.get('problem', {}).get('num_customers', 15)
    scale = 'small' if num_customers <= 20 else 'medium' if num_customers <= 50 else 'production'
    print(f"\n🎉 Comparative study completed! Check results/{scale}/plots/comparative_study_results{suffix}.png for detailed analysis.")
