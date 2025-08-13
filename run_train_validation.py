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

def get_device_from_config(config):
    """CPU-only device selection (GPU disabled)."""
    # Hard-disable CUDA visibility to avoid accidental GPU usage
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    device = torch.device('cpu')
    print("🖥️  Using CPU (CPU-optimized)")
    return device

def setup_logging(config=None):
    """Setup logging with configuration"""
    if config and 'logging' in config:
        level_str = config['logging'].get('level', 'INFO')
        format_str = config['logging'].get('format', '%(message)s')
        level = getattr(logging, level_str.upper(), logging.INFO)
    else:
        level = logging.INFO
        format_str = '%(message)s'
    
    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger(__name__)

def set_seeds(config=None, seed=None):
    """Set random seeds from config or explicit value"""
    if seed is None:
        seed = config.get('experiment', {}).get('random_seed', 42) if config else 42
    torch.manual_seed(seed)
    np.random.seed(seed)


def configure_cpu_threads(config=None):
    """Enhanced CPU threading configuration from config parameters."""
    import os
    try:
        # Get CPU optimization settings from config
        cpu_config = config.get('system', {}).get('cpu_optimization', {}) if config else {}
        openmp_config = config.get('system', {}).get('openmp_settings', {}) if config else {}
        
        # Determine max threads
        if cpu_config.get('auto_detect_threads', True):
            max_threads = cpu_config.get('max_threads') or os.cpu_count() or 4
        else:
            max_threads = cpu_config.get('max_threads', 4)
        
        # Set PyTorch threading
        torch.set_num_threads(max_threads)
        
        # Set inter-op threads with configurable divisor
        divisor = cpu_config.get('inter_op_threads_divisor', 4)
        torch.set_num_interop_threads(max(1, max_threads // divisor))
        
        # Configure OpenMP if available (use config values or defaults)
        os.environ['OMP_NUM_THREADS'] = str(max_threads)
        os.environ['MKL_NUM_THREADS'] = str(max_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(max_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(max_threads)
        
        # Apply OpenMP settings from config
        os.environ['KMP_BLOCKTIME'] = openmp_config.get('kmp_blocktime', '0')
        os.environ['KMP_SETTINGS'] = openmp_config.get('kmp_settings', '0')
        os.environ['KMP_AFFINITY'] = openmp_config.get('kmp_affinity', 'granularity=fine,compact,1,0')
        
        print(f"🚀 CPU optimization: {max_threads} threads configured")
        print(f"   PyTorch threads: {torch.get_num_threads()}")
        print(f"   Inter-op threads: {torch.get_num_interop_threads()}")
        
    except Exception as e:
        print(f"⚠️ CPU threading configuration failed: {e}")

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
    
    def __init__(self, input_dim, hidden_dim, config=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Simple node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Basic attention mechanism
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Pointer network with configurable multiplier
        input_multiplier = config['model']['pointer_network']['input_multiplier']
        
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * input_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps, temperature, greedy, config):
        batch_size = len(instances)
        
        # Calculate max steps from config
        max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        
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
        
        attention_scaling = config['inference']['attention_temperature_scaling']
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** attention_scaling)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.bmm(attention_weights, V)
        
        return self._generate_routes(attended, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances, config)
    
    def _generate_routes(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances, config):
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
            
            # Use config-based masked score value instead of hardcoded -1e9
            masked_score_value = config['inference']['masked_score_value']
            scores = scores.masked_fill(mask, masked_score_value)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_prob_epsilon = config['inference']['log_prob_epsilon']
            log_probs = torch.log(probs + log_prob_epsilon)

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
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config):
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
                dim_feedforward=hidden_dim * feedforward_multiplier,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Graph-level aggregation
        self.graph_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Pointer network with configurable context multiplier
        context_multiplier = config['model']['pointer_network']['context_multiplier']
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * context_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=None, greedy=False, config=None):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        if temperature is None:
            temperature = config['inference']['default_temperature']
        
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
        
        return self._generate_routes(enhanced_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances, config)
    
    def _generate_routes(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances, config):
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
            
            masked_score_value = config['inference']['masked_score_value']
            scores = scores.masked_fill(mask, masked_score_value)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_prob_epsilon = config['inference']['log_prob_epsilon']
            log_probs = torch.log(probs + log_prob_epsilon)

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
    """Pipeline 3: Pure Greedy Attention Baseline (No RL training)
    Greedy routing using attention-style scores: from current node, pick the most
    important unmasked next node by dot-product attention over embeddings, until
    capacity is exhausted; then go back to depot. Repeat until all customers are visited.
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder stack for building node embeddings
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * feedforward_multiplier,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Simple attention projection heads (query/key) for greedy selection
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, instances, max_steps=None, temperature=None, greedy=True, config=None):  # Greedy baseline ignores temperature
        batch_size = len(instances)
        if batch_size == 0:
            return [], torch.tensor([]), torch.tensor([])
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        device = next(self.parameters()).device
        node_features = torch.zeros(batch_size, max_nodes, 3, device=device)
        demands_batch = torch.zeros(batch_size, max_nodes, device=device)
        capacities = torch.zeros(batch_size, device=device)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            capacities[i] = inst['capacity']
        
        # Build embeddings with transformer encoder
        x = self.node_embedding(node_features)  # [B, N, H]
        for layer in self.transformer_layers:
            x = layer(x)
        
        routes, logp, ent = self._greedy_routes(x, demands_batch, capacities, max_steps, instances)
        return routes, logp, ent
    
    def _greedy_routes(self, node_embeddings, demands_batch, capacities, max_steps, instances):
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        device = node_embeddings.device
        routes = [[] for _ in range(batch_size)]
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        current_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)  # start at depot 0
        
        # Initialize routes at depot
        for b in range(batch_size):
            routes[b].append(0)
        
        # Precompute keys for all nodes
        K_all = self.attn_k(node_embeddings)  # [B, N, H]
        scale = float(self.hidden_dim) ** 0.5
        
        for step in range(max_steps):
            # Check completion per batch: all customers visited and currently at depot
            done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords'])
                customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                at_depot = (routes[b][-1] == 0)
                if customers_visited and at_depot:
                    done_mask[b] = True
            if done_mask.all():
                break
            
            # For each active batch, compute attention scores from current node to all nodes
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            for b in range(batch_size):
                if done_mask[b]:
                    actions[b] = 0
                    continue
                actual_nodes = len(instances[b]['coords'])
                # Build mask: visited customers, capacity infeasible customers, padded nodes
                mask = torch.zeros(max_nodes, dtype=torch.bool, device=device)
                # mask visited
                mask |= visited[b]
                # mask beyond actual nodes
                if actual_nodes < max_nodes:
                    mask[actual_nodes:] = True
                # capacity infeasible (customers only; depot is always allowed unless no feasible customers exist)
                infeasible = demands_batch[b] > remaining_capacity[b]
                infeasible[0] = False  # depot never infeasible
                mask |= infeasible
                
                # Enforce: if any feasible unvisited customer exists, depot must be masked
                # Feasible unvisited customers are those not masked in indices [1:actual_nodes)
                feasible_unvisited_exists = (~mask[1:actual_nodes]).any() if actual_nodes > 1 else False
                if feasible_unvisited_exists:
                    mask[0] = True
                else:
                    mask[0] = False  # no feasible customer: allow returning to depot
                
                # If all masked (can happen due to padding/visited), ensure we can still return to depot
                currently_at_depot = (routes[b][-1] == 0)
                if mask.all() and not currently_at_depot:
                    mask[0] = False
                # If still all masked (e.g., nothing to do), pick depot
                if mask.all():
                    actions[b] = 0
                    continue
                
                curr = routes[b][-1]
                q = self.attn_q(node_embeddings[b, curr:curr+1, :])  # [1, H]
                k = K_all[b, :, :]  # [N, H]
                scores = (q @ k.t()).squeeze(0) / scale  # [N]
                scores = scores.masked_fill(mask, -1e9)
                next_node = int(torch.argmax(scores).item())
                actions[b] = next_node
            
            # Update state
            for b in range(batch_size):
                if done_mask[b]:
                    continue
                a = int(actions[b].item())
                routes[b].append(a)
                if a == 0:
                    remaining_capacity[b] = capacities[b]
                else:
                    visited[b, a] = True
                    remaining_capacity[b] -= demands_batch[b, a]
                    if remaining_capacity[b] < 0:
                        remaining_capacity[b] = torch.tensor(0.0, device=device)
                current_nodes[b] = a
        
        # Ensure non-empty
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        # No RL: return zero log-probs and entropies (no gradients)
        logp = torch.zeros(batch_size, dtype=torch.float32, device=device)
        ent = torch.zeros(batch_size, dtype=torch.float32, device=device)
        return routes, logp, ent

class DynamicGraphTransformerNetwork(nn.Module):
    """Pipeline 4: Dynamic Graph Transformer with adaptive updates"""
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config):
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
                dim_feedforward=hidden_dim * feedforward_multiplier,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Dynamic update components with config-based parameters
        dgt_config = config['model']['dynamic_graph_transformer']
        state_features = dgt_config['state_features']
        self.state_encoder = nn.Linear(state_features, hidden_dim)
        
        # PreNorm + gated residual for stability of dynamic updates
        self.pre_norm = nn.LayerNorm(hidden_dim)
        residual_gate_init = dgt_config['residual_gate_init']
        self.res_gate = nn.Parameter(torch.tensor(residual_gate_init))
        
        update_multiplier = dgt_config['update_input_multiplier']
        self.dynamic_update = nn.Sequential(
            nn.Linear(hidden_dim * update_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced pointer network with configurable dimensions
        pointer_multiplier = 3  # node + context + state
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * pointer_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=None, greedy=False, config=None):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        if temperature is None:
            temperature = config['inference']['default_temperature']
        
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
        
        return self._generate_routes_dynamic(x, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances, config)
    
    def _generate_routes_dynamic(self, node_embeddings, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances=None, config=None):
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
            
            masked_score_value = config['inference']['masked_score_value']
            scores = scores.masked_fill(mask, masked_score_value)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_prob_epsilon = config['inference']['log_prob_epsilon']
            log_probs = torch.log(probs + log_prob_epsilon)

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
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, edge_embedding_divisor, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Edge feature embedding (distance) with config-based parameters
        gat_config = config['model']['graph_attention_transformer']
        edge_input_dim = 1  # Distance scalar
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim // edge_embedding_divisor)
        
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
        
        # Enhanced pointer network with configurable multiplier
        input_multiplier = gat_config['input_multiplier']
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * input_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gat_config['output_dim'])
        )
        
    def forward(self, instances, max_steps=None, temperature=None, greedy=False, config=None):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        if temperature is None:
            temperature = config['inference']['default_temperature']
        
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
        
        return self._generate_routes_gat(enhanced_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances, config)
    
    def _generate_routes_gat(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances, config):
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
            
            masked_score_value = config['inference']['masked_score_value']
            scores = scores.masked_fill(mask, masked_score_value)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_prob_epsilon = config['inference']['log_prob_epsilon']
            log_probs = torch.log(probs + log_prob_epsilon)

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
    """Compute total Euclidean cost of a route (no penalty)"""
    if len(route) <= 1:
        return 0.0
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]]
    return cost

def count_internal_depot_visits(route):
    """Count depot visits excluding the starting and ending depot nodes"""
    if not route or len(route) < 3:
        return 0
    return sum(1 for node in route[1:-1] if node == 0)

def compute_route_cost_with_penalty(route, distances, penalty_per_visit=0.0):
    base = compute_route_cost(route, distances)
    if penalty_per_visit and penalty_per_visit != 0.0:
        base += penalty_per_visit * count_internal_depot_visits(route)
    return base

def compute_normalized_cost(route, distances, n_customers):
    """Compute cost per customer (normalized cost)"""
    total_cost = compute_route_cost(route, distances)
    return total_cost / n_customers if n_customers > 0 else 0.0

def compute_naive_baseline_cost(instance, depot_penalty_per_visit: float = 0.0):
    """Compute cost of naive solution: depot->customer->depot for each customer
    Includes optional depot penalty per internal return when configured.
    """
    distances = instance['distances']
    n_customers = len(instance['coords']) - 1  # excluding depot
    naive_cost = 0.0
    
    for customer_idx in range(1, n_customers + 1):  # customers are indexed 1 to n
        naive_cost += distances[0, customer_idx] * 2  # depot->customer->depot
    
    # Internal depot visits in naive route: n_customers - 1 (exclude first start and final end)
    if depot_penalty_per_visit and depot_penalty_per_visit != 0.0 and n_customers > 0:
        naive_cost += depot_penalty_per_visit * (n_customers - 1)
    
    return naive_cost

def quick_validate_route(route, n_customers):
    """Lightweight validation for speed: start/end depot, no consecutive depots, indices in range."""
    if not route:
        return False
    if route[0] != 0 or route[-1] != 0:
        return False
    # no consecutive depots
    for i in range(len(route) - 1):
        if route[i] == 0 and route[i+1] == 0:
            return False
    # index bounds
    max_idx = n_customers
    for node in route:
        if node < 0 or node > max_idx:
            return False
    return True

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
    warmup_epochs = int(config['warmup_epochs'])
    min_lr = float(config['min_lr'])
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
    
    # Split data using config value
    train_val_split = config.get('training', {}).get('train_val_split', 0.8)
    split_idx = int(train_val_split * len(instances))
    train_instances = instances[:split_idx]
    val_instances = instances[split_idx:]
    
    logger.info(f"🏋️ Training {model_name}...")
    
    # Strict validation flag controls depth of route checks, not frequency
    strict_flag = bool(config.get('experiment', {}).get('strict_validation', config.get('strict_validation', False)))

    for epoch in range(config['num_epochs'] + 1):
        # Training
        model.train()
        epoch_losses = []
        epoch_costs = []
        
        batch_size = config['batch_size']
        # Ensure we always run at least one batch to avoid empty means/NaNs
        num_batches = max(1, math.ceil(len(train_instances) / batch_size))
        
        # Temperature schedule: cosine from temp_start -> temp_min
        temp_start = float(config['temp_start'])
        temp_min = float(config['temp_min'])
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
            
            if model_name == 'Pointer+RL':
                routes, log_probs, entropies = model(batch_instances, max_steps=None, temperature=current_temp, greedy=False, config=config)
            elif model_name in ['GT+RL', 'GT-Greedy']:
                routes, log_probs, entropies = model(batch_instances, max_steps=None, temperature=current_temp, greedy=False, config=config)
            elif model_name in ['DGT+RL', 'GAT+RL']:
                routes, log_probs, entropies = model(batch_instances, max_steps=None, temperature=current_temp, greedy=False, config=config)
            else:
                routes, log_probs, entropies = model(batch_instances, temperature=current_temp)
            
            # Compute costs and validate routes
            costs = []
            normalized_costs = []
            for route, instance in zip(routes, batch_instances):
                n_customers = len(instance['coords']) - 1
                
                # VALIDATE ROUTE
                if strict_flag:
                    # Full, time-consuming checks (capacity, coverage, trips)
                    validate_route(route, n_customers, f"{model_name}-TRAIN", instance)
                else:
                    # Lightweight validation: basic structure and index bounds
                    if not quick_validate_route(route, n_customers):
                        print(f"\n🚨 VALIDATION FAILED (quick): {model_name}-TRAIN")
                        print(f"Route: {route}")
                        sys.exit(1)

                penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
                total_cost = compute_route_cost_with_penalty(route, instance['distances'], penalty)
                normalized_cost = total_cost / n_customers
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
        # Store training cost normalized per customer
        train_costs.append(np.mean(normalized_costs) if len(normalized_costs) > 0 else float('nan'))
        
        # Validation using config frequency, and ALWAYS at the final epoch
        validation_frequency = config.get('training', {}).get('validation_frequency', 3)
        if (epoch % validation_frequency == 0) or (epoch == int(config['num_epochs'])):
            model.eval()
            val_batch_costs = []
            val_batch_normalized = []
            
            with torch.no_grad():
                for i in range(0, len(val_instances), batch_size):
                    batch_val = val_instances[i:i + batch_size]
                    if model_name == 'Pointer+RL':
                        # Calculate max steps and temperature from config
                        max_steps_val = len(batch_val[0]['coords']) * config['inference']['max_steps_multiplier'] if batch_val else 0
                        temp_val = config['inference'].get('default_temperature', config['temp_min'])
                        routes, _, _ = model(batch_val, max_steps_val, temp_val, True, config)
                    elif model_name in ['GT+RL', 'GT-Greedy']:
                        routes, _, _ = model(batch_val, max_steps=None, temperature=None, greedy=True, config=config)
                    elif model_name in ['DGT+RL', 'GAT+RL']:
                        routes, _, _ = model(batch_val, max_steps=None, temperature=None, greedy=True, config=config)
                    else:
                        routes, _, _ = model(batch_val, greedy=True)
                    
                    for j, (route, instance) in enumerate(zip(routes, batch_val)):
                        n_customers = len(instance['coords']) - 1
                        
                        # VALIDATE ROUTE
                        if strict_flag:
                            validate_route(route, n_customers, f"{model_name}-VAL", instance)
                        else:
                            if not quick_validate_route(route, n_customers):
                                print(f"\n🚨 VALIDATION FAILED (quick): {model_name}-VAL")
                                print(f"Route: {route}")
                                sys.exit(1)

                        penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
                        total_cost = compute_route_cost_with_penalty(route, instance['distances'], penalty)
                        normalized_cost = total_cost / n_customers
                        val_batch_costs.append(total_cost)
                        val_batch_normalized.append(normalized_cost)
            
            val_cost = np.mean(val_batch_costs)
            val_normalized = np.mean(val_batch_normalized)
            # Store validation cost normalized per customer
            val_costs.append(val_normalized)
            
            logger.info(f"   Epoch {epoch:2d}: Loss={train_losses[-1]:.3f}, Train={np.mean(normalized_costs):.3f}/cust, Val={val_normalized:.3f}/cust")
        else:
            logger.info(f"   Epoch {epoch:2d}: Loss={train_losses[-1]:.3f}, Train={np.mean(normalized_costs):.3f}/cust")
    
    return {
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs,
        'final_val_cost': val_costs[-1] if val_costs else train_costs[-1]
    }


def load_config(config_path):
    """Unified config loader: deep-merge default + override, validate, normalize, and flatten.
    Ensures all parameters originate from YAML without hidden defaults.
    """
    from src.utils.config import load_config as _shared_load
    return _shared_load(config_path)

# Unified naming for artifacts (CSV and model files)
def model_key(name: str) -> str:
    mapping = {
        'Pointer+RL': 'pointer_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'GAT+RL': 'gat_rl',
        'GT-Greedy': 'gt_greedy',
        'GAT+RL (legacy)': 'gat_rl_legacy',
    }
    if name in mapping:
        return mapping[name]
    return name.lower().replace(' ', '_').replace('+', '_').replace('-', '_').replace('/', '_')

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
    parser.add_argument('--hidden_dim', type=int, default=None, help='Override hidden dimension from config')
    parser.add_argument('--entropy_coef', type=float, default=None, help='Override initial entropy regularization coefficient')
    parser.add_argument('--entropy_min', type=float, default=None, help='Override minimum entropy coefficient')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='Override warmup epochs')
    parser.add_argument('--min_lr', type=float, default=None, help='Override minimum LR')
    parser.add_argument('--temp_start', type=float, default=None, help='Override initial sampling temperature')
    parser.add_argument('--temp_min', type=float, default=None, help='Override minimum sampling temperature')
    parser.add_argument('--only_dgt', action='store_true', help='Train only the DGT+RL model to avoid extra deps')
    parser.add_argument('--exclude_dgt', action='store_true', help='Train all models except DGT+RL and reuse prior DGT results')
    parser.add_argument('--reuse_dgt_path', type=str, default=None, help='Path to prior DGT results file to reuse when --exclude_dgt is set')
    parser.add_argument('--only_greedy', action='store_true', help='Run only the GT-Greedy model (no training)')
    parser.add_argument('--only_gat_rl_legacy', action='store_true', help='Run only the legacy GAT+RL model (requires torch_geometric)')

    # New fine-grained single-model selectors
    parser.add_argument('--only_pointer', action='store_true', help='Run only Pointer+RL')
    parser.add_argument('--only_gt_rl', action='store_true', help='Run only GT+RL')
    parser.add_argument('--only_dgt_rl', action='store_true', help='Run only DGT+RL')
    parser.add_argument('--only_gat_rl', action='store_true', help='Run only GAT+RL (non-legacy)')
    parser.add_argument('--only_gt_greedy', action='store_true', help='Run only GT-Greedy (no RL training)')
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


def evaluate_greedy_model(model, instances, config, model_name, logger):
    """Evaluate the greedy (non-RL) model on the validation set only.
    No training/validation loops are executed. Returns an object mimicking the
    structure of training results so downstream code can persist and plot.
    """
    # Split data using config value
    train_val_split = config.get('training', {}).get('train_val_split', 0.8)
    split_idx = int(train_val_split * len(instances))
    val_instances = instances[split_idx:]

    logger.info(f"🧪 Evaluating {model_name} (greedy inference only; no training)...")

    batch_size = int(config['batch_size'])
    val_batch_costs = []
    with torch.no_grad():
        for i in range(0, len(val_instances), batch_size):
            batch_val = val_instances[i:i + batch_size]
            if not batch_val:
                continue
            routes, _, _ = model(batch_val, max_steps=None, temperature=None, greedy=True, config=config)
            for route, instance in zip(routes, batch_val):
                n_customers = len(instance['coords']) - 1
                # Validate and compute cost with optional depot penalty
                validate_route(route, n_customers, f"{model_name}-EVAL", instance)
                penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
                total_cost = compute_route_cost_with_penalty(route, instance['distances'], penalty)
                val_batch_costs.append(total_cost)

    final_val_cost = float(np.mean(val_batch_costs)) if val_batch_costs else float('nan')
    logger.info(f"   Final greedy evaluation cost: {final_val_cost:.3f} ({final_val_cost / config['num_customers']:.3f}/cust)")

    # Return minimal history (no epochs)
    return {
        'train_losses': [],
        'train_costs': [],
        # Store per-customer validation cost for consistency
        'val_costs': [final_val_cost / float(config['num_customers'])],
        'final_val_cost': final_val_cost / float(config['num_customers'])
    }

def train_legacy_gat_rl(model, instances, config, model_name, logger, base_dir):
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
    # Use config-based train/val split from config file
    train_val_split = config.get('training', {}).get('train_val_split', 0.8)
    split_idx = int(train_val_split * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    batch_size = config['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Legacy training parameters - made configurable from training_advanced section
    legacy_training = config.get('training_advanced', {}).get('legacy_gat', {})
    n_steps = config['num_customers'] * legacy_training.get('max_steps_multiplier', config['inference']['max_steps_multiplier'])
    T = legacy_training.get('temperature', config['inference']['default_temperature'])
    lr = legacy_training.get('learning_rate', config['learning_rate'])
    num_epochs = int(config['num_epochs'])
    total_epochs = num_epochs + 1  # inclusive 0..num_epochs

    # Run the legacy training loop (kept intact)
    logger.info(f"🏋️ Training {model_name} (legacy RL training)...")
    logger.info(f"   Legacy training config: lr={lr}, n_steps={n_steps}, num_epochs={num_epochs}, T={T}")
    ckpt_folder = os.path.join(base_dir, 'checkpoints', 'legacy_checkpoints')
    os.makedirs(ckpt_folder, exist_ok=True)
    # Ensure legacy log dir exists inside working dir
    legacy_log_dir = os.path.join(base_dir, 'logs', 'training')
    os.makedirs(legacy_log_dir, exist_ok=True)
    # Capture time to find latest CSV emitted by legacy loop (both default and working-dir locations)
    before_csv = set(glob.glob(os.path.join(legacy_log_dir, '*.csv'))) | set(glob.glob('logs/training/*.csv'))

    # Capture stdout from legacy training to parse per-epoch metrics printed on screen
    import io, re
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass
    stdout_buffer = io.StringIO()
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, stdout_buffer)
    try:
        legacy_train(model, train_loader, val_loader, ckpt_folder, 'actor.pt', lr, n_steps, num_epochs, T)
    finally:
        sys.stdout = orig_stdout
    legacy_stdout = stdout_buffer.getvalue()

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
                # Apply depot penalty if configured
                penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
                if penalty and penalty != 0.0:
                    internal_zeros = sum(1 for node in route[1:-1] if node == 0)
                    cost += penalty * internal_zeros
                val_epoch_costs.append(cost)
    # Normalize per customer for consistency
    final_val_cost = float(np.mean([c / float(config['num_customers']) for c in val_epoch_costs]) if val_epoch_costs else 0.0)

    # Extract training curves first from captured stdout (authoritative), then from CSV as fallback
    train_losses = [float('nan')] * total_epochs
    train_costs = [float('nan')] * total_epochs

    # Parse lines like: "Epoch 1, mean loss: 88.400, mean reward: 12.060, time: 3.75"
    try:
        # Match positive or negative floats for loss/reward; be tolerant to extra text between fields
        pattern = re.compile(r"Epoch\s+(\d+)\s*,\s*mean\s+loss:\s*([-+]?\d*\.?\d+)\s*,\s*mean\s+reward:\s*([-+]?\d*\.?\d+)")
        epoch_lines = pattern.findall(legacy_stdout)
        for ep_str, loss_str, reward_str in epoch_lines:
            ep = int(ep_str)
            if 0 <= ep <= num_epochs:
                train_losses[ep] = float(loss_str)
                # reward is total cost; normalize per customer
                train_costs[ep] = float(reward_str) / float(config['num_customers'])
    except Exception:
        pass

    # Extract training curves from the legacy CSV if available
    try:
        # Look for CSVs in working-dir legacy log path first, then fallback to default
        legacy_log_dir = os.path.join(base_dir, 'logs', 'training')
        after_csv = set(glob.glob(os.path.join(legacy_log_dir, '*.csv'))) | set(glob.glob('logs/training/*.csv'))
        new_csvs = sorted(list(after_csv - before_csv), key=lambda p: os.path.getmtime(p))
        csv_path = new_csvs[-1] if new_csvs else (sorted(list(after_csv), key=lambda p: os.path.getmtime(p))[-1] if after_csv else None)
        if csv_path is None:
            raise FileNotFoundError("No legacy CSV logs found after training")
        # If CSV is outside working dir, copy it in for persistence under working_dir_path
        try:
            if not os.path.commonpath([os.path.abspath(csv_path), os.path.abspath(legacy_log_dir)]) == os.path.abspath(legacy_log_dir):
                import shutil
                dest_path = os.path.join(legacy_log_dir, os.path.basename(csv_path))
                shutil.copy2(csv_path, dest_path)
                csv_path = dest_path
        except Exception:
            pass
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
            for i, v in enumerate(legacy_losses):
                if i < num_epochs and (np.isnan(train_losses[i]) or train_losses[i] is None):
                    train_losses[i] = v
        if 'mean_reward' in df.columns:
            # Legacy CSV stores mean_reward as total cost; convert to per-customer cost
            legacy_costs = [float(x) / float(config['num_customers']) for x in df['mean_reward'].tolist()][:num_epochs]
            logger.info(f"   💰 Extracted {len(legacy_costs)} cost values (requested {num_epochs}) [normalized per customer]")
            for i, v in enumerate(legacy_costs):
                if i < num_epochs and (np.isnan(train_costs[i]) or train_costs[i] is None):
                    train_costs[i] = v
    except Exception as e:
        # CRITICAL: Legacy CSV extraction failure indicates training data corruption or missing files
        raise RuntimeError(f"❌ CRITICAL: Failed to extract legacy training CSV data for {model_name}. This indicates corrupted training logs or missing CSV files. Error: {e}")

    # Ensure we have lists of length total_epochs (0..num_epochs inclusive)
    train_losses = (train_losses + [float('nan')] * max(0, total_epochs - len(train_losses)))[:total_epochs]
    train_costs = (train_costs + [float('nan')] * max(0, total_epochs - len(train_costs)))[:total_epochs]

    # Compute per-epoch training cost by evaluating each checkpoint on the training set (greedy, per-customer) only if missing
    # This fills in epochs that the legacy CSV didn't record
    try:
        # Short-circuit if we already have all epochs filled
        if all([not np.isnan(x) for x in train_costs]):
            train_costs_full = train_costs
        else:
            train_costs_full = [float('nan')] * total_epochs
        with torch.no_grad():
            for e in range(total_epochs):
                ckpt_path = os.path.join(ckpt_folder, f'{e}', 'actor.pt')
                if not os.path.exists(ckpt_path):
                    continue
                # Load checkpoint
                try:
                    state = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        model.load_state_dict(state['model_state_dict'])
                    else:
                        model.load_state_dict(state)
                except Exception:
                    pass
                # Evaluate on training set for this epoch
                epoch_costs = []
                for batch in train_loader:
                    actions, tour_logp = model(batch, n_steps=n_steps, greedy=True, T=T)
                    depot = torch.zeros(actions.size(0), 1, dtype=torch.long)
                    actions_with_depot = torch.cat([depot, actions, depot], dim=1)
                    ptr = batch.ptr if hasattr(batch, 'ptr') else None
                    for b in range(actions_with_depot.size(0)):
                        route = actions_with_depot[b].cpu().tolist()
                        if ptr is not None:
                            start = int(ptr[b].item())
                            end = int(ptr[b + 1].item())
                            coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                        else:
                            coords = batch.x.view(-1, 2).cpu().numpy()
                        # Compute Euclidean route length
                        csum = 0.0
                        for i in range(len(route) - 1):
                            a = route[i]
                            c = route[i + 1]
                            pa = coords[a]
                            pb = coords[c]
                            csum += float(((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2) ** 0.5)
                        # Apply depot penalty if configured
                        penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
                        if penalty and penalty != 0.0:
                            internal_zeros = sum(1 for node in route[1:-1] if node == 0)
                            csum += penalty * internal_zeros
                        epoch_costs.append(csum)
                # Normalize per customer
                if epoch_costs:
                    train_costs_full[e] = float(np.mean([c / float(config['num_customers']) for c in epoch_costs]))
        # Prefer computed full series; fallback to CSV where computed is NaN
        if not train_costs:
            train_costs = train_costs_full
        else:
            train_costs = [train_costs[i] if i < len(train_costs) and not np.isnan(train_costs[i]) else train_costs_full[i] for i in range(total_epochs)]
    except Exception:
        # If any issue, keep CSV-derived costs and pad
        if len(train_costs) < total_epochs:
            train_costs += [float('nan')] * (total_epochs - len(train_costs))

    # Build validation curve by evaluating checkpoints at epochs [0, 4, 8, ...]
    val_costs = [float('nan')] * total_epochs
    eval_epochs = sorted(set(list(range(0, total_epochs, 4)) + [num_epochs]))
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
                        penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
                        if penalty and penalty != 0.0:
                            internal_zeros = sum(1 for node in route[1:-1] if node == 0)
                            cost += penalty * internal_zeros
                        epoch_costs.append(cost)
                # Normalize per customer for validation as well
                val_costs[e] = float(np.mean([c / float(config['num_customers']) for c in epoch_costs]) if epoch_costs else float('nan'))
            else:
                # leave as NaN for epochs without checkpoints
                pass

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

    # Ensure terminal validation point is included at epoch=num_epochs
    if np.isfinite(final_val_cost):
        val_costs[num_epochs] = float(final_val_cost)

    # Backfill final epoch training metrics if missing
    try:
        import math as _math
        # train_cost at final epoch (compute from final model on training set)
        if num_epochs < len(train_costs) and (np.isnan(train_costs[num_epochs]) or train_costs[num_epochs] is None):
            epoch_costs = []
            with torch.no_grad():
                for batch in train_loader:
                    actions, tour_logp = model(batch, n_steps=n_steps, greedy=True, T=T)
                    depot = torch.zeros(actions.size(0), 1, dtype=torch.long)
                    actions_with_depot = torch.cat([depot, actions, depot], dim=1)
                    ptr = batch.ptr if hasattr(batch, 'ptr') else None
                    for b in range(actions_with_depot.size(0)):
                        route = actions_with_depot[b].cpu().tolist()
                        if ptr is not None:
                            start = int(ptr[b].item())
                            end = int(ptr[b + 1].item())
                            coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                        else:
                            coords = batch.x.view(-1, 2).cpu().numpy()
                        csum = 0.0
                        for i in range(len(route) - 1):
                            a = route[i]
                            c = route[i + 1]
                            pa = coords[a]
                            pb = coords[c]
                            csum += float(((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2) ** 0.5)
                        penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
                        if penalty and penalty != 0.0:
                            internal_zeros = sum(1 for node in route[1:-1] if node == 0)
                            csum += penalty * internal_zeros
                        epoch_costs.append(csum)
            if epoch_costs:
                train_costs[num_epochs] = float(np.mean([c / float(config['num_customers']) for c in epoch_costs]))
        # train_loss at final epoch: copy last available epoch loss if missing
        if num_epochs < len(train_losses) and (np.isnan(train_losses[num_epochs]) or train_losses[num_epochs] is None):
            if num_epochs - 1 >= 0 and not np.isnan(train_losses[num_epochs - 1]):
                train_losses[num_epochs] = float(train_losses[num_epochs - 1])
    except Exception:
        pass
    
    # Log explicit final epoch validation line for legacy model
    try:
        logger.info(f"   Epoch {int(num_epochs)}: Val={final_val_cost:.3f}/cust")
    except Exception:
        pass

    return {
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs,  # length total_epochs with NaNs except evaluated epochs
        'final_val_cost': final_val_cost
    }


def run_comparative_study():
    """Main function to run all 6 pipelines and compare"""
    args = parse_args()
    
    # Load base configuration from YAML
    config = load_config(args.config)
    
    # Initialize system settings from config
    logger = setup_logging(config)
    device = get_device_from_config(config)
    set_seeds(config)

    # Global strict validation switch (depth only, not frequency)
    strict_flag_global = bool(config.get('experiment', {}).get('strict_validation', config.get('strict_validation', False)))
    
    # Configure CPU optimization if using CPU
    if device.type == 'cpu':
        configure_cpu_threads(config)
    
    logger.info(f"🚀 Starting Comparative Study: 6 Pipeline Architectures ({device.type.upper()})")
    
    # Base working directory for all artifacts
    base_dir = str(Path(config.get('working_dir_path', 'results')).as_posix())
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # STRICT CONFIG VALIDATION: Extract nested config values with mandatory checks
    # Problem settings - REQUIRED
    if 'problem' not in config:
        raise ValueError("❌ CRITICAL: Missing 'problem' section in configuration file")
    
    required_problem_keys = ['num_customers', 'vehicle_capacity', 'coord_range', 'demand_range']
    for key in required_problem_keys:
        if key not in config['problem']:
            raise ValueError(f"❌ CRITICAL: Missing required problem configuration: '{key}'")
    
    config.update({
        'num_customers': config['problem']['num_customers'],
        'capacity': config['problem']['vehicle_capacity'],
        'coord_range': config['problem']['coord_range'],
        'demand_range': config['problem']['demand_range']
    })
    
    # Training settings - REQUIRED
    if 'training' not in config:
        raise ValueError("❌ CRITICAL: Missing 'training' section in configuration file")
    
    required_training_keys = ['num_instances', 'batch_size', 'num_epochs', 'learning_rate']
    for key in required_training_keys:
        if key not in config['training']:
            raise ValueError(f"❌ CRITICAL: Missing required training configuration: '{key}'")
    
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
        penalty_cfg = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
        naive_costs.append(compute_naive_baseline_cost(instance, penalty_cfg))
    
    naive_avg_cost = np.mean(naive_costs)
    naive_normalized = naive_avg_cost / config['num_customers']
    if config.get('cost', {}).get('depot_penalty_per_visit', 0.0):
        logger.info(f"📍 Naive baseline (with depot penalty): {naive_avg_cost:.3f} ({naive_normalized:.3f}/cust)")
    else:
        logger.info(f"📍 Naive baseline (depot->customer->depot): {naive_avg_cost:.3f} ({naive_normalized:.3f}/cust)")
    
    # Initialize models with config-based parameters
    input_dim = config.get('model', {}).get('input_dim', 3)
    dropout = config.get('model', {}).get('transformer_dropout', 0.1)
    feedforward_multiplier = config.get('model', {}).get('feedforward_multiplier', 4)
    edge_embedding_divisor = config.get('model', {}).get('edge_embedding_divisor', 4)
    models = {
        'Pointer+RL': BaselinePointerNetwork(input_dim, config['hidden_dim'], config),
        'GT-Greedy': GraphTransformerGreedy(input_dim, config['hidden_dim'], config['num_heads'], config['num_layers'], dropout, feedforward_multiplier, config),
        'GT+RL': GraphTransformerNetwork(input_dim, config['hidden_dim'], config['num_heads'], config['num_layers'], dropout, feedforward_multiplier, config),
        'DGT+RL': DynamicGraphTransformerNetwork(input_dim, config['hidden_dim'], config['num_heads'], config['num_layers'], dropout, feedforward_multiplier, config),
        'GAT+RL': GraphAttentionTransformer(input_dim, config['hidden_dim'], config['num_heads'], config['num_layers'], dropout, edge_embedding_divisor, config)
    }
    
    # Optionally include legacy model if not restricted to DGT and dependency is available
    if not args.only_dgt:
        try:
            from src_batch.model.Model import Model as LegacyGATModel
            # Legacy GAT model with config-based parameters where applicable
            legacy_config = config.get('model', {}).get('legacy_gat', {})
            models['GAT+RL (legacy)'] = LegacyGATModel(
                node_input_dim=3, 
                edge_input_dim=1, 
                hidden_dim=legacy_config.get('hidden_dim', config.get('hidden_dim', 128)), 
                edge_dim=legacy_config.get('edge_dim', 16),
                layers=legacy_config.get('layers', config.get('num_layers', 4)), 
                negative_slope=legacy_config.get('negative_slope', 0.2), 
                dropout=legacy_config.get('dropout', 0.6)
            )
        except Exception as e:
            # CRITICAL: Legacy model loading failure requires torch_geometric dependency
            if 'only_dgt' not in vars(args) or not args.only_dgt:
                raise RuntimeError(f"❌ CRITICAL: Legacy GAT+RL model loading failed. This likely indicates missing torch_geometric dependency. Use --only_dgt flag to skip legacy models or install torch_geometric. Error: {e}")
            logger.warning(f"Legacy GAT+RL unavailable (torch_geometric not installed?): {e}")
    
    # Training results storage
    results = {}
    training_times = {}
    
    # Unified single-model selection handling (new flags + existing ones)
    single_flags = {
        'only_pointer': 'Pointer+RL',
        'only_gt_rl': 'GT+RL',
        'only_dgt_rl': 'DGT+RL',
        'only_gat_rl': 'GAT+RL',
        'only_gt_greedy': 'GT-Greedy',
        'only_gat_rl_legacy': 'GAT+RL (legacy)',
        'only_greedy': 'GT-Greedy',  # backward-compatible alias
        'only_dgt': 'DGT+RL',        # backward-compatible alias
    }
    selected = [name for name in single_flags if getattr(args, name, False)]
    # If multiple are specified, raise to avoid ambiguity
    if len(selected) > 1:
        raise RuntimeError(f"Multiple --only_* flags specified: {selected}. Please specify exactly one.")
    if len(selected) == 1:
        target_model = single_flags[selected[0]]
        if target_model not in models:
            raise RuntimeError(f"{selected[0]} was requested, but '{target_model}' is unavailable (missing deps or excluded).")
        models = {target_model: models[target_model]}
    else:
        # No single selector provided — apply optional exclusion
        if args.exclude_dgt and 'DGT+RL' in models:
            models.pop('DGT+RL')
    
    # Use global model_key() for naming artifacts consistently

    # Train each model
    for model_name, model in models.items():
        logger.info(f"\n🎯 Training {model_name}")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        start_time = time.time()
        if model_name == 'GAT+RL (legacy)':
            result = train_legacy_gat_rl(model, instances, config, model_name, logger, base_dir)
        elif model_name == 'GT-Greedy':
            # Greedy model has no RL; skip training/validation loops and only evaluate on validation set
            result = evaluate_greedy_model(model, instances, config, model_name, logger)
        else:
            result = train_model(model, instances, config, model_name, logger)
        training_time = time.time() - start_time
        
        results[model_name] = result
        training_times[model_name] = training_time
        
        # Immediately dump per-model training history to CSV (epochs-aligned)
        try:
            # Target number of epochs from config
            num_epochs = int(config['num_epochs'])
            train_losses = result.get('train_losses', [])
            train_costs = result.get('train_costs', [])
            vc = result.get('val_costs', [])
            val_freq = int(config.get('training', {}).get('validation_frequency', 3))

            if model_name == 'GAT+RL (legacy)':
                # Legacy: now write epochs 0..num_epochs inclusive with sparse val_cost series already aligned
                total_epochs = num_epochs + 1
                # vc in result is a per-epoch list already aligned (length total_epochs)
                val_costs_aligned = (vc + [float('nan')] * max(0, total_epochs - len(vc)))[:total_epochs]
                df_hist = pd.DataFrame({
                    'epoch': list(range(total_epochs)),
                    'train_loss': (train_losses + [float('nan')] * max(0, total_epochs - len(train_losses)))[:total_epochs],
                    'train_cost': (train_costs + [float('nan')] * max(0, total_epochs - len(train_costs)))[:total_epochs],
                    'val_cost': val_costs_aligned
                })
            else:
                # Non-legacy: epochs 0..num_epochs inclusive with full metrics
                total_epochs = num_epochs + 1
                # Map validation costs to epochs assuming frequency and forced final epoch
                val_costs_sparse = [float('nan')] * total_epochs
                idx = 0
                for ep in range(0, num_epochs + 1):
                    should_val = (ep % max(1, val_freq) == 0) or (ep == num_epochs)
                    if should_val and idx < len(vc):
                        val_costs_sparse[ep] = vc[idx]
                        idx += 1
                df_hist = pd.DataFrame({
                    'epoch': list(range(total_epochs)),
                    'train_loss': (train_losses + [float('nan')] * max(0, total_epochs - len(train_losses)))[:total_epochs],
                    'train_cost': (train_costs + [float('nan')] * max(0, total_epochs - len(train_costs)))[:total_epochs],
                    'val_cost': val_costs_sparse
                })
            
            # Ensure CSV directory exists
            csv_dir = os.path.join(base_dir, "csv")
            os.makedirs(csv_dir, exist_ok=True)
            
            out_hist = os.path.join(csv_dir, f"history_{model_key(model_name)}.csv")
            df_hist.to_csv(out_hist, index=False)
            logger.info(f"   🧾 Saved history CSV: {out_hist}")
        except Exception as e:
            # CRITICAL: CSV saving failure indicates serious file system or data integrity issues
            raise RuntimeError(f"❌ CRITICAL: Failed to save training history CSV for {model_name}. This indicates file system issues or data corruption. Error: {e}")
        
        logger.info(f"   ✅ {model_name} completed in {training_time:.1f}s")
        logger.info(f"   Final validation cost: {result['final_val_cost']:.3f}/cust")
        # For legacy, also log last available training entries for epoch=num_epochs
        if model_name == 'GAT+RL (legacy)':
            try:
                logger.info(f"   Epoch {num_epochs}: train_loss={train_losses[min(num_epochs,len(train_losses)-1)]}, train_cost={train_costs[min(num_epochs,len(train_costs)-1)]}, val_cost={vc[min(num_epochs,len(vc)-1)] if vc else float('nan')}")
            except Exception:
                pass
    
    # If we excluded DGT, try to reuse prior DGT results for comparison
    if args.exclude_dgt:
        try:
            # Use scale-aware path if no explicit path provided
            reuse_path = args.reuse_dgt_path or os.path.join(base_dir, 'analysis', 'comparative_study_complete.pt')
            prior = torch.load(reuse_path, map_location='cpu')
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
            sd = torch.load(os.path.join(base_dir, 'pytorch', 'model_dgtplus_rl.pt'), map_location='cpu')
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
    
    # Save results
    save_results(results, training_times, models, config, base_dir=base_dir)

    # Optionally run strict baseline validation (time-consuming)
    if strict_flag_global:
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
        logger.info(f"   Final validation cost: {result['final_val_cost']:.2f}/cust")
        # Safe handling for models without training history (e.g., Greedy)
        train_costs_list = result.get('train_costs') or []
        last_train_cost = train_costs_list[-1] if len(train_costs_list) > 0 else float('nan')
        logger.info(f"   Final training cost: {last_train_cost:.2f}")
        logger.info("")
    
    # Save results
    save_results(results, training_times, models, config, base_dir=base_dir)
    
    return results


def validate_naive_baseline_correctness(results, naive_avg_cost, config, logger, instances):
    """STRICT VALIDATION: Naive baseline is absolute maximum - no model can exceed it, even slightly"""
    
    # CRITICAL FIX: Calculate naive baseline from VALIDATION SET ONLY (same data used for model validation)
    split_idx = int(0.8 * len(instances))
    val_instances = instances[split_idx:]
    
    penalty_cfg = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
    val_naive_costs = [compute_naive_baseline_cost(inst, penalty_cfg) for inst in val_instances]
    val_naive_avg = np.mean(val_naive_costs)
    val_naive_normalized = val_naive_avg / config['num_customers']
    
    logger.info("\n🔍 STRICT VALIDATION: Checking naive baseline correctness...")
    if penalty_cfg:
        logger.info(f"📊 Training set naive baseline (with depot penalty): {naive_avg_cost / config['num_customers']:.4f} cost/customer")
        logger.info(f"📊 Validation set naive baseline (with depot penalty, used for comparison): {val_naive_normalized:.4f} cost/customer")
    else:
        logger.info(f"📊 Training set naive baseline: {naive_avg_cost / config['num_customers']:.4f} cost/customer")
        logger.info(f"📊 Validation set naive baseline (used for comparison): {val_naive_normalized:.4f} cost/customer")
    
    validation_passed = True
    violations = []
    
    for model_name, result in results.items():
        # final_val_cost is already normalized per customer
        final_cost_normalized = result['final_val_cost']
        
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

def save_results(results, training_times, models, config, base_dir: str):
    """Save all results and models to organized results structure under working_dir_path"""
    
    # Create organized directories
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    analysis_dir = os.path.join(base_dir, 'analysis')
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save individual models to pytorch directory
    for model_name, model in models.items():
        filename = os.path.join(pytorch_dir, f"model_{model_key(model_name)}.pt")
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
        'config': config
    }, os.path.join(analysis_dir, 'comparative_study_complete.pt'))

if __name__ == "__main__":
    results = run_comparative_study()
    print(f"\n🎉 Training and validation completed! Results saved to organized directory structure.")
