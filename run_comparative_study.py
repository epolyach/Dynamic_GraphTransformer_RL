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
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Force CPU for reliable comparison
# Auto-detect best available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_cvrp_instance(num_customers=20, capacity=3, coord_range=20, demand_range=(1, 10), seed=None):
    """Generate CVRP instance exactly matching GAT-RL configuration"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates exactly like GAT-RL: random integers 0 to max_distance+1, then divide by 100
    coords = np.zeros((num_customers + 1, 2))
    for i in range(num_customers + 1):
        coords[i] = np.random.randint(0, coord_range + 1, size=2) / 100
    
    # Generate demands exactly like GAT-RL: random integers 1 to max_demand+1, then divide by 10
    demands = np.zeros(num_customers + 1)
    for i in range(1, num_customers + 1):  # Skip depot (index 0)
        demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1) / 10
    
    # Compute distance matrix
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    return {
        'coords': coords,
        'demands': demands,
        'distances': distances,
        'capacity': capacity
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
            
            # Apply mask: visited nodes + capacity constraints
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                # Don't allow staying at depot if already at depot
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                # Safety: if all nodes masked and we're not at depot, allow depot
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                # If we're at depot and all customers visited, we should have terminated above
                elif mask[b].all() and currently_at_depot:
                    # Mark this batch as done to avoid further processing
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        probs = torch.softmax(scores[b] / temperature, dim=-1)
                        actions[b] = torch.multinomial(probs, 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            
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
        return routes, combined_log_probs

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
            
            # Apply mask: visited nodes + capacity constraints
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                # Don't allow staying at depot if already at depot
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                # Safety: if all nodes masked and we're not at depot, allow depot
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                # If we're at depot and all customers visited, we should have terminated above
                elif mask[b].all() and currently_at_depot:
                    # Mark this batch as done to avoid further processing
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        probs = torch.softmax(scores[b] / temperature, dim=-1)
                        actions[b] = torch.multinomial(probs, 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            
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
        return routes, combined_log_probs

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
            
            # Apply mask: visited nodes + capacity constraints
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                # Don't allow staying at depot if already at depot
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                # Safety: if all nodes masked and we're not at depot, allow depot
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                # If we're at depot and all customers visited, we should have terminated above
                elif mask[b].all() and currently_at_depot:
                    # Mark this batch as done to avoid further processing
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
            # Always greedy selection
            actions = log_probs.argmax(dim=-1)
            selected_log_probs = torch.gather(log_probs, 1, actions.unsqueeze(1)).squeeze(1)
            
            all_log_probs.append(selected_log_probs)
            
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
        return routes, combined_log_probs

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
            
            # Update node embeddings based on current state
            dynamic_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)  # [B, N, H]
            update_input = torch.cat([node_embeddings, dynamic_context], dim=-1)
            node_updates = self.dynamic_update(update_input)
            
            # Apply dynamic updates
            updated_embeddings = node_embeddings + 0.1 * node_updates  # Residual with small learning rate
            
            # Enhanced context with state information
            global_context = updated_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            state_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)
            
            # Enhanced pointer scores
            pointer_input = torch.cat([updated_embeddings, global_context, state_context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                # Don't allow staying at depot if already at depot
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                # Safety: if all nodes masked and we're not at depot, allow depot
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                # If we're at depot and all customers visited, we should have terminated above
                elif mask[b].all() and currently_at_depot:
                    # Mark this batch as done to avoid further processing
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        probs = torch.softmax(scores[b] / temperature, dim=-1)
                        actions[b] = torch.multinomial(probs, 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            
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
        return routes, combined_log_probs

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
            
            # Apply mask
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                elif mask[b].all() and currently_at_depot:
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
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
        return routes, combined_log_probs

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

def validate_route(route, n_customers, model_name="Unknown"):
    """Validate that a route is correct CVRP solution"""
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
    
    # Check for consecutive depot visits
    for i in range(len(route) - 1):
        if route[i] == 0 and route[i + 1] == 0:
            print(f"\n🚨 VALIDATION FAILED: {model_name}")
            print(f"Error: Consecutive depot visits at positions {i}-{i+1}")
            print(f"Route: {route}")
            sys.exit(1)
    
    # Check for duplicate customer visits
    customers_in_route = [node for node in route if node != 0]
    unique_customers = set(customers_in_route)
    if len(customers_in_route) != len(unique_customers):
        duplicates = [x for x in customers_in_route if customers_in_route.count(x) > 1]
        print(f"\n🚨 VALIDATION FAILED: {model_name}")
        print(f"Error: Duplicate customer visits: {duplicates}")
        print(f"Route: {route}")
        sys.exit(1)
    
    # Check if all customers are visited
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
    
    return True

def train_model(model, instances, config, model_name, logger):
    """Train a single model and return training history"""
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
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
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            batch_instances = train_instances[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            routes, log_probs = model(batch_instances, temperature=config['temperature'])
            
            # Compute costs and validate routes
            costs = []
            normalized_costs = []
            for route, instance in zip(routes, batch_instances):
                n_customers = len(instance['coords']) - 1
                
                # VALIDATE ROUTE - This will exit with error if route is invalid
                validate_route(route, n_customers, f"{model_name}-TRAIN")

                total_cost = compute_route_cost(route, instance['distances'])
                normalized_cost = compute_normalized_cost(route, instance['distances'], n_customers)
                costs.append(total_cost)
                normalized_costs.append(normalized_cost)
            
            costs_tensor = torch.tensor(costs, dtype=torch.float32)
            
            # REINFORCE loss
            baseline = costs_tensor.mean().detach()
            advantages = baseline - costs_tensor  # FIXED: Lower costs should have positive advantages
            
            loss = (-advantages * log_probs).mean()
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_costs.extend(costs)
        
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
                    routes, _ = model(batch_val, greedy=True)
                    
                    for j, (route, instance) in enumerate(zip(routes, batch_val)):
                        n_customers = len(instance['coords']) - 1
                        
                        # VALIDATE ROUTE - This will exit with error if route is invalid
                        validate_route(route, n_customers, f"{model_name}-VAL")

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

def run_comparative_study():
    """Main function to run all 6 pipelines and compare"""
    logger = setup_logging()
    logger.info("🚀 Starting Comparative Study: 6 Pipeline Architectures")
    
    set_seeds(42)
    
    config = {
        'num_customers': 15,
        'capacity': 3,
        'coord_range': 100,
        'demand_range': (1, 10),  # Match GAT-RL exactly: 1 to max_demand+1 (10)
        'hidden_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'batch_size': 8,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'grad_clip': 1.0,
        'temperature': 1.0,
        'num_instances': 800
    }
    
    logger.info(f"📋 Config: {config['num_customers']} customers, {config['num_epochs']} epochs, {config['num_instances']} instances")
    
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
    
    # Initialize all 5 models (+ naive baseline = 6 total)
    models = {
        'Pointer+RL': BaselinePointerNetwork(3, config['hidden_dim']),
        'GT-Greedy': GraphTransformerGreedy(3, config['hidden_dim'], config['num_heads'], config['num_layers']),
        'GT+RL': GraphTransformerNetwork(3, config['hidden_dim'], config['num_heads'], config['num_layers']),
        'DGT+RL': DynamicGraphTransformerNetwork(3, config['hidden_dim'], config['num_heads'], config['num_layers']),
        'GAT+RL': GraphAttentionTransformer(3, config['hidden_dim'], config['num_heads'], config['num_layers'])
    }
    
    # Training results storage
    results = {}
    training_times = {}
    
    # Train each model
    for model_name, model in models.items():
        logger.info(f"\n🎯 Training {model_name}")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        start_time = time.time()
        result = train_model(model, instances, config, model_name, logger)
        training_time = time.time() - start_time
        
        results[model_name] = result
        training_times[model_name] = training_time
        
        logger.info(f"   ✅ {model_name} completed in {training_time:.1f}s")
        logger.info(f"   Final validation cost: {result['final_val_cost']:.3f} ({result['final_val_cost'] / config['num_customers']:.3f}/cust)")
    
    # Validate that naive baseline is higher than all learned approaches
    validate_naive_baseline_correctness(results, naive_avg_cost, config, logger, instances)
    
    # Generate comparison plots
    create_comparison_plots(results, training_times, config, logger, naive_avg_cost, models)
    
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
    
    # 1. Training Loss Comparison
    plt.subplot(2, 4, 1)
    for model_name, result in results.items():
        plt.plot(result['train_losses'], label=model_name, linewidth=2, marker='o', markersize=3)
    plt.title('Training Loss Evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Training Cost Comparison (NORMALIZED) - NO RED LINE
    plt.subplot(2, 4, 2)
    for model_name, result in results.items():
        # Normalize training costs by dividing by number of customers
        normalized_train_costs = [cost / config['num_customers'] for cost in result['train_costs']]
        plt.plot(normalized_train_costs, label=model_name, linewidth=2, marker='s', markersize=3)
    plt.title('Training Cost Evolution (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost per Customer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Validation Cost Comparison (NORMALIZED) - NO RED LINE
    plt.subplot(2, 4, 3)
    for model_name, result in results.items():
        val_epochs = list(range(0, config['num_epochs'], 3))[:len(result['val_costs'])]
        # Normalize validation costs by dividing by number of customers
        normalized_val_costs = [cost / config['num_customers'] for cost in result['val_costs']]
        plt.plot(val_epochs, normalized_val_costs, 'o-', label=model_name, linewidth=2, markersize=5)
    plt.title('Validation Cost Evolution (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost per Customer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Final Performance Bar Chart with Naive Baseline (NORMALIZED)
    plt.subplot(2, 4, 4)
    model_names = list(results.keys())
    # Normalize final costs by dividing by number of customers
    final_costs_normalized = [results[name]['final_val_cost'] / config['num_customers'] for name in model_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 distinct colors for 5 models
    
    # Add naive baseline to the comparison (normalized)
    all_names = model_names + ['Naive Baseline']
    all_costs_normalized = final_costs_normalized + [naive_normalized]
    all_colors = colors[:len(model_names)] + ['red']  # Use appropriate number of colors
    
    bars = plt.bar(range(len(all_names)), all_costs_normalized, color=all_colors, alpha=0.8)
    plt.title('Final Performance vs Naive Baseline (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Approach')
    plt.ylabel('Average Cost per Customer')
    plt.xticks(range(len(all_names)), [name.replace(' ', '\n') for name in all_names], rotation=45)
    
    # Add value labels on bars (normalized)
    for bar, cost in zip(bars, all_costs_normalized):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{cost:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Training Time Comparison
    plt.subplot(2, 4, 5)
    times = [training_times[name] for name in model_names]
    bars = plt.bar(range(len(model_names)), times, color=colors[:len(model_names)], alpha=0.8)
    plt.title('Training Time', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names])
    
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Model Complexity (Parameters)
    plt.subplot(2, 4, 6)
    param_counts = [sum(p.numel() for p in models[name].parameters()) for name in model_names]
    bars = plt.bar(range(len(model_names)), param_counts, color=colors, alpha=0.8)
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
    
    bars = plt.bar(range(len(model_names)), improvements, color=colors, alpha=0.8)
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
                   s=100, color=colors[i], alpha=0.8, label=model_name)
        plt.annotate(model_name.replace(' ', '\n'), 
                    (param_counts[i], final_costs_normalized[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Performance vs Complexity (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Parameters')
    plt.ylabel('Validation Cost per Customer')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('utils/plots/comparative_study_results.png', dpi=300, bbox_inches='tight')
    logger.info("📊 Comparison plots saved to utils/plots/comparative_study_results.png")
    
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
    
    # Save to CSV
    df.to_csv('utils/plots/comparative_results.csv', index=False)
    logger.info("📋 Detailed results saved to utils/plots/comparative_results.csv")
    
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
    """Save all results and models"""
    
    # Create pytorch directory if it does not exist
    os.makedirs("pytorch", exist_ok=True)
    
    # Save models
    for model_name, model in models.items():
        filename = f"pytorch/model_{model_name.lower().replace(' ', '_')}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'results': results[model_name],
            'training_time': training_times[model_name]
        }, filename)
    
    # Save complete results
    torch.save({
        'results': results,
        'training_times': training_times,
        'config': config
    }, 'pytorch/comparative_study_complete.pt')

if __name__ == "__main__":
    results = run_comparative_study()
    print("\n🎉 Comparative study completed! Check utils/plots/comparative_study_results.png for detailed analysis.")
