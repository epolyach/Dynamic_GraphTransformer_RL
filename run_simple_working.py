#!/usr/bin/env python3
"""
Simple Working CVRP Training
Focuses on getting the basic pipeline working with proper REINFORCE gradients
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Force CPU for reliable training
device = torch.device('cpu')
print(f"🖥️  Using device: {device}")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_cvrp_instance(num_customers=10, capacity=20, coord_range=50, demand_range=(1, 5), seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates (depot at center)
    coords = np.random.uniform(0, coord_range, (num_customers + 1, 2))
    coords[0] = [coord_range/2, coord_range/2]  # Depot at center
    
    # Generate demands (depot has 0 demand)
    demands = np.zeros(num_customers + 1)
    demands[1:] = np.random.randint(demand_range[0], demand_range[1] + 1, num_customers)
    
    # Compute distance matrix
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    return {
        'coords': coords,
        'demands': demands, 
        'distances': distances,
        'capacity': capacity
    }

class PointerNetwork(nn.Module):
    """Simple pointer network for CVRP"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Attention layers
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Pointer mechanism
        self.pointer_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_features, mask=None):
        """
        Args:
            node_features: [batch_size, num_nodes, input_dim]
            mask: [batch_size, num_nodes] (True for masked positions)
        Returns:
            log_probs: [batch_size, num_nodes] 
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Embed nodes
        embedded = self.node_embedding(node_features)  # [B, N, H]
        
        # Self-attention
        attended, _ = self.attention(embedded, embedded, embedded)  # [B, N, H]
        
        # Compute context (mean of all nodes)
        context = attended.mean(dim=1, keepdim=True)  # [B, 1, H]
        context = context.expand(-1, num_nodes, -1)  # [B, N, H]
        
        # Pointer scores
        pointer_input = torch.cat([attended, context], dim=-1)  # [B, N, 2H]
        scores = self.pointer_net(pointer_input).squeeze(-1)  # [B, N]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Return log probabilities
        log_probs = torch.log_softmax(scores, dim=-1)
        return log_probs

class SimpleCVRPSolver(nn.Module):
    """Simple CVRP solver using pointer network"""
    
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.pointer_net = PointerNetwork(
            input_dim=3,  # x, y, demand
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        """
        Solve CVRP instances
        Args:
            instances: list of instance dicts
        Returns:
            routes: list of routes
            log_probs: tensor of log probabilities
        """
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
        
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        
        # Initialize state
        current_nodes = torch.zeros(batch_size, dtype=torch.long)  # Start at depot (node 0)
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        visited[:, 0] = True  # Mark depot as visited initially
        
        for step in range(max_steps):
            # Create mask for infeasible nodes
            mask = visited.clone()
            
            # Mask nodes that would exceed capacity
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
            
            # Always allow depot (for returning)
            mask[:, 0] = False
            
            # Get action probabilities
            log_probs = self.pointer_net(node_features, mask)  # [B, N]
            
            if greedy:
                next_nodes = log_probs.argmax(dim=1)
            else:
                # Sample from distribution
                probs = torch.softmax(log_probs / temperature, dim=1)
                next_nodes = torch.multinomial(probs, 1).squeeze(1)
            
            # Store log probabilities for selected actions
            selected_log_probs = log_probs.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
            all_log_probs.append(selected_log_probs)
            
            # Update routes and state
            for b in range(batch_size):
                next_node = next_nodes[b].item()
                routes[b].append(next_node)
                
                if next_node == 0:  # Returned to depot
                    remaining_capacity[b] = capacities[b]
                    visited[b] = torch.zeros(max_nodes, dtype=torch.bool)
                    visited[b, 0] = True
                else:
                    visited[b, next_node] = True
                    remaining_capacity[b] -= demands_batch[b, next_node]
                
                current_nodes[b] = next_node
            
            # Check if all done (all customers visited)
            all_customers_visited = True
            for b in range(batch_size):
                n_nodes = len(instances[b]['coords'])
                if not visited[b, 1:n_nodes].all():
                    all_customers_visited = False
                    break
            
            if all_customers_visited:
                break
        
        # Ensure all routes end at depot
        for b in range(batch_size):
            if len(routes[b]) == 0 or routes[b][-1] != 0:
                routes[b].append(0)
        
        # Combine log probabilities
        if all_log_probs:
            combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1)  # [B]
        else:
            combined_log_probs = torch.zeros(batch_size)
        
        return routes, combined_log_probs

def compute_route_cost(route, distances):
    """Compute total cost of a route"""
    if len(route) <= 1:
        return 0.0
    
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]]
    return cost

def train_epoch(model, instances, optimizer, config, logger):
    """Train for one epoch"""
    model.train()
    batch_size = config['batch_size']
    num_batches = len(instances) // batch_size
    
    epoch_losses = []
    epoch_costs = []
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        batch_instances = instances[batch_start:batch_end]
        
        optimizer.zero_grad()
        
        # Forward pass
        routes, log_probs = model(batch_instances, temperature=config['temperature'])
        
        # Compute costs
        costs = []
        for route, instance in zip(routes, batch_instances):
            cost = compute_route_cost(route, instance['distances'])
            costs.append(cost)
        
        costs_tensor = torch.tensor(costs, dtype=torch.float32)
        
        # REINFORCE loss
        baseline = costs_tensor.mean().detach()
        advantages = costs_tensor - baseline
        
        # Policy loss
        loss = (-advantages * log_probs).mean()
        
        # Backward pass
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
        
        epoch_losses.append(loss.item())
        epoch_costs.extend(costs)
    
    return np.mean(epoch_losses), np.mean(epoch_costs)

def validate_model(model, instances, config, logger):
    """Validate model"""
    model.eval()
    
    all_costs = []
    batch_size = config['batch_size']
    
    with torch.no_grad():
        for i in range(0, len(instances), batch_size):
            batch_instances = instances[i:i + batch_size]
            routes, _ = model(batch_instances, greedy=True)
            
            for route, instance in zip(routes, batch_instances):
                cost = compute_route_cost(route, instance['distances'])
                all_costs.append(cost)
    
    return np.mean(all_costs), np.std(all_costs)

def main():
    logger = setup_logging()
    logger.info("🚀 Starting Simple CVRP Training")
    
    set_seeds(42)
    
    config = {
        'num_customers': 10,
        'capacity': 20,
        'coord_range': 50,
        'demand_range': (1, 5),
        'hidden_dim': 64,
        'num_heads': 4,
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 1e-3,
        'grad_clip': 1.0,
        'temperature': 1.0,
        'num_train_instances': 1000,
        'num_val_instances': 200
    }
    
    logger.info(f"📋 Config: {config['num_customers']} customers, {config['num_epochs']} epochs")
    
    # Generate training data
    logger.info("🔄 Generating training instances...")
    train_instances = []
    for i in tqdm(range(config['num_train_instances'])):
        instance = generate_cvrp_instance(
            config['num_customers'], config['capacity'], 
            config['coord_range'], config['demand_range'], seed=i
        )
        train_instances.append(instance)
    
    # Generate validation data
    logger.info("🔄 Generating validation instances...")
    val_instances = []
    for i in range(config['num_val_instances']):
        instance = generate_cvrp_instance(
            config['num_customers'], config['capacity'],
            config['coord_range'], config['demand_range'], seed=10000 + i
        )
        val_instances.append(instance)
    
    # Create model
    model = SimpleCVRPSolver(config['hidden_dim'], config['num_heads'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Model parameters: {num_params:,}")
    
    # Training loop
    train_losses = []
    train_costs = []
    val_costs = []
    best_val_cost = float('inf')
    
    logger.info("🏋️ Starting training...")
    for epoch in range(config['num_epochs']):
        # Shuffle training data
        np.random.shuffle(train_instances)
        
        # Train
        epoch_loss, epoch_cost = train_epoch(model, train_instances, optimizer, config, logger)
        train_losses.append(epoch_loss)
        train_costs.append(epoch_cost)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_cost, val_std = validate_model(model, val_instances, config, logger)
            val_costs.append(val_cost)
            
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                torch.save(model.state_dict(), 'best_simple_model.pt')
            
            logger.info(f"Epoch {epoch:2d}: Loss={epoch_loss:.4f}, Train={epoch_cost:.2f}, Val={val_cost:.2f}±{val_std:.2f}")
        else:
            logger.info(f"Epoch {epoch:2d}: Loss={epoch_loss:.4f}, Train={epoch_cost:.2f}")
    
    # Final validation
    final_val_cost, final_val_std = validate_model(model, val_instances, config, logger)
    
    # Test on sample instances
    logger.info("📝 Sample solutions:")
    test_instances = val_instances[:3]
    with torch.no_grad():
        routes, _ = model(test_instances, greedy=True)
        for i, (route, instance) in enumerate(zip(routes, test_instances)):
            cost = compute_route_cost(route, instance['distances'])
            logger.info(f"   Instance {i+1}: {route} -> Cost {cost:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_costs, label='Train')
    if val_costs:
        val_epochs = list(range(0, len(train_costs), 5))[:len(val_costs)]
        plt.plot(val_epochs, val_costs, 'ro-', label='Validation')
    plt.title('Route Costs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_training_results.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Results saved to simple_training_results.png")
    
    # Summary
    logger.info(f"📈 Final Results:")
    logger.info(f"   Best validation cost: {best_val_cost:.2f}")
    logger.info(f"   Final validation cost: {final_val_cost:.2f} ± {final_val_std:.2f}")
    logger.info(f"   Model parameters: {num_params:,}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_cost': best_val_cost,
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs
    }, 'simple_final_model.pt')
    
    logger.info("✅ Training completed successfully!")
    return best_val_cost

if __name__ == "__main__":
    best_cost = main()
    print(f"\n🎉 Training completed with best validation cost: {best_cost:.2f}")
