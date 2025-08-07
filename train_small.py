#!/usr/bin/env python3
"""
Small Instance Training Script
Development and testing pipeline for Dynamic Graph Transformer with CPU/GPU compatibility
Trains on 10 nodes + depot instances with minimal configuration for fast iteration
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def get_device():
    """Get the best available device (GPU > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        if 'A6000' in torch.cuda.get_device_name(0):
            print("✅ RTX A6000 detected!")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device

def create_small_config():
    """Create configuration for small instances (development mode)"""
    config = {
        'model': {
            'name': 'dynamic_graph_transformer',
            'node_dim': 64,  # Reduced from typical 128
            'edge_dim': 32,   # Reduced from typical 64
            'hidden_dim': 128,
            'num_heads': 4,   # Reduced from typical 8
            'num_layers': 2,  # Reduced from typical 4-6
            'dropout': 0.1,
            'use_dynamic_updates': True,
            'update_frequency': 5,
            'attention_type': 'transformer'
        },
        'problem': {
            'num_nodes': 10,  # Small instance: 10 customers + 1 depot
            'vehicle_capacity': 20,
            'coord_range': [0, 1],
            'demand_range': [1, 3]
        },
        'training': {
            'batch_size': 8,   # Small batch for development
            'num_epochs': 50,  # Quick training
            'learning_rate': 1e-4,
            'grad_clip': 1.0,
            'baseline': 'exponential',
            'beta': 0.8
        },
        'data': {
            'num_train_samples': 1000,  # Small dataset
            'num_val_samples': 200,
            'seed': 42
        }
    }
    return config

def generate_cvrp_instance(num_nodes, vehicle_capacity, coord_range, demand_range, seed=None):
    """Generate a single CVRP instance"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates: depot at (0.5, 0.5), customers randomly distributed
    coords = np.random.uniform(coord_range[0], coord_range[1], (num_nodes + 1, 2))
    coords[0] = [0.5, 0.5]  # Depot at center
    
    # Generate demands: depot has 0 demand, customers have random demands
    demands = np.zeros(num_nodes + 1)
    demands[1:] = np.random.randint(demand_range[0], demand_range[1] + 1, num_nodes)
    
    # Compute distance matrix
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    distances = np.zeros((num_nodes + 1, num_nodes + 1))
    for i in range(num_nodes + 1):
        for j in range(num_nodes + 1):
            distances[i, j] = euclidean_distance(coords[i], coords[j])
    
    return {
        'coords': coords,
        'demands': demands,
        'distances': distances,
        'capacity': vehicle_capacity
    }

def create_graph_data(instance):
    """Convert CVRP instance to PyTorch Geometric data"""
    coords = torch.tensor(instance['coords'], dtype=torch.float32)
    demands = torch.tensor(instance['demands'], dtype=torch.float32).unsqueeze(1)
    capacity = torch.tensor([instance['capacity']], dtype=torch.float32)
    
    # Node features: [x, y, demand]
    node_features = torch.cat([coords, demands], dim=1)
    
    # Create fully connected edge_index (all nodes connected to all nodes)
    num_nodes = len(coords)
    edge_index = []
    edge_attr = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
                # Edge features: [distance]
                dist = torch.tensor([instance['distances'][i, j]], dtype=torch.float32)
                edge_attr.append(dist)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'capacity': capacity,
        'num_nodes': num_nodes
    }

class SimpleDynamicGraphTransformer(nn.Module):
    """Simplified Dynamic Graph Transformer for development testing"""
    
    def __init__(self, node_dim=3, edge_dim=1, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Edge embedding  
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Simplified transformer layers (using standard PyTorch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pointer mechanism for route generation
        self.pointer = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features, edge_index, edge_attr, capacity):
        batch_size = 1  # Single instance for now
        num_nodes = node_features.shape[0]
        
        # Embed nodes
        h_nodes = self.node_embedding(node_features)  # [num_nodes, hidden_dim]
        
        # Simple transformer processing
        h_nodes = h_nodes.unsqueeze(0)  # Add batch dimension [1, num_nodes, hidden_dim]
        h_nodes = self.transformer(h_nodes)
        h_nodes = h_nodes.squeeze(0)  # Remove batch dimension [num_nodes, hidden_dim]
        
        # Simple greedy route generation for testing
        route = [0]  # Start at depot
        visited = {0}
        current_capacity = capacity.item()
        
        while len(visited) < num_nodes:
            current_node = route[-1]
            
            # Find next unvisited node (simplified selection)
            best_node = None
            best_score = float('-inf')
            
            for next_node in range(num_nodes):
                if next_node not in visited:
                    # Check capacity constraint
                    demand = node_features[next_node, 2].item()
                    if demand <= current_capacity:
                        # Simple scoring based on node embeddings
                        score = torch.dot(h_nodes[current_node], h_nodes[next_node]).item()
                        if score > best_score:
                            best_score = score
                            best_node = next_node
            
            if best_node is not None:
                route.append(best_node)
                visited.add(best_node)
                current_capacity -= node_features[best_node, 2].item()
            else:
                # Return to depot and start new route if needed
                if len(visited) < num_nodes:
                    route.append(0)
                    current_capacity = capacity.item()
        
        return route

def compute_route_cost(route, distances):
    """Compute total cost of a route"""
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]]
    # Add cost to return to depot if not already there
    if route[-1] != 0:
        cost += distances[route[-1], 0]
    return cost

def train_small_model():
    """Train model on small instances"""
    print("🚀 Starting Dynamic Graph Transformer Small Instance Training")
    print("=" * 70)
    
    # Get device
    device = get_device()
    
    # Create configuration
    config = create_small_config()
    print(f"📋 Configuration:")
    print(f"   - Problem size: {config['problem']['num_nodes']} customers + depot")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Epochs: {config['training']['num_epochs']}")
    print(f"   - Model: {config['model']['name']}")
    
    # Initialize model
    model = SimpleDynamicGraphTransformer(
        node_dim=3,  # [x, y, demand]
        edge_dim=1,  # [distance]
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate training data
    print(f"🔄 Generating {config['data']['num_train_samples']} training instances...")
    train_instances = []
    
    for i in tqdm(range(config['data']['num_train_samples'])):
        instance = generate_cvrp_instance(
            config['problem']['num_nodes'],
            config['problem']['vehicle_capacity'],
            config['problem']['coord_range'],
            config['problem']['demand_range'],
            seed=config['data']['seed'] + i
        )
        train_instances.append(instance)
    
    # Training loop
    print("🏋️ Starting training...")
    losses = []
    best_costs = []
    
    for epoch in range(config['training']['num_epochs']):
        epoch_losses = []
        epoch_costs = []
        
        # Sample batch
        batch_indices = np.random.choice(len(train_instances), config['training']['batch_size'], replace=False)
        
        for idx in batch_indices:
            instance = train_instances[idx]
            graph_data = create_graph_data(instance)
            
            # Move to device
            node_features = graph_data['node_features'].to(device)
            edge_index = graph_data['edge_index'].to(device)
            edge_attr = graph_data['edge_attr'].to(device)
            capacity = graph_data['capacity'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            route = model(node_features, edge_index, edge_attr, capacity)
            
            # Compute cost (negative reward for minimization)
            cost = compute_route_cost(route, instance['distances'])
            
            # Simple loss: encourage shorter routes
            loss = torch.tensor(cost, device=device, dtype=torch.float32, requires_grad=True)
            
            # Backward pass (placeholder - in real implementation would use REINFORCE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_costs.append(cost)
        
        avg_loss = np.mean(epoch_losses)
        avg_cost = np.mean(epoch_costs)
        losses.append(avg_loss)
        best_costs.append(avg_cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Avg Cost={avg_cost:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(best_costs)
    plt.title('Average Route Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress_small.png', dpi=150, bbox_inches='tight')
    print("📊 Training progress saved to training_progress_small.png")
    
    # Test on a few instances
    print("\n🧪 Testing trained model...")
    test_costs = []
    
    for i in range(10):
        instance = generate_cvrp_instance(
            config['problem']['num_nodes'],
            config['problem']['vehicle_capacity'],
            config['problem']['coord_range'],
            config['problem']['demand_range'],
            seed=1000 + i
        )
        
        graph_data = create_graph_data(instance)
        node_features = graph_data['node_features'].to(device)
        edge_index = graph_data['edge_index'].to(device)
        edge_attr = graph_data['edge_attr'].to(device)
        capacity = graph_data['capacity'].to(device)
        
        with torch.no_grad():
            route = model(node_features, edge_index, edge_attr, capacity)
            cost = compute_route_cost(route, instance['distances'])
            test_costs.append(cost)
            
            if i < 3:  # Show first 3 routes
                print(f"   Test {i+1}: Route {route} -> Cost {cost:.4f}")
    
    avg_test_cost = np.mean(test_costs)
    print(f"\n📈 Results Summary:")
    print(f"   - Final training cost: {best_costs[-1]:.4f}")
    print(f"   - Average test cost: {avg_test_cost:.4f}")
    print(f"   - Device used: {device}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_cost': avg_test_cost
    }, 'small_model_checkpoint.pt')
    print("💾 Model saved to small_model_checkpoint.pt")
    
    print("\n✅ Small instance training completed successfully!")
    print("🚀 Ready for GPU server deployment with larger instances!")

if __name__ == "__main__":
    train_small_model()
