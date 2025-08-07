#!/usr/bin/env python3
"""
Small Configuration Training Script
Uses the existing project architecture for a comprehensive small-scale test
Designed for development and validation on Mac with A6000 compatibility
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
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import project modules
try:
    from models.DynamicGraphTransformerModel import DynamicGraphTransformerModel
    from models.model_factory import create_model
    from models.graph_transformer import GraphTransformerEncoder
    from models.GAT_Decoder import GAT_Decoder
    from training.train_model import baseline
    from utils.RL.euclidean_cost_eval import euclidean_cost_eval
except ImportError as e:
    print(f"⚠️  Warning: Could not import project modules: {e}")
    print("   Will use simplified implementation instead")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def get_device():
    """Get the best available device"""
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
    """Create small configuration based on default_config.yaml"""
    config = {
        'model': {
            'type': 'dynamic_transformer',
            'encoder': {
                'hidden_dim': 64,        # Reduced from 128
                'num_heads': 4,          # Reduced from 8
                'num_layers': 3,         # Reduced from 6
                'dropout': 0.1,
                'use_layer_norm': True,
                'use_residual': True,
                'activation': 'relu'
            },
            'positional_encoding': {
                'use_pe': True,
                'pe_type': 'sinusoidal',
                'max_distance': 100.0,
                'pe_dim': 32             # Reduced from 64
            },
            'dynamic_updates': {
                'enabled': True,
                'update_frequency': 2,   # Update every 2 steps
                'update_node_features': True,
                'update_edge_weights': True,
                'adaptive_masking': True
            },
            'decoder': {
                'hidden_dim': 64,
                'num_heads': 4,
                'num_layers': 2,         # Reduced from 3
                'use_pointer_networks': True,
                'attention_type': 'scaled_dot_product',
                'temperature': 1.0
            }
        },
        'problem': {
            'name': 'cvrp',
            'max_nodes': 15,             # Small: 15 customers + depot
            'min_nodes': 15,
            'vehicle_capacity': 30,      # Adjusted for small instances
            'depot_location': 'center',
            'node_distribution': 'uniform',
            'demand_range': [1, 5],      # Smaller demands
            'coordinate_range': [0, 50]  # Smaller coordinate space
        },
        'training': {
            'batch_size': 16,            # Small batch for development
            'num_epochs': 30,            # Quick training
            'learning_rate': 1e-3,       # Slightly higher for faster convergence
            'weight_decay': 1e-5,
            'gradient_clip_norm': 1.0,
            'rl': {
                'algorithm': 'reinforce',
                'baseline': 'exponential',
                'entropy_weight': 0.01,
                'gamma': 0.99
            },
            'validation': {
                'frequency': 5,
                'num_instances': 100
            }
        },
        'data': {
            'train': {
                'num_instances': 2000,   # Small dataset for development
                'data_dir': 'data/small_train'
            },
            'validation': {
                'num_instances': 200,
                'data_dir': 'data/small_val'
            }
        },
        'hardware': {
            'device': 'auto',
            'num_threads': 2
        },
        'seed': {
            'main': 42,
            'data_generation': 123,
            'training': 456
        }
    }
    return config

def set_seeds(config):
    """Set random seeds for reproducibility"""
    torch.manual_seed(config['seed']['main'])
    np.random.seed(config['seed']['main'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed']['main'])
        torch.cuda.manual_seed_all(config['seed']['main'])

def generate_cvrp_instance(config, seed=None):
    """Generate a single CVRP instance based on config"""
    if seed is not None:
        np.random.seed(seed)
    
    problem_config = config['problem']
    num_customers = problem_config['max_nodes']
    capacity = problem_config['vehicle_capacity']
    coord_range = problem_config['coordinate_range']
    demand_range = problem_config['demand_range']
    
    # Generate coordinates
    coords = np.random.uniform(coord_range[0], coord_range[1], (num_customers + 1, 2))
    if problem_config['depot_location'] == 'center':
        coords[0] = [coord_range[1]/2, coord_range[1]/2]
    
    # Generate demands (depot has 0 demand)
    demands = np.zeros(num_customers + 1)
    demands[1:] = np.random.randint(demand_range[0], demand_range[1] + 1, num_customers)
    
    # Compute distance matrix
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    distances = np.zeros((num_customers + 1, num_customers + 1))
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            distances[i, j] = euclidean_distance(coords[i], coords[j])
    
    return {
        'coords': coords,
        'demands': demands,
        'distances': distances,
        'capacity': capacity,
        'num_nodes': num_customers + 1
    }

def create_pytorch_geometric_data(instances, device):
    """Convert instances to PyTorch Geometric batch format"""
    from torch_geometric.data import Data, Batch
    
    data_list = []
    for instance in instances:
        coords = torch.tensor(instance['coords'], dtype=torch.float32)
        demands = torch.tensor(instance['demands'], dtype=torch.float32).unsqueeze(1)
        capacity = torch.tensor([instance['capacity']], dtype=torch.float32)
        
        # Node features: [x, y, demand]
        x = torch.cat([coords, demands], dim=1)
        
        # Create fully connected edge_index
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
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            demand=demands.squeeze(),
            capacity=capacity
        )
        data_list.append(data)
    
    # Create batch
    batch = Batch.from_data_list(data_list)
    return batch.to(device)

class SimplifiedDynamicModel(nn.Module):
    """Simplified model for testing when full implementation is not available"""
    
    def __init__(self, config):
        super().__init__()
        encoder_config = config['model']['encoder']
        self.hidden_dim = encoder_config['hidden_dim']
        
        # Simple transformer encoder
        self.node_embedding = nn.Linear(3, self.hidden_dim)  # x, y, demand
        self.edge_embedding = nn.Linear(1, self.hidden_dim)  # distance
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=encoder_config['num_heads'],
            dim_feedforward=self.hidden_dim * 2,
            dropout=encoder_config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, encoder_config['num_layers'])
        
        # Simple pointer-based decoder
        self.pointer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, data, n_steps, greedy=False, T=1.0):
        """Forward pass with simplified routing"""
        batch_size = data.num_graphs
        total_nodes = data.x.size(0)
        num_nodes = total_nodes // batch_size
        
        # Embed nodes
        x = self.node_embedding(data.x)
        x = x.view(batch_size, num_nodes, self.hidden_dim)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Simple greedy decoding for testing
        routes = []
        log_probs = []
        
        for b in range(batch_size):
            route = [0]  # Start at depot
            visited = {0}
            current_capacity = data.capacity[b].item()
            log_prob = torch.tensor(0.0, device=data.x.device, dtype=torch.float32)
            
            while len(visited) < num_nodes and len(route) < n_steps:
                current_node = route[-1]
                
                # Find feasible next nodes
                feasible_nodes = []
                for next_node in range(num_nodes):
                    if next_node not in visited:
                        demand = data.demand[b * num_nodes + next_node].item()
                        if demand <= current_capacity:
                            feasible_nodes.append(next_node)
                
                if not feasible_nodes:
                    # Return to depot if no feasible nodes
                    if route[-1] != 0:
                        route.append(0)
                    break
                
                # Simple scoring based on embeddings and distance
                scores = []
                for next_node in feasible_nodes:
                    score = torch.dot(x[b, current_node], x[b, next_node]).item()
                    scores.append(score)
                
                # Select next node (greedy or probabilistic)
                if greedy or len(feasible_nodes) == 1:
                    best_idx = np.argmax(scores)
                else:
                    # Probabilistic selection
                    scores = np.array(scores) / T
                    probs = np.exp(scores) / np.sum(np.exp(scores))
                    best_idx = np.random.choice(len(feasible_nodes), p=probs)
                    log_prob += torch.log(torch.tensor(probs[best_idx], device=data.x.device, dtype=torch.float32))
                
                next_node = feasible_nodes[best_idx]
                route.append(next_node)
                visited.add(next_node)
                current_capacity -= data.demand[b * num_nodes + next_node].item()
            
            # Ensure route ends at depot
            if route[-1] != 0:
                route.append(0)
            
            routes.append(route)
            log_probs.append(log_prob)
        
        return routes, torch.stack(log_probs)

def compute_route_costs(routes, instances):
    """Compute costs for all routes in batch"""
    costs = []
    for route, instance in zip(routes, instances):
        cost = 0.0
        for i in range(len(route) - 1):
            cost += instance['distances'][route[i], route[i + 1]]
        costs.append(cost)
    return np.array(costs)

def train_epoch(model, data_loader, optimizer, config, device, logger):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    epoch_costs = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch_idx, (batch_data, instances) in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Forward pass
        n_steps = config['problem']['max_nodes'] * 2  # Allow for multiple routes
        routes, log_probs = model(batch_data, n_steps, greedy=False, T=config['model']['decoder']['temperature'])
        
        # Compute costs (rewards)
        costs = compute_route_costs(routes, instances)
        costs_tensor = torch.tensor(costs, device=device, dtype=torch.float32)
        
        # REINFORCE loss with proper gradient flow
        baseline_costs = costs_tensor.mean().detach()  # Detach baseline from computation graph
        advantages = costs_tensor - baseline_costs
        
        # Policy loss: negative log likelihood weighted by advantages
        # We minimize negative expected reward, so we use negative advantages
        loss = (-advantages.detach() * log_probs).mean()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        optimizer.step()
        
        epoch_losses.append(loss.item())
        epoch_costs.extend(costs)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_cost': f"{np.mean(costs):.2f}"
        })
    
    return np.mean(epoch_losses), np.mean(epoch_costs)

def validate_model(model, instances, config, device, logger):
    """Validate model on test instances"""
    model.eval()
    all_costs = []
    
    # Process instances in small batches
    batch_size = config['training']['batch_size']
    for i in range(0, len(instances), batch_size):
        batch_instances = instances[i:i + batch_size]
        batch_data = create_pytorch_geometric_data(batch_instances, device)
        
        with torch.no_grad():
            n_steps = config['problem']['max_nodes'] * 2
            routes, _ = model(batch_data, n_steps, greedy=True)
            costs = compute_route_costs(routes, batch_instances)
            all_costs.extend(costs)
    
    return np.mean(all_costs), np.std(all_costs)

def create_data_generator(config, device):
    """Create data generator for training and validation"""
    def generate_batch(batch_size, seed_offset=0):
        instances = []
        for i in range(batch_size):
            instance = generate_cvrp_instance(config, seed=config['seed']['data_generation'] + seed_offset + i)
            instances.append(instance)
        
        batch_data = create_pytorch_geometric_data(instances, device)
        return batch_data, instances
    
    return generate_batch

def run_small_configuration():
    """Main training function for small configuration"""
    logger = setup_logging()
    logger.info("🚀 Starting Dynamic Graph Transformer Small Configuration Training")
    
    # Setup
    device = get_device()
    config = create_small_config()
    set_seeds(config)
    
    logger.info("📋 Configuration Summary:")
    logger.info(f"   Problem size: {config['problem']['max_nodes']} customers + depot")
    logger.info(f"   Batch size: {config['training']['batch_size']}")
    logger.info(f"   Epochs: {config['training']['num_epochs']}")
    logger.info(f"   Hidden dim: {config['model']['encoder']['hidden_dim']}")
    logger.info(f"   Dynamic updates: {config['model']['dynamic_updates']['enabled']}")
    
    # Create model
    try:
        # Try to use the full implementation
        model = create_model(config).to(device)
        logger.info("✅ Using full Dynamic Graph Transformer implementation")
    except:
        # Fall back to simplified implementation
        model = SimplifiedDynamicModel(config).to(device)
        logger.info("⚠️  Using simplified model implementation")
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Model parameters: {num_params:,}")
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create data generator
    data_generator = create_data_generator(config, device)
    
    # Generate validation data
    logger.info("🔄 Generating validation instances...")
    val_instances = []
    for i in range(config['data']['validation']['num_instances']):
        instance = generate_cvrp_instance(config, seed=10000 + i)
        val_instances.append(instance)
    
    # Training loop
    logger.info("🏋️ Starting training...")
    train_losses = []
    train_costs = []
    val_costs = []
    best_val_cost = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        # Generate training data for this epoch
        num_batches = config['data']['train']['num_instances'] // config['training']['batch_size']
        data_loader = []
        
        for batch_idx in range(num_batches):
            seed_offset = epoch * num_batches * config['training']['batch_size'] + batch_idx * config['training']['batch_size']
            batch_data, instances = data_generator(config['training']['batch_size'], seed_offset)
            data_loader.append((batch_data, instances))
        
        # Train epoch
        epoch_loss, epoch_cost = train_epoch(model, data_loader, optimizer, config, device, logger)
        train_losses.append(epoch_loss)
        train_costs.append(epoch_cost)
        
        # Validation
        if epoch % config['training']['validation']['frequency'] == 0:
            val_cost, val_std = validate_model(model, val_instances, config, device, logger)
            val_costs.append(val_cost)
            
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_cost': val_cost
                }, 'best_small_model.pt')
            
            logger.info(f"Epoch {epoch:3d}: Loss={epoch_loss:.4f}, Train Cost={epoch_cost:.2f}, Val Cost={val_cost:.2f}±{val_std:.2f}")
        else:
            logger.info(f"Epoch {epoch:3d}: Loss={epoch_loss:.4f}, Train Cost={epoch_cost:.2f}")
    
    # Plot training progress
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Training cost
    axes[0, 1].plot(train_costs)
    axes[0, 1].set_title('Training Cost')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Route Cost')
    axes[0, 1].grid(True)
    
    # Validation cost
    val_epochs = list(range(0, len(train_costs), config['training']['validation']['frequency']))[:len(val_costs)]
    axes[1, 0].plot(val_epochs, val_costs, 'o-', color='red')
    axes[1, 0].set_title('Validation Cost')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Average Route Cost')
    axes[1, 0].grid(True)
    
    # Training vs Validation
    axes[1, 1].plot(train_costs, label='Train', alpha=0.7)
    axes[1, 1].plot(val_epochs, val_costs, 'o-', color='red', label='Validation')
    axes[1, 1].set_title('Training vs Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Average Route Cost')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('small_config_training_progress.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Training progress saved to small_config_training_progress.png")
    
    # Final evaluation
    logger.info("\n🧪 Final evaluation...")
    final_val_cost, final_val_std = validate_model(model, val_instances, config, device, logger)
    
    # Test on a few instances with detailed output
    logger.info("📝 Sample solutions:")
    test_instances = val_instances[:3]
    test_batch = create_pytorch_geometric_data(test_instances, device)
    
    with torch.no_grad():
        routes, _ = model(test_batch, config['problem']['max_nodes'] * 2, greedy=True)
        costs = compute_route_costs(routes, test_instances)
        
        for i, (route, cost, instance) in enumerate(zip(routes, costs, test_instances)):
            logger.info(f"   Instance {i+1}: Route {route[:8]}... -> Cost {cost:.2f}")
    
    # Summary
    logger.info(f"\n📈 Training Summary:")
    logger.info(f"   Final training cost: {train_costs[-1]:.2f}")
    logger.info(f"   Best validation cost: {best_val_cost:.2f}")
    logger.info(f"   Final validation cost: {final_val_cost:.2f} ± {final_val_std:.2f}")
    logger.info(f"   Device used: {device}")
    logger.info(f"   Model parameters: {num_params:,}")
    
    # Save final model and config
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs,
        'final_cost': final_val_cost
    }, 'small_config_final_model.pt')
    
    # Save config
    with open('small_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("💾 Model and config saved")
    logger.info("✅ Small configuration training completed successfully!")
    logger.info("🚀 Ready for scaling up to A6000 GPU server!")
    
    return {
        'final_val_cost': final_val_cost,
        'best_val_cost': best_val_cost,
        'num_parameters': num_params,
        'device': str(device)
    }

if __name__ == "__main__":
    results = run_small_configuration()
    print(f"\n🎉 Training completed with validation cost: {results['final_val_cost']:.2f}")
