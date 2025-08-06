"""
Ablation Study Models for Dynamic Graph Transformer CVRP
This module implements three model variants for comprehensive comparison:
0. Baseline: Greedy Graph Transformer (no RL, no dynamic updates)
1. Static RL: Graph Transformer + RL (no dynamic updates)  
2. Dynamic RL: Graph Transformer + RL + Dynamic Updates (full pipeline)
"""

import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any
import torch.nn.functional as F

from .graph_transformer import GraphTransformerEncoder
from .dynamic_updater import DynamicGraphUpdater, RoutingStateTracker
from .GAT_Decoder import GAT_Decoder


class GreedyGraphTransformerBaseline(nn.Module):
    """
    Variant 0: Baseline Greedy Graph Transformer
    - Graph Transformer encoder only
    - Greedy nearest-neighbor style decoding
    - No RL training, no dynamic updates
    - Pure architectural comparison vs GAT
    """
    
    def __init__(self,
                 node_input_dim: int = 3,
                 edge_input_dim: int = 1,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 pe_type: str = "sinusoidal",
                 pe_dim: int = 64,
                 max_distance: float = 100.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Graph Transformer Encoder
        self.encoder = GraphTransformerEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pe_type=pe_type,
            pe_dim=pe_dim,
            max_distance=max_distance,
            use_edge_weights=True
        )
        
        # Simple output projection for similarity computation
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, data, n_steps: int, greedy: bool = True, T: float = 1.0):
        """
        Forward pass with greedy decoding
        
        Args:
            data: PyTorch Geometric data object
            n_steps: Number of routing steps
            greedy: Always True for baseline (ignored)
            T: Temperature (ignored for greedy)
            
        Returns:
            actions: Selected routes
            log_p: Log probabilities (zeros for greedy)
        """
        device = data.x.device
        batch_size = data.num_graphs
        total_nodes = data.x.size(0)
        num_nodes = total_nodes // batch_size
        
        # Encode with Graph Transformer
        x = self.encoder(data)  # [batch_size, num_nodes, hidden_dim]
        
        # Project embeddings for similarity computation
        x_proj = self.output_projection(x)  # [batch_size, num_nodes, hidden_dim]
        
        # Extract problem data
        coordinates = data.x[:, :2].view(batch_size, num_nodes, 2)
        demands = data.demand.view(batch_size, num_nodes)
        capacity = data.capacity.view(batch_size, num_nodes)[:, 0]  # Vehicle capacity
        
        # Greedy routing
        actions, log_probs = self._greedy_routing(
            x_proj, coordinates, demands, capacity, n_steps, device
        )
        
        return actions, log_probs
    
    def _greedy_routing(self, embeddings, coordinates, demands, capacity, n_steps, device):
        """
        Greedy routing based on transformer embeddings + distance
        """
        batch_size, num_nodes, hidden_dim = embeddings.shape
        
        # Storage for actions and dummy log probabilities
        actions = []
        log_probs = []
        
        # Initialize routing state
        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)  # Start at depot
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        remaining_capacity = capacity.clone()
        
        for step in range(n_steps):
            # Current node embeddings
            current_embeddings = embeddings[torch.arange(batch_size), current_node]  # [batch_size, hidden_dim]
            
            # Compute similarities to all nodes
            similarities = torch.bmm(
                current_embeddings.unsqueeze(1),  # [batch_size, 1, hidden_dim]
                embeddings.transpose(1, 2)        # [batch_size, hidden_dim, num_nodes]
            ).squeeze(1)  # [batch_size, num_nodes]
            
            # Add distance penalty (encourage visiting nearby nodes)
            current_coords = coordinates[torch.arange(batch_size), current_node]  # [batch_size, 2]
            distances = torch.norm(
                coordinates - current_coords.unsqueeze(1), dim=2
            )  # [batch_size, num_nodes]
            
            # Combined score: similarity - distance_penalty
            scores = similarities - 0.1 * distances
            
            # Apply constraints
            # Cannot visit already visited nodes (except depot)
            visited_mask = visited.clone()
            visited_mask[:, 0] = False  # Always allow depot
            scores.masked_fill_(visited_mask, -float('inf'))
            
            # Cannot visit nodes that exceed capacity (except depot)
            capacity_mask = demands > remaining_capacity.unsqueeze(1)
            capacity_mask[:, 0] = False  # Always allow depot
            scores.masked_fill_(capacity_mask, -float('inf'))
            
            # Select next node (greedy)
            next_node = torch.argmax(scores, dim=1)
            
            # Update state
            actions.append(next_node.unsqueeze(-1))
            log_probs.append(torch.zeros(batch_size, 1, device=device))  # Dummy log probs
            
            # Update routing state
            visited[torch.arange(batch_size), next_node] = True
            
            # Update capacity (reset if returning to depot)
            is_depot = (next_node == 0)
            remaining_capacity = torch.where(
                is_depot,
                capacity,
                remaining_capacity - demands[torch.arange(batch_size), next_node]
            )
            
            current_node = next_node
            
            # Early stopping if all customers visited
            if visited[:, 1:].all(dim=1).all():
                break
        
        # Stack results
        actions = torch.stack(actions, dim=1)  # [batch_size, n_steps, 1]
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, n_steps, 1]
        
        return actions, log_probs


class StaticRLGraphTransformer(nn.Module):
    """
    Variant 1: Graph Transformer + RL (Static)
    - Graph Transformer encoder
    - RL-trained decoder (GAT_Decoder)
    - No dynamic graph updates during routing
    - Shows benefit of RL training over greedy
    """
    
    def __init__(self,
                 node_input_dim: int = 3,
                 edge_input_dim: int = 1,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 pe_type: str = "sinusoidal",
                 pe_dim: int = 64,
                 max_distance: float = 100.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Graph Transformer Encoder (same as baseline)
        self.encoder = GraphTransformerEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pe_type=pe_type,
            pe_dim=pe_dim,
            max_distance=max_distance,
            use_edge_weights=True
        )
        
        # RL-trainable decoder (same as original)
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)
        
    def forward(self, data, n_steps: int, greedy: bool = False, T: float = 1.0):
        """
        Forward pass with RL decoder (no dynamic updates)
        
        Args:
            data: PyTorch Geometric data object
            n_steps: Number of routing steps
            greedy: Whether to use greedy decoding
            T: Temperature for sampling
            
        Returns:
            actions: Selected routes
            log_p: Log probabilities from RL policy
        """
        device = data.x.device
        batch_size = data.num_graphs
        total_nodes = data.x.size(0)
        num_nodes = total_nodes // batch_size
        
        # Encode with Graph Transformer
        x = self.encoder(data)  # [batch_size, num_nodes, hidden_dim]
        
        # Compute graph embedding
        graph_embedding = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Flatten for decoder compatibility
        x_flat = x.view(batch_size * num_nodes, self.hidden_dim)
        
        # Prepare data for decoder
        demand = data.demand.reshape(batch_size, -1).float().to(device)
        capacity = data.capacity.reshape(batch_size, -1).float().to(device)
        
        # RL decoder (static - no dynamic updates)
        actions, log_p = self.decoder(x_flat, graph_embedding, capacity, demand, n_steps, T, greedy)
        
        return actions, log_p


class DynamicRLGraphTransformer(nn.Module):
    """
    Variant 2: Full Dynamic Pipeline
    - Graph Transformer encoder
    - RL-trained decoder
    - Dynamic graph updates during routing
    - Complete system with all innovations
    """
    
    def __init__(self,
                 node_input_dim: int = 3,
                 edge_input_dim: int = 1,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 pe_type: str = "sinusoidal",
                 pe_dim: int = 64,
                 max_distance: float = 100.0,
                 update_frequency: int = 1,
                 vehicle_capacity: float = 50.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.update_frequency = update_frequency
        self.vehicle_capacity = vehicle_capacity
        
        # Graph Transformer Encoder (same as others)
        self.encoder = GraphTransformerEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pe_type=pe_type,
            pe_dim=pe_dim,
            max_distance=max_distance,
            use_edge_weights=True
        )
        
        # Dynamic Graph Updater
        self.dynamic_updater = DynamicGraphUpdater(
            hidden_dim=hidden_dim,
            update_node_features=True,
            update_edge_weights=True,
            adaptive_masking=True,
            capacity_aware=True
        )
        
        # RL-trainable decoder
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)
        
    def forward(self, data, n_steps: int, greedy: bool = False, T: float = 1.0):
        """
        Forward pass with dynamic graph updates during routing
        
        Args:
            data: PyTorch Geometric data object
            n_steps: Number of routing steps
            greedy: Whether to use greedy decoding
            T: Temperature for sampling
            
        Returns:
            actions: Selected routes
            log_p: Log probabilities from RL policy
        """
        device = data.x.device
        batch_size = data.num_graphs
        total_nodes = data.x.size(0)
        num_nodes = total_nodes // batch_size
        
        # Initial encoding
        x = self.encoder(data)  # [batch_size, num_nodes, hidden_dim]
        
        # For dynamic routing, use step-by-step approach
        return self._dynamic_forward(data, x, n_steps, T, greedy)
    
    def _dynamic_forward(self, data, x, n_steps, T, greedy):
        """
        Forward pass with dynamic updates at each step
        """
        device = data.x.device
        batch_size, num_nodes, _ = x.shape
        
        # Initialize routing state tracker
        state_tracker = RoutingStateTracker(
            batch_size=batch_size,
            num_nodes=num_nodes,
            vehicle_capacity=self.vehicle_capacity,
            device=device
        )
        
        # Extract problem data
        coordinates = data.x[:, :2].view(batch_size, num_nodes, 2)
        demands = data.demand.view(batch_size, num_nodes)
        capacity = data.capacity.reshape(batch_size, -1).float().to(device)
        demand_formatted = data.demand.reshape(batch_size, -1).float().to(device)
        
        # Storage for results
        all_actions = []
        all_log_probs = []
        
        # Initial edge weights
        edge_weights = self._compute_edge_weights(coordinates)
        
        # Dynamic routing loop
        for step in range(n_steps):
            # Update graph dynamically
            if step % self.update_frequency == 0:
                routing_state = state_tracker.get_state_dict(coordinates, demands)
                x, edge_weights, attention_mask = self.dynamic_updater(
                    node_embeddings=x,
                    edge_weights=edge_weights,
                    routing_state=routing_state,
                    step=step
                )
            
            # Compute graph embedding
            graph_embedding = x.mean(dim=1)
            
            # Single step prediction
            x_flat = x.view(batch_size * num_nodes, self.hidden_dim)
            
            # Get feasible nodes
            feasible_mask = state_tracker.get_feasible_nodes(demands)
            
            # Single step decode (would need decoder modification for proper masking)
            actions, log_p = self.decoder(x_flat, graph_embedding, capacity, demand_formatted, 1, T, greedy)
            
            # Apply feasibility constraints post-hoc (temporary solution)
            actions = self._apply_feasibility_constraints(actions, feasible_mask)
            
            all_actions.append(actions)
            all_log_probs.append(log_p)
            
            # Update routing state
            if step < n_steps - 1:
                state_tracker.update(actions.squeeze(-1), coordinates, demands)
                
                if state_tracker.is_done().all():
                    break
        
        # Combine results
        final_actions = torch.stack(all_actions, dim=1)
        final_log_probs = torch.stack(all_log_probs, dim=1)
        
        return final_actions, final_log_probs
    
    def _compute_edge_weights(self, coordinates):
        """Compute initial edge weights from coordinates"""
        batch_size, num_nodes, _ = coordinates.shape
        device = coordinates.device
        
        edge_weights = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        
        for b in range(batch_size):
            coords = coordinates[b]
            coord_i = coords.unsqueeze(1).expand(-1, num_nodes, -1)
            coord_j = coords.unsqueeze(0).expand(num_nodes, -1, -1)
            distances = torch.norm(coord_i - coord_j, dim=-1)
            edge_weights[b] = distances
        
        return edge_weights
    
    def _apply_feasibility_constraints(self, actions, feasible_mask):
        """Apply feasibility constraints to decoder output (temporary solution)"""
        # This is a simple post-processing step
        # In practice, the decoder should be modified to handle masking properly
        batch_size = actions.shape[0]
        
        for b in range(batch_size):
            action = actions[b, 0, 0]  # Current action for this batch
            if not feasible_mask[b, action]:
                # Find nearest feasible node
                feasible_nodes = torch.where(feasible_mask[b])[0]
                if len(feasible_nodes) > 0:
                    actions[b, 0, 0] = feasible_nodes[0]
        
        return actions


# Factory function for creating ablation models
def create_ablation_model(variant: str, config: Dict[str, Any]) -> nn.Module:
    """
    Create ablation study model variants
    
    Args:
        variant: '0_baseline', '1_static_rl', or '2_dynamic_rl'
        config: Model configuration
        
    Returns:
        Model instance
    """
    encoder_config = config.get('encoder', {})
    pe_config = config.get('positional_encoding', {})
    dynamic_config = config.get('dynamic_updates', {})
    problem_config = config.get('problem', {})
    
    common_args = {
        'node_input_dim': encoder_config.get('node_input_dim', 3),
        'edge_input_dim': encoder_config.get('edge_input_dim', 1),
        'hidden_dim': encoder_config.get('hidden_dim', 128),
        'num_heads': encoder_config.get('num_heads', 8),
        'num_layers': encoder_config.get('num_layers', 6),
        'dropout': encoder_config.get('dropout', 0.1),
        'pe_type': pe_config.get('pe_type', 'sinusoidal'),
        'pe_dim': pe_config.get('pe_dim', 64),
        'max_distance': pe_config.get('max_distance', 100.0),
    }
    
    if variant == '0_baseline':
        return GreedyGraphTransformerBaseline(**common_args)
    elif variant == '1_static_rl':
        return StaticRLGraphTransformer(**common_args)
    elif variant == '2_dynamic_rl':
        return DynamicRLGraphTransformer(
            **common_args,
            update_frequency=dynamic_config.get('update_frequency', 1),
            vehicle_capacity=problem_config.get('vehicle_capacity', 50.0)
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test all three variants
    import torch
    from torch_geometric.data import Data
    
    # Create test data
    batch_size = 2
    num_nodes = 10
    
    coordinates = torch.rand(batch_size * num_nodes, 2) * 100
    demands = torch.rand(batch_size * num_nodes, 1) * 10
    
    # Simple edge structure
    edge_indices = []
    edge_attrs = []
    
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    source = b * num_nodes + i
                    target = b * num_nodes + j
                    edge_indices.append([source, target])
                    
                    coord_i = coordinates[source]
                    coord_j = coordinates[target]
                    distance = torch.norm(coord_i - coord_j, dim=0, keepdim=True)
                    edge_attrs.append(distance)
    
    edge_index = torch.tensor(edge_indices).t().long()
    edge_attr = torch.stack(edge_attrs)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    data = Data(
        x=coordinates,
        edge_index=edge_index,
        edge_attr=edge_attr,
        demand=demands,
        capacity=torch.full((batch_size * num_nodes,), 50.0),
        batch=batch
    )
    data.num_graphs = batch_size
    
    config = {
        'encoder': {'hidden_dim': 128, 'num_heads': 8, 'num_layers': 3},
        'positional_encoding': {'pe_type': 'sinusoidal'},
        'dynamic_updates': {'update_frequency': 1},
        'problem': {'vehicle_capacity': 50.0}
    }
    
    print("Testing Ablation Study Models...")
    
    variants = ['0_baseline', '1_static_rl', '2_dynamic_rl']
    names = ['Greedy Baseline', 'Static RL', 'Dynamic RL']
    
    for variant, name in zip(variants, names):
        print(f"\n=== {name} ===")
        model = create_ablation_model(variant, config)
        
        # Forward pass
        actions, log_probs = model(data, n_steps=5, greedy=True)
        
        print(f"Actions shape: {actions.shape}")
        print(f"Log probs shape: {log_probs.shape}")
        print(f"Actions sample: {actions[0, :3, 0].tolist()}")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    print("\nAll ablation models tested successfully!")
