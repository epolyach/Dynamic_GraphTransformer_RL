"""
Dynamic Graph Transformer Model for CVRP
This model replaces the GAT encoder with Graph Transformer and integrates dynamic graph updates.
"""

import torch
from torch import nn
from typing import Optional, Tuple

from .graph_transformer import GraphTransformerEncoder
from .dynamic_updater import DynamicGraphUpdater, RoutingStateTracker
from .GAT_Decoder import GAT_Decoder  # Keep existing decoder for now


class DynamicGraphTransformerModel(nn.Module):
    """
    Main model class that combines Graph Transformer encoder with dynamic updates
    and the existing GAT decoder for CVRP solving.
    """
    
    def __init__(self,
                 # Graph Transformer parameters
                 node_input_dim: int = 3,  # x, y, demand
                 edge_input_dim: int = 1,  # distance
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 
                 # Positional encoding parameters
                 pe_type: str = "sinusoidal",
                 pe_dim: int = 64,
                 max_distance: float = 100.0,
                 
                 # Dynamic updates parameters
                 use_dynamic_updates: bool = True,
                 update_frequency: int = 1,
                 
                 # Vehicle parameters
                 vehicle_capacity: float = 50.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_dynamic_updates = use_dynamic_updates
        self.update_frequency = update_frequency
        self.vehicle_capacity = vehicle_capacity
        
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
        
        # Dynamic Graph Updater
        if use_dynamic_updates:
            self.dynamic_updater = DynamicGraphUpdater(
                hidden_dim=hidden_dim,
                update_node_features=True,
                update_edge_weights=True,
                adaptive_masking=True,
                capacity_aware=True
            )
        else:
            self.dynamic_updater = None
        
        # Decoder (keeping existing GAT decoder for now)
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
            actions: Selected actions/routes
            log_p: Log probabilities of actions
        """
        device = data.x.device
        batch_size = data.num_graphs
        total_nodes = data.x.size(0)
        num_nodes = total_nodes // batch_size
        
        # Initial encoding with Graph Transformer
        x = self.encoder(data)  # Shape: (batch_size, num_nodes, hidden_dim)
        
        # Compute graph embedding (mean of all node embeddings)
        graph_embedding = x.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Prepare data for decoder (original format expected by GAT_Decoder)
        x_flat = x.view(batch_size * num_nodes, self.hidden_dim)  # Flatten for decoder
        
        # Get demand and capacity
        demand = data.demand.reshape(batch_size, -1).float().to(device)
        capacity = data.capacity.reshape(batch_size, -1).float().to(device)
        
        if not self.use_dynamic_updates:
            # Standard forward pass without dynamic updates
            actions, log_p = self.decoder(x_flat, graph_embedding, capacity, demand, n_steps, T, greedy)
            return actions, log_p
        
        # Dynamic routing with graph updates
        return self._dynamic_forward(data, x, graph_embedding, demand, capacity, n_steps, T, greedy)
    
    def _dynamic_forward(self, data, x, graph_embedding, demand, capacity, n_steps, T, greedy):
        """
        Forward pass with dynamic graph updates during route construction
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
        
        # Extract coordinates and demands for state tracking
        coordinates = data.x[:, :2].view(batch_size, num_nodes, 2)  # Assuming first 2 features are coordinates
        demands_2d = demand  # Already in (batch_size, num_nodes) format
        
        # Storage for actions and log probabilities
        all_actions = []
        all_log_probs = []
        
        # Initial edge weights from encoder (if available)
        edge_weights = self._compute_initial_edge_weights(data, batch_size, num_nodes)
        
        # Route construction loop
        for step in range(n_steps):
            # Update graph dynamically (every update_frequency steps)
            if step % self.update_frequency == 0:
                routing_state = state_tracker.get_state_dict(coordinates, demands_2d)
                x, edge_weights, attention_mask = self.dynamic_updater(
                    node_embeddings=x,
                    edge_weights=edge_weights,
                    routing_state=routing_state,
                    step=step
                )
                
                # Update graph embedding
                graph_embedding = x.mean(dim=1)
            
            # Prepare current node embeddings for decoder
            x_flat = x.view(batch_size * num_nodes, self.hidden_dim)
            
            # Get feasible nodes mask
            feasible_mask = state_tracker.get_feasible_nodes(demands_2d)
            
            # Single step prediction using decoder
            # Note: We need to modify the decoder to accept feasible mask
            # For now, using the original decoder interface
            actions, log_p = self._single_step_decode(
                x_flat, graph_embedding, capacity, demand, feasible_mask, T, greedy
            )
            
            # Store results
            all_actions.append(actions)
            all_log_probs.append(log_p)
            
            # Update routing state
            if step < n_steps - 1:  # Don't update on last step
                state_tracker.update(actions.squeeze(-1), coordinates, demands_2d)
                
                # Check if all routes are complete
                if state_tracker.is_done().all():
                    break
        
        # Combine all actions and log probabilities
        final_actions = torch.stack(all_actions, dim=1)  # (batch_size, n_steps, 1)
        final_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, n_steps, 1)
        
        return final_actions, final_log_probs
    
    def _compute_initial_edge_weights(self, data, batch_size, num_nodes):
        """
        Compute initial edge weights from the data for dynamic updates
        """
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            return None
            
        device = data.x.device
        edge_weights = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        
        # Simple initialization - can be made more sophisticated
        coordinates = data.x[:, :2].view(batch_size, num_nodes, 2)
        
        # Compute pairwise distances as initial edge weights
        for b in range(batch_size):
            coords = coordinates[b]  # (num_nodes, 2)
            # Compute distance matrix
            coord_i = coords.unsqueeze(1).expand(-1, num_nodes, -1)  # (num_nodes, num_nodes, 2)
            coord_j = coords.unsqueeze(0).expand(num_nodes, -1, -1)   # (num_nodes, num_nodes, 2)
            distances = torch.norm(coord_i - coord_j, dim=-1)  # (num_nodes, num_nodes)
            edge_weights[b] = distances
        
        return edge_weights
    
    def _single_step_decode(self, x_flat, graph_embedding, capacity, demand, feasible_mask, T, greedy):
        """
        Single step decoding with feasibility constraints
        This is a simplified version - the full decoder integration would require more work
        """
        # For now, use the original decoder for a single step
        # This would need to be modified to properly handle the feasibility mask
        
        # Temporary: use original decoder (this won't use feasible_mask properly)
        actions, log_p = self.decoder(x_flat, graph_embedding, capacity, demand, 1, T, greedy)
        
        return actions, log_p


# Backward compatibility: alias for the old Model class
class Model(DynamicGraphTransformerModel):
    """
    Backward compatibility wrapper for the original Model interface
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout):
        # Map old parameters to new ones
        super().__init__(
            node_input_dim=node_input_dim + 1,  # +1 for demand (now included)
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_heads=8,  # Default
            num_layers=layers,
            dropout=dropout,
            pe_type="sinusoidal",
            use_dynamic_updates=True  # Enable by default
        )


if __name__ == "__main__":
    # Test the new model
    import torch
    from torch_geometric.data import Data
    
    # Create test data
    batch_size = 2
    num_nodes = 10
    
    # Node features (x, y coordinates)
    coordinates = torch.rand(batch_size * num_nodes, 2) * 100
    demands = torch.rand(batch_size * num_nodes, 1) * 10
    
    # Create simple edge structure
    edge_indices = []
    edge_attrs = []
    
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    source = b * num_nodes + i
                    target = b * num_nodes + j
                    edge_indices.append([source, target])
                    
                    # Distance as edge attribute
                    coord_i = coordinates[source]
                    coord_j = coordinates[target]
                    distance = torch.norm(coord_i - coord_j, dim=0, keepdim=True)
                    edge_attrs.append(distance)
    
    edge_index = torch.tensor(edge_indices).t().long()
    edge_attr = torch.stack(edge_attrs)
    
    # Create batch indices
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Create data object
    data = Data(
        x=coordinates,
        edge_index=edge_index,
        edge_attr=edge_attr,
        demand=demands,
        capacity=torch.full((batch_size * num_nodes,), 50.0),
        batch=batch
    )
    data.num_graphs = batch_size
    
    # Test model
    model = DynamicGraphTransformerModel(
        node_input_dim=2,  # x, y (demand added automatically)
        edge_input_dim=1,  # distance
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        use_dynamic_updates=True
    )
    
    print(f"Testing Dynamic Graph Transformer Model...")
    print(f"Input coordinates shape: {coordinates.shape}")
    print(f"Input demands shape: {demands.shape}")
    print(f"Batch size: {batch_size}, Nodes per graph: {num_nodes}")
    
    # Forward pass
    n_steps = 5
    actions, log_probs = model(data, n_steps=n_steps, greedy=True)
    
    print(f"Output actions shape: {actions.shape}")
    print(f"Output log_probs shape: {log_probs.shape}")
    print("Dynamic Graph Transformer Model test completed successfully!")
