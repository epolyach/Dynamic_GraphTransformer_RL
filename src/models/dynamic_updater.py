"""
Dynamic Graph Updater for CVRP
This module implements dynamic graph update mechanisms that modify node and edge features
in real-time as routes are being constructed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Tuple, List, Dict
import math


class DynamicGraphUpdater(nn.Module):
    """
    Handles dynamic updates to graph structure and features during routing
    """
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 update_node_features: bool = True,
                 update_edge_weights: bool = True,
                 adaptive_masking: bool = True,
                 capacity_aware: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.update_node_features = update_node_features
        self.update_edge_weights = update_edge_weights
        self.adaptive_masking = adaptive_masking
        self.capacity_aware = capacity_aware
        
        # Node update networks
        if update_node_features:
            self.node_update_mlp = nn.Sequential(
                nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for visit status, remaining capacity, distance to depot, time since visit
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        # Edge weight update networks
        if update_edge_weights:
            self.edge_update_mlp = nn.Sequential(
                nn.Linear(3, hidden_dim // 4),  # distance, capacity constraint violation, temporal factor
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()  # Output weight multiplier
            )
        
        # Attention mask update
        if adaptive_masking:
            self.mask_update_mlp = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim // 2),  # node features + capacity + demand
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def update_node_features(self, 
                           node_embeddings: Tensor,
                           visited_mask: Tensor,
                           current_positions: Tensor,
                           remaining_capacities: Tensor,
                           demands: Tensor,
                           coordinates: Tensor,
                           step: int) -> Tensor:
        """
        Update node features based on current routing state
        
        Args:
            node_embeddings: Current node embeddings [batch_size, num_nodes, hidden_dim]
            visited_mask: Binary mask of visited nodes [batch_size, num_nodes]
            current_positions: Current vehicle positions [batch_size, 2]
            remaining_capacities: Remaining vehicle capacities [batch_size]
            demands: Node demands [batch_size, num_nodes]
            coordinates: Node coordinates [batch_size, num_nodes, 2]
            step: Current step in the route construction
            
        Returns:
            Updated node embeddings [batch_size, num_nodes, hidden_dim]
        """
        if not self.update_node_features:
            return node_embeddings
            
        batch_size, num_nodes, _ = node_embeddings.size()
        device = node_embeddings.device
        
        # Compute additional features
        # 1. Visit status (0 if unvisited, 1 if visited)
        visit_status = visited_mask.float().unsqueeze(-1)  # [batch_size, num_nodes, 1]
        
        # 2. Normalized remaining capacity
        capacity_status = remaining_capacities.unsqueeze(-1).expand(-1, num_nodes, -1) / 50.0  # Normalize by typical capacity
        
        # 3. Distance to current vehicle position
        current_pos_expanded = current_positions.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch_size, num_nodes, 2]
        distances_to_vehicle = torch.norm(coordinates - current_pos_expanded, dim=-1, keepdim=True)  # [batch_size, num_nodes, 1]
        distances_to_vehicle = distances_to_vehicle / 100.0  # Normalize
        
        # 4. Time factor (how long since visiting each node)
        time_factor = torch.ones_like(visit_status, device=device) * (step / 100.0)  # Normalize step count
        time_factor = time_factor * (1 - visited_mask.float().unsqueeze(-1))  # Zero for visited nodes
        
        # Combine additional features
        additional_features = torch.cat([
            visit_status,
            capacity_status, 
            distances_to_vehicle,
            time_factor
        ], dim=-1)  # [batch_size, num_nodes, 4]
        
        # Concatenate with existing embeddings
        combined_features = torch.cat([node_embeddings, additional_features], dim=-1)
        
        # Update through MLP
        updated_embeddings = self.node_update_mlp(combined_features)
        
        # Residual connection
        return node_embeddings + updated_embeddings
    
    def update_edge_weights(self,
                          edge_weights: Optional[Tensor],
                          coordinates: Tensor,
                          demands: Tensor,
                          remaining_capacities: Tensor,
                          visited_mask: Tensor,
                          step: int) -> Optional[Tensor]:
        """
        Update edge weights based on current routing state
        
        Args:
            edge_weights: Current edge weights [batch_size, num_nodes, num_nodes]
            coordinates: Node coordinates [batch_size, num_nodes, 2]
            demands: Node demands [batch_size, num_nodes]
            remaining_capacities: Remaining capacities [batch_size]
            visited_mask: Visited node mask [batch_size, num_nodes]
            step: Current routing step
            
        Returns:
            Updated edge weights [batch_size, num_nodes, num_nodes]
        """
        if not self.update_edge_weights or edge_weights is None:
            return edge_weights
            
        batch_size, num_nodes, _ = coordinates.size()
        device = coordinates.device
        
        # Initialize updated weights
        updated_weights = edge_weights.clone()
        
        # Compute pairwise distances (if not already available)
        coord_i = coordinates.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch_size, num_nodes, num_nodes, 2]
        coord_j = coordinates.unsqueeze(1).expand(-1, num_nodes, -1, -1)   # [batch_size, num_nodes, num_nodes, 2]
        distances = torch.norm(coord_i - coord_j, dim=-1)  # [batch_size, num_nodes, num_nodes]
        
        # Normalize distances
        distances = distances / (distances.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # Capacity constraint violations
        demand_i = demands.unsqueeze(2).expand(-1, -1, num_nodes)  # [batch_size, num_nodes, num_nodes]
        capacity_expanded = remaining_capacities.unsqueeze(1).unsqueeze(2).expand(-1, num_nodes, num_nodes)
        capacity_violations = torch.relu(demand_i - capacity_expanded) / 50.0  # Normalize
        
        # Temporal factor
        temporal_factor = torch.ones_like(distances, device=device) * (step / 100.0)
        
        # Combine factors
        edge_features = torch.stack([
            distances,
            capacity_violations,
            temporal_factor
        ], dim=-1)  # [batch_size, num_nodes, num_nodes, 3]
        
        # Compute weight multipliers
        weight_multipliers = self.edge_update_mlp(edge_features).squeeze(-1)  # [batch_size, num_nodes, num_nodes]
        
        # Apply multipliers
        updated_weights = updated_weights * weight_multipliers
        
        # Zero out weights for visited nodes (except depot)
        visited_expanded = visited_mask.unsqueeze(1).expand(-1, num_nodes, -1)
        updated_weights = updated_weights * (1 - visited_expanded.float())
        
        return updated_weights
    
    def update_attention_mask(self,
                            node_embeddings: Tensor,
                            demands: Tensor,
                            remaining_capacities: Tensor,
                            visited_mask: Tensor) -> Tensor:
        """
        Create adaptive attention masks based on routing constraints
        
        Args:
            node_embeddings: Node embeddings [batch_size, num_nodes, hidden_dim]
            demands: Node demands [batch_size, num_nodes]
            remaining_capacities: Remaining capacities [batch_size]
            visited_mask: Visited nodes mask [batch_size, num_nodes]
            
        Returns:
            Attention mask [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = node_embeddings.size()
        device = node_embeddings.device
        
        if not self.adaptive_masking:
            # Simple constraint-based mask
            capacity_expanded = remaining_capacities.unsqueeze(1).expand(-1, num_nodes)
            capacity_mask = (demands <= capacity_expanded).float()  # [batch_size, num_nodes]
            visit_mask = (1 - visited_mask.float())  # [batch_size, num_nodes]
            
            # Allow depot (node 0) always
            visit_mask[:, 0] = 1.0
            
            # Combined mask
            node_mask = capacity_mask * visit_mask  # [batch_size, num_nodes]
            
            # Expand to attention mask
            attention_mask = node_mask.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch_size, num_nodes, num_nodes]
            return attention_mask
        
        # Advanced adaptive masking using learned features
        capacity_expanded = remaining_capacities.unsqueeze(1).expand(-1, num_nodes, -1)
        demands_expanded = demands.unsqueeze(-1)
        
        # Combine features
        mask_features = torch.cat([
            node_embeddings,
            capacity_expanded,
            demands_expanded
        ], dim=-1)  # [batch_size, num_nodes, hidden_dim + 2]
        
        # Compute mask probabilities
        mask_probs = self.mask_update_mlp(mask_features).squeeze(-1)  # [batch_size, num_nodes]
        
        # Apply hard constraints
        capacity_mask = (demands <= remaining_capacities.unsqueeze(1)).float()
        visit_mask = (1 - visited_mask.float())
        
        # Allow depot always
        visit_mask[:, 0] = 1.0
        capacity_mask[:, 0] = 1.0
        
        # Combine learned and hard constraints
        final_mask = mask_probs * capacity_mask * visit_mask
        
        # Expand to attention dimensions
        attention_mask = final_mask.unsqueeze(1).expand(-1, num_nodes, -1)
        
        return attention_mask
    
    def forward(self,
               node_embeddings: Tensor,
               edge_weights: Optional[Tensor],
               routing_state: Dict[str, Tensor],
               step: int) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Perform dynamic graph updates
        
        Args:
            node_embeddings: Current node embeddings [batch_size, num_nodes, hidden_dim]
            edge_weights: Current edge weights [batch_size, num_nodes, num_nodes] or None
            routing_state: Dictionary containing routing state information
            step: Current routing step
            
        Returns:
            Tuple of (updated_node_embeddings, updated_edge_weights, attention_mask)
        """
        # Extract routing state
        visited_mask = routing_state['visited_mask']
        current_positions = routing_state['current_positions']
        remaining_capacities = routing_state['remaining_capacities']
        demands = routing_state['demands']
        coordinates = routing_state['coordinates']
        
        # Update node features
        updated_embeddings = self.update_node_features(
            node_embeddings=node_embeddings,
            visited_mask=visited_mask,
            current_positions=current_positions,
            remaining_capacities=remaining_capacities,
            demands=demands,
            coordinates=coordinates,
            step=step
        )
        
        # Update edge weights
        updated_edge_weights = self.update_edge_weights(
            edge_weights=edge_weights,
            coordinates=coordinates,
            demands=demands,
            remaining_capacities=remaining_capacities,
            visited_mask=visited_mask,
            step=step
        )
        
        # Update attention mask
        attention_mask = self.update_attention_mask(
            node_embeddings=updated_embeddings,
            demands=demands,
            remaining_capacities=remaining_capacities,
            visited_mask=visited_mask
        )
        
        return updated_embeddings, updated_edge_weights, attention_mask


class RoutingStateTracker:
    """
    Utility class to track routing state across timesteps
    """
    
    def __init__(self, batch_size: int, num_nodes: int, vehicle_capacity: float, device: torch.device):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.vehicle_capacity = vehicle_capacity
        self.device = device
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset routing state for new episodes"""
        self.visited_mask = torch.zeros(self.batch_size, self.num_nodes, dtype=torch.bool, device=self.device)
        self.remaining_capacities = torch.full((self.batch_size,), self.vehicle_capacity, device=self.device)
        self.current_positions = torch.zeros(self.batch_size, 2, device=self.device)  # Start at depot (0,0)
        self.routes = [[] for _ in range(self.batch_size)]  # Store routes for each batch
        self.step = 0
    
    def update(self, selected_nodes: Tensor, coordinates: Tensor, demands: Tensor):
        """
        Update routing state after node selection
        
        Args:
            selected_nodes: Selected node indices [batch_size]
            coordinates: Node coordinates [batch_size, num_nodes, 2]
            demands: Node demands [batch_size, num_nodes]
        """
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        # Update visited mask
        self.visited_mask[batch_indices, selected_nodes] = True
        
        # Update current positions
        self.current_positions = coordinates[batch_indices, selected_nodes]
        
        # Update remaining capacities (only if not returning to depot)
        node_demands = demands[batch_indices, selected_nodes]
        is_depot = (selected_nodes == 0)
        
        # If returning to depot, reset capacity; otherwise, subtract demand
        self.remaining_capacities = torch.where(
            is_depot,
            torch.full_like(self.remaining_capacities, self.vehicle_capacity),
            self.remaining_capacities - node_demands
        )
        
        # Update routes
        for b in range(self.batch_size):
            self.routes[b].append(selected_nodes[b].item())
        
        self.step += 1
    
    def get_state_dict(self, coordinates: Tensor, demands: Tensor) -> Dict[str, Tensor]:
        """
        Get current routing state as dictionary
        
        Args:
            coordinates: Node coordinates [batch_size, num_nodes, 2]
            demands: Node demands [batch_size, num_nodes]
            
        Returns:
            Dictionary containing current routing state
        """
        return {
            'visited_mask': self.visited_mask,
            'current_positions': self.current_positions,
            'remaining_capacities': self.remaining_capacities,
            'coordinates': coordinates,
            'demands': demands,
            'step': self.step
        }
    
    def is_done(self) -> Tensor:
        """
        Check if routing is complete for each batch
        
        Returns:
            Boolean tensor indicating completion [batch_size]
        """
        # Done if all non-depot nodes are visited (except depot node 0)
        all_visited = self.visited_mask[:, 1:].all(dim=1)
        return all_visited
    
    def get_feasible_nodes(self, demands: Tensor) -> Tensor:
        """
        Get mask of feasible nodes for next selection
        
        Args:
            demands: Node demands [batch_size, num_nodes]
            
        Returns:
            Feasible node mask [batch_size, num_nodes]
        """
        # Nodes are feasible if: not visited AND (demand <= remaining_capacity OR is_depot)
        capacity_feasible = (demands <= self.remaining_capacities.unsqueeze(1))
        is_depot = torch.zeros_like(demands, dtype=torch.bool)
        is_depot[:, 0] = True  # Depot is always feasible
        
        capacity_feasible = capacity_feasible | is_depot
        not_visited = ~self.visited_mask
        
        feasible = capacity_feasible & not_visited
        
        # Always allow depot
        feasible[:, 0] = True
        
        return feasible


if __name__ == "__main__":
    # Test the dynamic graph updater
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test parameters
    batch_size = 2
    num_nodes = 10
    hidden_dim = 128
    
    # Create test data
    node_embeddings = torch.randn(batch_size, num_nodes, hidden_dim, device=device)
    edge_weights = torch.randn(batch_size, num_nodes, num_nodes, device=device)
    coordinates = torch.rand(batch_size, num_nodes, 2, device=device) * 100
    demands = torch.rand(batch_size, num_nodes, device=device) * 10
    
    # Initialize updater and state tracker
    updater = DynamicGraphUpdater(
        hidden_dim=hidden_dim,
        update_node_features=True,
        update_edge_weights=True,
        adaptive_masking=True
    ).to(device)
    
    state_tracker = RoutingStateTracker(
        batch_size=batch_size,
        num_nodes=num_nodes,
        vehicle_capacity=50.0,
        device=device
    )
    
    print("Testing Dynamic Graph Updater...")
    
    # Simulate routing steps
    for step in range(5):
        # Get current state
        routing_state = state_tracker.get_state_dict(coordinates, demands)
        
        # Update graph
        updated_embeddings, updated_edge_weights, attention_mask = updater(
            node_embeddings=node_embeddings,
            edge_weights=edge_weights,
            routing_state=routing_state,
            step=step
        )
        
        print(f"Step {step}:")
        print(f"  Node embeddings shape: {updated_embeddings.shape}")
        print(f"  Edge weights shape: {updated_edge_weights.shape if updated_edge_weights is not None else 'None'}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Feasible nodes: {state_tracker.get_feasible_nodes(demands).sum(dim=1).tolist()}")
        
        # Simulate node selection (random feasible node)
        feasible_mask = state_tracker.get_feasible_nodes(demands)
        selected_nodes = torch.multinomial(feasible_mask.float(), 1).squeeze(1)
        
        # Update state
        state_tracker.update(selected_nodes, coordinates, demands)
        
        print(f"  Selected nodes: {selected_nodes.tolist()}")
        print(f"  Remaining capacities: {state_tracker.remaining_capacities.tolist()}")
        print()
    
    print("Dynamic Graph Updater test completed successfully!")
