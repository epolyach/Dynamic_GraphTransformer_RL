"""
Advanced Graph Transformer (GT+RL) with complete architecture.
This is a serious upgrade over legacy GAT+RL with:
- Positional and spatial encoding
- Distance-aware attention
- Dynamic state tracking
- Multi-head pointer attention
- Dynamic graph updates
- CVRP-specific inductive biases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Any, Tuple, Optional


class SpatialPositionalEncoding(nn.Module):
    """Combine sinusoidal positional encoding with learned spatial encoding."""
    
    def __init__(self, hidden_dim: int, max_nodes: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Sinusoidal positional encoding for sequence position
        pe = torch.zeros(max_nodes, hidden_dim)
        position = torch.arange(0, max_nodes).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Learned spatial encoding from coordinates
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Combine positional and spatial
        self.combine = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
            coords: Node coordinates [batch_size, num_nodes, 2]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Sinusoidal position encoding
        pos_enc = self.pe[:num_nodes].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Spatial encoding from coordinates
        spatial_enc = self.coord_encoder(coords)
        
        # Combine with input
        combined = torch.cat([x + pos_enc, spatial_enc], dim=-1)
        return self.combine(combined)


class DistanceAwareAttention(nn.Module):
    """Multi-head attention with distance-based bias and edge features."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Distance embedding
        self.distance_embedding = nn.Embedding(50, num_heads)  # 50 distance bins
        
        # Learnable attention bias
        self.attention_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, distance_matrix: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, num_nodes, hidden_dim]
            distance_matrix: Pairwise distances [batch_size, num_nodes, num_nodes]
            mask: Attention mask [batch_size, num_nodes]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, num_nodes, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add distance-based bias
        distance_bins = torch.clamp((distance_matrix * 10).long(), 0, 49)
        distance_bias = self.distance_embedding(distance_bins)
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # [batch_size, num_heads, num_nodes, num_nodes]
        scores = scores + distance_bias + self.attention_bias
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, num_nodes]
            scores = scores.masked_fill(mask.expand(-1, self.num_heads, num_nodes, -1), -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, num_nodes, self.hidden_dim)
        
        return self.out_proj(out)


class StateEncoder(nn.Module):
    """Encode the current state of route construction."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encode current node
        self.current_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Encode visited mask as features
        self.visited_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Encode remaining capacity
        self.capacity_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Combine state components
        self.state_combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_embeddings: torch.Tensor, current_node: torch.Tensor,
                visited: torch.Tensor, remaining_capacity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [batch_size, num_nodes, hidden_dim]
            current_node: [batch_size] indices of current nodes
            visited: [batch_size, num_nodes] boolean mask
            remaining_capacity: [batch_size, 1] normalized remaining capacity
        
        Returns:
            State encoding [batch_size, hidden_dim]
        """
        batch_size = node_embeddings.size(0)
        
        # Get current node embeddings
        current_emb = node_embeddings[torch.arange(batch_size), current_node]
        current_emb = self.current_encoder(current_emb)
        
        # Encode visited status
        visited_emb = self.visited_encoder(visited.float().unsqueeze(-1))
        visited_emb = visited_emb.mean(dim=1)  # Aggregate over nodes
        
        # Encode remaining capacity
        capacity_emb = self.capacity_encoder(remaining_capacity)
        
        # Combine all state components
        state = torch.cat([current_emb, visited_emb, capacity_emb.squeeze(1)], dim=-1)
        return self.state_combiner(state)


class MultiHeadPointerAttention(nn.Module):
    """Multi-head pointer attention for node selection."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query from state
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Key and value from nodes
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Pointer projection
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads)
        )
        
        # Learnable temperature per head
        self.temperature = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, state: torch.Tensor, nodes: torch.Tensor, 
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Current state [batch_size, hidden_dim]
            nodes: Node embeddings [batch_size, num_nodes, hidden_dim]
            mask: Feasibility mask [batch_size, num_nodes]
        
        Returns:
            logits: Unnormalized scores [batch_size, num_nodes]
            attention: Attention weights [batch_size, num_heads, num_nodes]
        """
        batch_size, num_nodes, _ = nodes.shape
        
        # Project state as query
        Q = self.q_proj(state).view(batch_size, 1, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        
        # Project nodes as keys and values
        K = self.k_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)  # [batch_size, num_heads, num_nodes, head_dim]
        V = self.v_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_heads, 1, num_nodes]
        scores = scores.squeeze(2) / math.sqrt(self.head_dim)
        
        # Apply temperature scaling per head
        scores = scores / self.temperature.view(1, -1, 1)
        
        # Compute pointer scores using values directly
        # Reshape V for pointer network: [batch_size, num_nodes, num_heads * head_dim]
        V_for_pointer = V.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        pointer_scores = self.pointer(V_for_pointer)  # [batch_size, num_nodes, num_heads]
        
        # Combine scores across heads
        combined_scores = (scores + pointer_scores.transpose(1, 2)).mean(dim=1)
        
        # Apply mask
        combined_scores = combined_scores.masked_fill(mask, -1e9)
        
        return combined_scores, scores


class DynamicGraphUpdater(nn.Module):
    """Update node embeddings based on partial solution."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Update network
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_embeddings: torch.Tensor, state: torch.Tensor,
                visited: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [batch_size, num_nodes, hidden_dim]
            state: Current state [batch_size, hidden_dim]
            visited: [batch_size, num_nodes] boolean mask
        
        Returns:
            Updated embeddings [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = node_embeddings.shape
        
        # Expand state to all nodes
        state_expanded = state.unsqueeze(1).expand(-1, num_nodes, -1)
        
        # Compute gate and update
        combined = torch.cat([node_embeddings, state_expanded], dim=-1)
        gate_values = self.gate(combined)
        updates = self.update(combined)
        
        # Apply gated update (stronger for unvisited nodes)
        visited_weight = visited.float().unsqueeze(-1)
        updates = updates * (1 - visited_weight * 0.5)  # Reduce updates for visited nodes
        
        return node_embeddings + gate_values * updates


class ImprovedTransformerLayer(nn.Module):
    """Enhanced transformer layer with pre-LN and GLU."""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Pre-LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Distance-aware attention
        self.attention = DistanceAwareAttention(hidden_dim, num_heads, dropout)
        
        # GLU-based feedforward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, distance_matrix: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN + Attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, distance_matrix, mask)
        x = residual + self.dropout(x)
        
        # Pre-LN + FFN
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + self.dropout(x)
        
        return x


class GraphTransformer(nn.Module):
    """
    Advanced Graph Transformer with all missing components.
    A serious upgrade over legacy GAT+RL.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, 
                 num_layers: int, dropout: float, feedforward_multiplier: int, config: Dict):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initial node embedding with demand awareness
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Demand-specific encoding
        self.demand_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Depot special embedding
        self.depot_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim // 4))
        
        # Combine features
        self.feature_combiner = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        
        # Spatial and positional encoding
        self.spatial_encoder = SpatialPositionalEncoding(hidden_dim)
        
        # Stack of improved transformer layers
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerLayer(hidden_dim, num_heads, feedforward_multiplier, dropout)
            for _ in range(num_layers)
        ])
        
        # State encoder
        self.state_encoder = StateEncoder(hidden_dim)
        
        # Multi-head pointer attention
        self.pointer_attention = MultiHeadPointerAttention(hidden_dim, num_heads)
        
        # Dynamic graph updater
        self.graph_updater = DynamicGraphUpdater(hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def compute_distance_matrix(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        # coords: [batch_size, num_nodes, 2]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        return torch.norm(diff, dim=-1)
    
    def forward(self, instances: List[Dict], max_steps: Optional[int] = None,
                temperature: Optional[float] = None, greedy: bool = False,
                config: Optional[Dict] = None) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic updates and state tracking."""
        
        config = config or {}
        batch_size = len(instances)
        
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config.get('inference', {}).get('max_steps_multiplier', 2)
        if temperature is None:
            temperature = config.get('inference', {}).get('default_temperature', 1.0)
        
        # Prepare batch data
        max_nodes = max(len(inst['coords']) for inst in instances)
        device = next(self.parameters()).device
        
        node_coords = torch.zeros(batch_size, max_nodes, 2, device=device)
        node_features = torch.zeros(batch_size, max_nodes, 3, device=device)
        demands_batch = torch.zeros(batch_size, max_nodes, device=device)
        capacities = torch.zeros(batch_size, device=device)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            coords_tensor = inst["coords"].detach().clone().to(device) if isinstance(inst["coords"], torch.Tensor) else torch.tensor(inst["coords"], dtype=torch.float32, device=device)
            demands_tensor = inst["demands"].detach().clone().float().to(device) if isinstance(inst["demands"], torch.Tensor) else torch.tensor(inst["demands"], dtype=torch.float32, device=device)
            
            node_coords[i, :n_nodes] = coords_tensor
            node_features[i, :n_nodes, :2] = coords_tensor
            node_features[i, :n_nodes, 2] = demands_tensor
            demands_batch[i, :n_nodes] = demands_tensor
            capacities[i] = inst['capacity']
        
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(node_coords)
        
        # Initial embeddings
        embedded = self.node_embedding(node_features)
        
        # Add demand encoding
        demand_enc = self.demand_encoder(demands_batch.unsqueeze(-1))
        
        # Add depot special encoding
        depot_enc = torch.zeros(batch_size, max_nodes, self.hidden_dim // 4, device=device)
        depot_enc[:, 0, :] = self.depot_embedding.expand(batch_size, -1, -1).squeeze(1)
        
        # Combine all features
        combined = torch.cat([embedded, demand_enc, depot_enc], dim=-1)
        node_embeddings = self.feature_combiner(combined)
        
        # Add spatial encoding
        node_embeddings = self.spatial_encoder(node_embeddings, node_coords)
        
        # Initial transformer encoding
        x = node_embeddings
        for layer in self.transformer_layers:
            x = layer(x, distance_matrix)
        
        # Route generation with dynamic updates
        return self._generate_routes_with_state(
            x, node_coords, demands_batch, capacities, 
            distance_matrix, max_steps, temperature, greedy, instances, config
        )
    
    def _generate_routes_with_state(self, initial_embeddings: torch.Tensor,
                                   coords: torch.Tensor, demands: torch.Tensor,
                                   capacities: torch.Tensor, distance_matrix: torch.Tensor,
                                   max_steps: int, temperature: float, greedy: bool,
                                   instances: List[Dict], config: Dict) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Generate routes with dynamic state tracking and embedding updates."""
        
        batch_size, max_nodes, hidden_dim = initial_embeddings.shape
        device = initial_embeddings.device
        
        routes: List[List[int]] = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        # Initialize state
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        current_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)
        node_embeddings = initial_embeddings.clone()
        
        # Start from depot
        for b in range(batch_size):
            routes[b].append(0)
        
        for step in range(max_steps):
            # Check termination
            batch_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords'])
                customers_visited = visited[b, 1:actual_nodes].all()
                at_depot = (current_nodes[b] == 0)
                if customers_visited and at_depot:
                    batch_done[b] = True
            
            if batch_done.all():
                break
            
            # Encode current state
            normalized_capacity = (remaining_capacity / capacities).unsqueeze(-1)
            state = self.state_encoder(node_embeddings, current_nodes, visited, normalized_capacity)
            
            # Dynamic embedding update
            if step > 0:
                node_embeddings = self.graph_updater(node_embeddings, state, visited)
                
                # Re-encode with transformer layers (lighter pass)
                for i in range(min(2, self.num_layers)):  # Only use first 2 layers for updates
                    node_embeddings = self.transformer_layers[i](node_embeddings, distance_matrix)
            
            # Create mask
            mask = self._create_mask(visited, demands, remaining_capacity, 
                                    current_nodes, instances, batch_size, max_nodes)
            
            # Multi-head pointer attention
            logits, attention_weights = self.pointer_attention(state, node_embeddings, mask)
            
            # Sample or greedy selection
            probs = F.softmax(logits / temperature, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            
            if greedy:
                actions = logits.argmax(dim=-1)
            else:
                actions = torch.multinomial(probs, 1).squeeze(-1)
            
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            
            # Apply mask to outputs for finished batches
            selected_log_probs = selected_log_probs * (~batch_done).float()
            entropy = entropy * (~batch_done).float()
            
            all_log_probs.append(selected_log_probs)
            all_entropies.append(entropy)
            
            # Update state (avoid in-place modifications)
            new_current_nodes = current_nodes.clone()
            new_remaining_capacity = remaining_capacity.clone()
            new_visited = visited.clone()
            
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    new_current_nodes[b] = action
                    
                    if action == 0:  # Return to depot
                        new_remaining_capacity[b] = capacities[b]
                    else:
                        new_visited[b, action] = True
                        new_remaining_capacity[b] -= demands[b, action]
            
            current_nodes = new_current_nodes
            remaining_capacity = new_remaining_capacity
            visited = new_visited
        
        # Ensure all routes end at depot
        for b in range(batch_size):
            if routes[b][-1] != 0:
                routes[b].append(0)
        
        # Combine log probs and entropy
        if all_log_probs:
            combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1)
            combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1)
        else:
            combined_log_probs = torch.zeros(batch_size, device=device)
            combined_entropy = torch.zeros(batch_size, device=device)
        
        return routes, combined_log_probs, combined_entropy
    
    def _create_mask(self, visited: torch.Tensor, demands: torch.Tensor,
                    remaining_capacity: torch.Tensor, current_nodes: torch.Tensor,
                    instances: List[Dict], batch_size: int, max_nodes: int) -> torch.Tensor:
        """Create feasibility mask for node selection."""
        
        device = visited.device
        
        # Base mask: visited nodes and capacity constraints
        mask = visited.clone()
        capacity_mask = demands > remaining_capacity.unsqueeze(1)
        mask = mask | capacity_mask
        
        # Padding mask
        for b in range(batch_size):
            actual_nodes = len(instances[b]['coords'])
            if actual_nodes < max_nodes:
                mask[b, actual_nodes:] = True
        
        # Depot handling
        for b in range(batch_size):
            at_depot = (current_nodes[b] == 0)
            actual_nodes = len(instances[b]['coords'])
            
            # Can't stay at depot if unvisited customers exist
            if at_depot and not visited[b, 1:actual_nodes].all():
                mask[b, 0] = True
            
            # Must return to depot if all customers visited
            if visited[b, 1:actual_nodes].all() and not at_depot:
                mask[b, :] = True
                mask[b, 0] = False
        
        return mask


# Alias for backward compatibility
AdvancedGraphTransformer = GraphTransformer
