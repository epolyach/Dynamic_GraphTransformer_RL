import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import math


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for capturing both local and global patterns."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-scale attention with different kernel sizes
        self.local_attention = nn.MultiheadAttention(hidden_dim, num_heads // 2, dropout=dropout, batch_first=True)
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads // 2, dropout=dropout, batch_first=True)
        
        # Learnable combination weights
        self.scale_weights = nn.Parameter(torch.ones(2))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Attention mask [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Create distance-based attention mask for local attention
        local_mask = None
        if mask is not None and seq_len > 1:
            # Create a mask for local neighborhood (e.g., nearest neighbors)
            distance_matrix = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)).float()
            local_window = min(8, seq_len // 2)  # Adaptive window size
            local_mask = (distance_matrix <= local_window).to(x.device)
            if mask is not None:
                local_mask = local_mask & mask.unsqueeze(1) & mask.unsqueeze(2)
        
        # Local attention (focuses on nearby nodes)
        local_out, _ = self.local_attention(x, x, x, key_padding_mask=~mask if mask is not None else None)
        
        # Global attention (considers all nodes)
        global_out, _ = self.global_attention(x, x, x, key_padding_mask=~mask if mask is not None else None)
        
        # Weighted combination of local and global features
        scale_weights = F.softmax(self.scale_weights, dim=0)
        combined = scale_weights[0] * local_out + scale_weights[1] * global_out
        
        return self.layer_norm(x + combined)


class GeometricEmbedding(nn.Module):
    """Enhanced geometric embedding with distance and angle features."""
    
    def __init__(self, coord_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        
        # Coordinate embedding with positional encoding
        self.coord_proj = nn.Linear(coord_dim, hidden_dim // 2)
        
        # Distance and angle features
        self.distance_embedding = nn.Embedding(100, hidden_dim // 4)  # Discretized distances
        self.angle_embedding = nn.Embedding(36, hidden_dim // 4)      # 10-degree bins
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Coordinates tensor [batch_size, num_nodes, coord_dim]
        Returns:
            Enhanced geometric embeddings [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = coords.shape
        
        # Basic coordinate projection
        coord_emb = self.coord_proj(coords)
        
        # Compute pairwise distances and angles from depot (node 0)
        depot_coords = coords[:, 0:1, :]  # [batch_size, 1, coord_dim]
        relative_coords = coords - depot_coords  # [batch_size, num_nodes, coord_dim]
        
        # Distance from depot (discretized)
        distances = torch.norm(relative_coords, dim=-1)  # [batch_size, num_nodes]
        distance_bins = torch.clamp((distances * 10).long(), 0, 99)
        dist_emb = self.distance_embedding(distance_bins)
        
        # Angle from depot (discretized into 36 bins of 10 degrees)
        angles = torch.atan2(relative_coords[:, :, 1], relative_coords[:, :, 0])
        angle_bins = torch.clamp(((angles + math.pi) / (2 * math.pi) * 36).long(), 0, 35)
        angle_emb = self.angle_embedding(angle_bins)
        
        # Combine embeddings
        combined = torch.cat([coord_emb, dist_emb, angle_emb], dim=-1)
        return self.output_proj(combined)


class AdaptiveDynamicUpdate(nn.Module):
    """Adaptive dynamic update mechanism with gating and residual connections."""
    
    def __init__(self, hidden_dim: int, state_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # State encoder with batch normalization
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Dynamic update network
        self.update_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Adaptive gating mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_embeddings: torch.Tensor, state_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [batch_size, num_nodes, hidden_dim]
            state_features: [batch_size, state_dim]
        Returns:
            Updated node embeddings [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = node_embeddings.shape
        
        # Encode state features
        state_emb = self.state_encoder(state_features)  # [batch_size, hidden_dim]
        state_emb = state_emb.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch_size, num_nodes, hidden_dim]
        
        # Combine node embeddings with state
        combined = torch.cat([node_embeddings, state_emb], dim=-1)  # [batch_size, num_nodes, hidden_dim * 2]
        
        # Compute update and gate
        update = self.update_network(combined)
        gate = self.gate_network(combined)
        
        # Apply gated residual update
        updated = node_embeddings + gate * update
        
        return self.output_proj(updated)


class EnhancedDynamicGraphTransformer(nn.Module):
    """Enhanced Dynamic Graph Transformer with improved architectural components."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, 
                 dropout: float, feedforward_multiplier: int, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Enhanced geometric embedding
        self.geometric_embedding = GeometricEmbedding(coord_dim=2, hidden_dim=hidden_dim)
        
        # Demand embedding
        self.demand_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Input projection
        self.input_projection = nn.Linear(input_dim + hidden_dim // 2, hidden_dim)
        
        # Multi-scale transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiScaleAttention(hidden_dim, num_heads, dropout),
                'feedforward': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * feedforward_multiplier),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * feedforward_multiplier, hidden_dim)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        
        # Dynamic update mechanism
        dgt_config = config.get('model', {}).get('dynamic_graph_transformer', {})
        state_features = dgt_config.get('state_features', 4)
        self.dynamic_update = AdaptiveDynamicUpdate(hidden_dim, state_features, dropout)
        
        # Enhanced pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, instances: List[Dict[str, Any]], max_steps: int = None, 
                temperature: float = None, greedy: bool = False, config: Dict[str, Any] = None) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced architecture."""
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        if temperature is None:
            temperature = config['inference']['default_temperature']
            
        max_nodes = max(len(inst['coords']) for inst in instances)
        
        # Prepare input tensors
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
        
        # Enhanced embeddings
        coords = node_features[:, :, :2]
        demands = node_features[:, :, 2:3]
        
        # Geometric embedding
        geom_emb = self.geometric_embedding(coords)
        
        # Demand embedding
        demand_emb = self.demand_embedding(demands)
        
        # Combine and project
        basic_features = torch.cat([node_features, demand_emb], dim=-1)
        embeddings = self.input_projection(basic_features) + geom_emb
        
        # Apply transformer layers
        x = embeddings
        for layer in self.transformer_layers:
            # Multi-scale attention
            attended = layer['attention'](layer['norm1'](x))
            x = x + layer['dropout'](attended)
            
            # Feedforward
            fed_forward = layer['feedforward'](layer['norm2'](x))
            x = x + layer['dropout'](fed_forward)
        
        return self._generate_routes_enhanced(x, node_features, demands_batch, capacities, 
                                            distances_batch, max_steps, temperature, greedy, instances, config)
    
    def _generate_routes_enhanced(self, node_embeddings: torch.Tensor, node_features: torch.Tensor, 
                                demands_batch: torch.Tensor, capacities: torch.Tensor, 
                                distances_batch: torch.Tensor, max_steps: int, temperature: float, 
                                greedy: bool, instances: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Enhanced route generation with improved state tracking and decision making."""
        
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        # Initialize state
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        current_nodes = torch.zeros(batch_size, dtype=torch.long)
        
        # Start at depot
        for b in range(batch_size):
            routes[b].append(0)
        
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        for step in range(max_steps):
            # Check completion
            for b in range(batch_size):
                if not batch_done[b]:
                    actual_nodes = len(instances[b]['coords']) if instances and b < len(instances) else max_nodes
                    customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                    currently_at_depot = current_nodes[b].item() == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            if batch_done.all():
                break
            
            # Enhanced state features
            capacity_used = (capacities - remaining_capacity) / capacities
            step_progress = torch.full((batch_size,), step / max_steps)
            visited_ratio = visited.float().sum(dim=1) / max_nodes
            
            # Distance to depot
            distance_from_depot = torch.zeros(batch_size)
            for b in range(batch_size):
                current_pos = current_nodes[b].item()
                distance_from_depot[b] = distances_batch[b, current_pos, 0]
            
            # Combine state features
            state_features = torch.stack([
                capacity_used, step_progress, visited_ratio, distance_from_depot
            ], dim=1)
            
            # Apply dynamic update
            updated_embeddings = self.dynamic_update(node_embeddings, state_features)
            
            # Compute attention scores
            global_context = updated_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            state_context = state_features.unsqueeze(1).expand(-1, max_nodes, -1)
            
            # Add current position embedding
            current_embedding = torch.zeros(batch_size, max_nodes, hidden_dim)
            for b in range(batch_size):
                if not batch_done[b]:
                    current_pos = current_nodes[b].item()
                    if current_pos < max_nodes:
                        current_embedding[b, :, :] = updated_embeddings[b, current_pos, :].unsqueeze(0)
            
            # Enhanced pointer input
            pointer_input = torch.cat([updated_embeddings, global_context, current_embedding], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Create comprehensive mask
            cap_mask = demands_batch > remaining_capacity.unsqueeze(1)
            mask = visited | cap_mask
            
            # Padding mask
            pad_mask = torch.zeros_like(mask)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords']) if instances and b < len(instances) else max_nodes
                if actual_nodes < max_nodes:
                    pad_mask[b, actual_nodes:] = True
                pad_mask[b, 0] = False  # Always allow depot
            
            mask = mask | pad_mask
            
            # Depot constraint logic
            currently_at_depot_vec = torch.tensor([len(r) > 0 and r[-1] == 0 for r in routes])
            if currently_at_depot_vec.any():
                mask[currently_at_depot_vec, 0] = True
            
            # Allow depot if all other nodes are masked
            all_masked = mask.all(dim=1)
            need_allow_depot = all_masked & (~currently_at_depot_vec)
            if need_allow_depot.any():
                mask[need_allow_depot, 0] = False
            
            # Mark done if at depot with all customers visited
            done_mask = all_masked & currently_at_depot_vec
            batch_done[done_mask] = True
            
            # Apply mask and compute probabilities
            masked_score_value = config['inference']['masked_score_value']
            scores = scores.masked_fill(mask, masked_score_value)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            
            log_prob_epsilon = config['inference']['log_prob_epsilon']
            log_probs = torch.log(probs + log_prob_epsilon)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Sample actions
            actions = torch.zeros(batch_size, dtype=torch.long)
            selected_log_probs = torch.zeros(batch_size)
            
            for b in range(batch_size):
                if not batch_done[b]:
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        actions[b] = torch.multinomial(probs[b], 1).squeeze()
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            all_entropies.append(step_entropy)
            
            # Update state
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    current_nodes[b] = action
                    
                    if action == 0:  # Depot
                        remaining_capacity[b] = capacities[b]
                    else:  # Customer
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
        
        # Ensure all routes start at depot
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        # Combine log probabilities and entropies
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        
        return routes, combined_log_probs, combined_entropy
