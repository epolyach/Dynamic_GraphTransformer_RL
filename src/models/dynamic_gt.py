"""
Dynamic Graph Transformer (DGT+RL) - The Ultimate CVRP Model.
This is a true dynamic upgrade of GT+RL with:
- All GT+RL capabilities PLUS:
- Adaptive graph structure updates
- Multi-scale temporal reasoning
- Dynamic edge representations
- Learned update schedules
- Memory-augmented decision making
- Progressive refinement during decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Any, Tuple, Optional


class TemporalMemoryBank(nn.Module):
    """Memory bank for tracking decision history and patterns."""
    
    def __init__(self, hidden_dim: int, memory_slots: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        
        # Memory initialization
        self.memory_init = nn.Parameter(torch.randn(1, memory_slots, hidden_dim) * 0.01)
        
        # Memory update networks
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_slots),
            nn.Sigmoid()
        )
        
        self.memory_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Read attention
        self.read_query = nn.Linear(hidden_dim, hidden_dim)
        self.read_key = nn.Linear(hidden_dim, hidden_dim)
        self.read_value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, state: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Current state [batch_size, hidden_dim]
            memory: Previous memory state [batch_size, memory_slots, hidden_dim]
        
        Returns:
            read_vector: Information read from memory [batch_size, hidden_dim]
            updated_memory: Updated memory bank [batch_size, memory_slots, hidden_dim]
        """
        batch_size = state.size(0)
        
        # Initialize memory if not provided
        if memory is None:
            memory = self.memory_init.expand(batch_size, -1, -1)
        
        # Read from memory using attention
        query = self.read_query(state).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        keys = self.read_key(memory)  # [batch_size, memory_slots, hidden_dim]
        values = self.read_value(memory)  # [batch_size, memory_slots, hidden_dim]
        
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        read_vector = torch.matmul(attention_weights, values).squeeze(1)
        
        # Update memory with gating
        state_expanded = state.unsqueeze(1).expand(-1, self.memory_slots, -1)
        combined = torch.cat([memory, state_expanded], dim=-1)
        
        gate = self.memory_gate(combined.mean(dim=1)).unsqueeze(1)  # [batch_size, 1, memory_slots]
        gate = gate.transpose(1, 2)  # [batch_size, memory_slots, 1]
        
        update = self.memory_update(combined)
        updated_memory = memory * (1 - gate) + update * gate
        
        return read_vector, updated_memory


class DynamicEdgeProcessor(nn.Module):
    """Process and update edge representations dynamically."""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Edge feature extraction
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Dynamic edge update based on node states
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, edge_dim * 2),
            nn.LayerNorm(edge_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_dim * 2, edge_dim)
        )
        
        # Edge-to-node message passing
        self.edge_to_node = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, node_embeddings: torch.Tensor, distance_matrix: torch.Tensor,
                visited: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embeddings: [batch_size, num_nodes, hidden_dim]
            distance_matrix: [batch_size, num_nodes, num_nodes]
            visited: [batch_size, num_nodes] boolean mask
        
        Returns:
            edge_features: Dynamic edge representations [batch_size, num_nodes, num_nodes, edge_dim]
            edge_messages: Messages to add to nodes [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        
        # Encode distances
        distance_features = self.distance_encoder(distance_matrix.unsqueeze(-1))
        
        # Create dynamic edge features based on node pairs
        node_i = node_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = node_embeddings.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        edge_input = torch.cat([node_i, node_j, distance_features], dim=-1)
        edge_features = self.edge_update(edge_input)
        
        # Mask edges from visited nodes (reduce their influence)
        visited_mask = visited.unsqueeze(2).expand(-1, -1, num_nodes)
        edge_features = edge_features * (1 - visited_mask.unsqueeze(-1) * 0.5)
        
        # Aggregate edge messages for each node
        edge_messages = self.edge_to_node(edge_features.mean(dim=2))
        
        return edge_features, edge_messages


class MultiScaleTemporalAttention(nn.Module):
    """Multi-scale attention for different temporal horizons."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_scales: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        
        # Different attention modules for different scales
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads // num_scales, batch_first=True)
            for _ in range(num_scales)
        ])
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                step: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, num_nodes, hidden_dim]
            mask: Attention mask [batch_size, num_nodes]
            step: Current decoding step (for temporal awareness)
        """
        scale_outputs = []
        
        for i, (attn, proj) in enumerate(zip(self.scale_attentions, self.scale_projections)):
            # Apply different temporal biases for each scale
            scale_factor = 2 ** i  # Exponentially increasing receptive field
            
            # Project input for this scale
            x_scaled = proj(x)
            
            # Apply attention
            attn_out, _ = attn(x_scaled, x_scaled, x_scaled, 
                             key_padding_mask=mask if mask is not None else None)
            
            scale_outputs.append(attn_out)
        
        # Weighted combination of scales
        scale_weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * out for w, out in zip(scale_weights, scale_outputs))
        
        return self.output_proj(combined)


class AdaptiveGraphStructure(nn.Module):
    """Dynamically adapt graph structure based on solution progress."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Graph structure predictor
        self.structure_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Edge importance scorer
        self.edge_importance = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_embeddings: torch.Tensor, visited: torch.Tensor,
                remaining_capacity: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix based on current state.
        
        Returns:
            Adaptive adjacency weights [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        
        # Compute pairwise features
        node_i = node_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = node_embeddings.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        pair_features = torch.cat([node_i, node_j], dim=-1)
        
        # Predict edge importance
        edge_weights = self.edge_importance(pair_features).squeeze(-1)
        
        # Mask visited nodes
        visited_mask = visited.unsqueeze(1) | visited.unsqueeze(2)
        edge_weights = edge_weights.masked_fill(visited_mask, -1e9)
        
        # Apply softmax to get adjacency weights
        adjacency = F.softmax(edge_weights, dim=-1)
        
        return adjacency


class ProgressiveRefinementLayer(nn.Module):
    """Layer that progressively refines representations during decoding."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Refinement networks for different stages
        self.early_stage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.mid_stage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.late_stage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Stage mixer
        self.stage_gate = nn.Sequential(
            nn.Linear(1, 3),  # Input is progress ratio
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, progress: float) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, num_nodes, hidden_dim]
            progress: Solution progress ratio [0, 1]
        """
        # Compute stage weights based on progress
        progress_tensor = torch.tensor([progress], device=x.device)
        stage_weights = self.stage_gate(progress_tensor.unsqueeze(0))
        
        # Apply stage-specific transformations
        early_out = self.early_stage(x)
        mid_out = self.mid_stage(x)
        late_out = self.late_stage(x)
        
        # Weighted combination
        refined = (stage_weights[0, 0] * early_out + 
                  stage_weights[0, 1] * mid_out + 
                  stage_weights[0, 2] * late_out)
        
        return x + refined  # Residual connection


class DynamicGraphTransformer(nn.Module):
    """
    The Ultimate Dynamic Graph Transformer for CVRP.
    Combines all advanced features with true dynamic graph adaptation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int,
                 num_layers: int, dropout: float, feedforward_multiplier: int, config: Dict):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # ============= Initial Encoding (from GT+RL) =============
        # Node embedding with demand awareness
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
        
        # Spatial and positional encoding (from GT+RL)
        from src.models.gt import SpatialPositionalEncoding
        self.spatial_encoder = SpatialPositionalEncoding(hidden_dim)
        
        # ============= Dynamic Components (NEW) =============
        # Memory bank for temporal reasoning
        self.memory_bank = TemporalMemoryBank(hidden_dim, memory_slots=32)
        
        # Dynamic edge processor
        self.edge_processor = DynamicEdgeProcessor(hidden_dim, edge_dim=16)
        
        # Adaptive graph structure
        self.graph_adapter = AdaptiveGraphStructure(hidden_dim)
        
        # Multi-scale temporal attention
        self.temporal_attention = MultiScaleTemporalAttention(hidden_dim, num_heads, num_scales=3)
        
        # Progressive refinement layers
        self.refinement_layers = nn.ModuleList([
            ProgressiveRefinementLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # ============= Enhanced Transformer Layers =============
        from src.models.gt import ImprovedTransformerLayer
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerLayer(hidden_dim, num_heads, feedforward_multiplier, dropout)
            for _ in range(num_layers)
        ])
        
        # ============= Dynamic State Components =============
        # Enhanced state encoder with memory
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),  # +4 for state features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dynamic graph updater (enhanced from GT+RL)
        self.graph_updater = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learned update schedule
        self.update_schedule = nn.Sequential(
            nn.Linear(1, 8),  # Input is progress
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # ============= Enhanced Pointer Network =============
        # Multi-head pointer with memory augmentation
        self.pointer_attention = nn.ModuleDict({
            'q_proj': nn.Linear(hidden_dim * 2, hidden_dim),  # State + memory
            'k_proj': nn.Linear(hidden_dim, hidden_dim),
            'v_proj': nn.Linear(hidden_dim, hidden_dim),
            'pointer': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_heads)
            )
        })
        
        # Temperature controller
        self.temperature_controller = nn.Sequential(
            nn.Linear(4, 8),  # State features
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus()
        )
        
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
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_distance_matrix(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        return torch.norm(diff, dim=-1)
    
    def forward(self, instances: List[Dict], max_steps: Optional[int] = None,
                temperature: Optional[float] = None, greedy: bool = False,
                config: Optional[Dict] = None) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Forward pass with full dynamic graph adaptation."""
        
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
            coords_tensor = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            demands_tensor = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            
            node_coords[i, :n_nodes] = coords_tensor
            node_features[i, :n_nodes, :2] = coords_tensor
            node_features[i, :n_nodes, 2] = demands_tensor
            demands_batch[i, :n_nodes] = demands_tensor
            capacities[i] = inst['capacity']
        
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(node_coords)
        
        # Initial embeddings (same as GT+RL)
        embedded = self.node_embedding(node_features)
        demand_enc = self.demand_encoder(demands_batch.unsqueeze(-1))
        depot_enc = torch.zeros(batch_size, max_nodes, self.hidden_dim // 4, device=device)
        depot_enc[:, 0, :] = self.depot_embedding.expand(batch_size, -1, -1).squeeze(1)
        
        combined = torch.cat([embedded, demand_enc, depot_enc], dim=-1)
        node_embeddings = self.feature_combiner(combined)
        node_embeddings = self.spatial_encoder(node_embeddings, node_coords)
        
        # Initial transformer encoding with progressive refinement
        x = node_embeddings
        for i, (transformer_layer, refinement_layer) in enumerate(zip(self.transformer_layers, self.refinement_layers)):
            x = transformer_layer(x, distance_matrix)
            x = refinement_layer(x, progress=0.0)  # Initial encoding
        
        # Dynamic route generation
        return self._generate_routes_dynamic(
            x, node_coords, demands_batch, capacities,
            distance_matrix, max_steps, temperature, greedy, instances, config
        )
    
    def _generate_routes_dynamic(self, initial_embeddings: torch.Tensor,
                                coords: torch.Tensor, demands: torch.Tensor,
                                capacities: torch.Tensor, distance_matrix: torch.Tensor,
                                max_steps: int, temperature: float, greedy: bool,
                                instances: List[Dict], config: Dict) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Generate routes with full dynamic adaptation."""
        
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
        
        # Initialize memory bank
        memory = None
        
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
            
            # Compute progress
            progress = step / max_steps
            
            # === Dynamic Graph Updates ===
            # 1. Process edges dynamically
            edge_features, edge_messages = self.edge_processor(node_embeddings, distance_matrix, visited)
            node_embeddings = node_embeddings + edge_messages * 0.1  # Soft edge influence
            
            # 2. Adapt graph structure
            adjacency = self.graph_adapter(node_embeddings, visited, remaining_capacity)
            
            # 3. Apply multi-scale temporal attention
            node_embeddings = node_embeddings + self.temporal_attention(node_embeddings, visited, step)
            
            # 4. Progressive refinement based on progress
            for refinement_layer in self.refinement_layers[:2]:  # Use only first 2 layers during decoding
                node_embeddings = refinement_layer(node_embeddings, progress)
            
            # === Enhanced State Encoding ===
            # Compute state features
            normalized_capacity = (remaining_capacity / capacities).unsqueeze(-1)
            capacity_used = 1 - normalized_capacity.squeeze(-1)
            visited_ratio = visited.float().sum(dim=1) / max_nodes
            distance_from_depot = torch.zeros(batch_size, device=device)
            
            for b in range(batch_size):
                current_pos = current_nodes[b].item()
                if current_pos < max_nodes:
                    distance_from_depot[b] = distance_matrix[b, current_pos, 0]
            
            state_features = torch.stack([
                capacity_used, 
                torch.full((batch_size,), progress, device=device),
                visited_ratio, 
                distance_from_depot
            ], dim=1)
            
            # Read from memory
            current_emb = node_embeddings[torch.arange(batch_size), current_nodes]
            memory_read, memory = self.memory_bank(current_emb, memory)
            
            # Enhanced state with memory
            state_input = torch.cat([current_emb, memory_read, state_features], dim=-1)
            state = self.state_encoder(state_input)
            
            # === Learned Update Schedule ===
            update_strength = self.update_schedule(torch.tensor([[progress]], device=device))
            
            # === Dynamic Node Embedding Updates ===
            if step > 0:
                # Compute update with graph structure awareness
                state_expanded = state.unsqueeze(1).expand(-1, max_nodes, -1)
                adjacency_weighted = torch.matmul(adjacency, node_embeddings)
                
                update_input = torch.cat([node_embeddings, state_expanded, adjacency_weighted], dim=-1)
                update = self.graph_updater(update_input)
                
                # Apply update with learned schedule
                node_embeddings = node_embeddings + update * update_strength
                
                # Light transformer pass for coherence
                for i in range(min(1, len(self.transformer_layers))):
                    node_embeddings = self.transformer_layers[i](node_embeddings, distance_matrix, visited)
            
            # === Enhanced Pointer Attention ===
            # Multi-head pointer with memory augmentation
            state_with_memory = torch.cat([state, memory_read], dim=-1)
            
            Q = self.pointer_attention['q_proj'](state_with_memory).unsqueeze(1)
            K = self.pointer_attention['k_proj'](node_embeddings)
            V = self.pointer_attention['v_proj'](node_embeddings)
            
            # Attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hidden_dim)
            
            # Pointer network scores
            global_context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            state_context = state.unsqueeze(1).expand(-1, max_nodes, -1)
            edge_context = edge_features.mean(dim=2)  # Aggregate edge features
            
            pointer_input = torch.cat([node_embeddings, global_context, edge_context], dim=-1)
            pointer_scores = self.pointer_attention['pointer'](pointer_input)
            
            # Combine scores
            combined_scores = attention_scores.squeeze(1) + pointer_scores.mean(dim=-1)
            
            # === Adaptive Temperature ===
            adaptive_temp = self.temperature_controller(state_features).squeeze(-1)
            effective_temp = temperature * adaptive_temp.mean()
            
            # === Mask Creation ===
            mask = self._create_mask(visited, demands, remaining_capacity,
                                   current_nodes, instances, batch_size, max_nodes)
            
            # Apply mask
            logits = combined_scores.masked_fill(mask, -1e9)
            
            # Sample or greedy selection
            probs = F.softmax(logits / effective_temp, dim=-1)
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


# Alias for compatibility
DynamicGraphTransformerNetwork = DynamicGraphTransformer
