"""
Simplified Dynamic Graph Transformer with Reinforcement Learning for small CVRP instances.

This model reduces complexity compared to the full DGT_RL while maintaining core dynamic features:
- Simplified attention mechanism
- Reduced embedding dimensions  
- Fewer parameters for better performance on small instances
- Conservative dynamic updates to avoid instability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math


class SimplifiedDynamicGraphTransformer(nn.Module):
    """
    Simplified Dynamic Graph Transformer for small CVRP instances.
    
    Key simplifications:
    - Single attention head for reduced complexity
    - Smaller embedding dimensions
    - Conservative residual connections
    - Simplified positional encoding
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract simplified config parameters
        self.embed_dim = config.get('embed_dim', 64)  # Reduced from typical 128
        self.n_heads = min(config.get('n_heads', 1), 2)  # Max 2 heads for small problems
        self.n_layers = min(config.get('n_layers', 2), 3)  # Max 3 layers
        self.ff_hidden_dim = config.get('ff_hidden_dim', 128)  # Reduced feedforward
        
        # Dynamic features - more conservative
        self.use_dynamic_embedding = config.get('use_dynamic_embedding', True)
        self.residual_gate_init = config.get('residual_gate_init', 0.1)  # More conservative
        self.adaptive_temp = config.get('adaptive_temperature', False)
        
        # Problem-specific parameters
        self.node_dim = 2  # (x, y) coordinates
        self.vehicle_capacity = config.get('vehicle_capacity', 15)
        
        # Embedding layers
        self.node_embedding = nn.Linear(self.node_dim, self.embed_dim)
        self.depot_embedding = nn.Linear(self.node_dim, self.embed_dim)
        
        # Dynamic state embedding (simplified)
        if self.use_dynamic_embedding:
            self.state_embedding = nn.Linear(3, self.embed_dim // 4)  # capacity, steps, distance
            self.state_gate = nn.Parameter(torch.tensor(self.residual_gate_init))
        
        # Simplified positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(50, self.embed_dim) * 0.1)  # Max 50 nodes
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimplifiedTransformerLayer(
                embed_dim=self.embed_dim,
                n_heads=self.n_heads,
                ff_hidden_dim=self.ff_hidden_dim,
                dropout=config.get('dropout', 0.1)
            ) for _ in range(self.n_layers)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.embed_dim // 2, 1)
        )
        
        # Pointer mechanism (simplified)
        self.pointer_net = SimplifiedPointerNetwork(
            embed_dim=self.embed_dim,
            hidden_dim=config.get('pointer_hidden_dim', 64)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Conservative parameter initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Conservative gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, instances, max_steps=None, temperature=None, greedy=False, config=None):
        """
        Forward pass matching the expected interface.
        
        Args:
            instances: List of problem instances with coords, demands, capacity
            max_steps: Maximum number of construction steps
            temperature: Sampling temperature
            greedy: Whether to use greedy decoding
            config: Configuration dictionary
            
        Returns:
            routes: List of constructed routes
            log_probs: Log probabilities of selected actions
            entropy: Action entropy for each instance
        """
        batch_size = len(instances)
        
        # Set defaults
        if max_steps is None:
            max_steps = max(len(inst['coords']) for inst in instances) * 3
        if temperature is None:
            temperature = 1.0
        if config is not None:
            if max_steps is None:
                max_steps = max(len(inst['coords']) for inst in instances) * config['inference']['max_steps_multiplier']
            if temperature is None:
                temperature = config['inference']['default_temperature']
        
        # Convert instances to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3)  # x, y, demand
        demands_batch = torch.zeros(batch_size, max_nodes)
        capacities = torch.zeros(batch_size)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']
        
        # Extract coordinates for embedding
        coords = node_features[:, :, :2]  # [batch_size, max_nodes, 2]
        
        # Node embeddings
        depot_emb = self.depot_embedding(coords[:, 0:1])  # [batch_size, 1, embed_dim]
        customer_emb = self.node_embedding(coords[:, 1:])  # [batch_size, n_customers, embed_dim]
        
        # Combine embeddings
        node_emb = torch.cat([depot_emb, customer_emb], dim=1)  # [batch_size, max_nodes, embed_dim]
        
        # Add positional encoding
        if max_nodes <= self.pos_encoding.size(0):
            pos_enc = self.pos_encoding[:max_nodes].unsqueeze(0).expand(batch_size, -1, -1)
            node_emb = node_emb + pos_enc
        
        # Transformer layers
        hidden = node_emb
        for layer in self.layers:
            hidden = layer(hidden)
        
        # Final normalization
        hidden = self.final_norm(hidden)
        
        # Generate routes using the simplified route construction
        return self._generate_routes_simplified(
            hidden, node_features, demands_batch, capacities, 
            max_steps, temperature, greedy, instances, config
        )
    
    def _generate_routes_simplified(self, node_embeddings, node_features, demands_batch, capacities, 
                                  max_steps, temperature, greedy, instances, config):
        """Simplified route generation based on the standard interface."""
        from typing import List
        
        batch_size, max_nodes, _ = node_embeddings.shape
        routes: List[List[int]] = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        
        # Initialize state
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        # Start all routes at depot
        for b in range(batch_size):
            routes[b].append(0)
        
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        # Route construction loop
        for step in range(max_steps):
            # Check if all routes are complete
            for b in range(batch_size):
                if not batch_done[b]:
                    n_customers = len(instances[b]['coords']) - 1
                    customers_visited = visited[b, 1:n_customers+1].all() if n_customers > 0 else True
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            if batch_done.all():
                break
            
            # Get action probabilities using pointer network
            # Create simple context (mean of all embeddings)
            context = node_embeddings.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            context_expanded = context.expand(-1, max_nodes, -1)  # [batch_size, max_nodes, embed_dim]
            
            # Concatenate for pointer input (simplified version)
            pointer_input = torch.cat([node_embeddings, context_expanded], dim=-1)  # [batch_size, max_nodes, 2*embed_dim]
            
            # Simplified scoring (just use a linear layer)
            scores = torch.sum(node_embeddings * context_expanded, dim=-1)  # [batch_size, max_nodes]
            
            # Create mask for invalid actions
            mask = self._create_action_mask(
                visited, demands_batch, remaining_capacity, 
                routes, instances, batch_size, max_nodes
            )
            
            # Apply mask to scores
            masked_score_value = -1e9 if config is None else config.get('inference', {}).get('masked_score_value', -1e9)
            scores = scores.masked_fill(mask, masked_score_value)
            
            # Apply temperature and compute probabilities
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Compute log probabilities and entropy
            log_prob_epsilon = 1e-8 if config is None else config.get('inference', {}).get('log_prob_epsilon', 1e-8)
            log_probs = torch.log(probs + log_prob_epsilon)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Select actions
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
            
            # Execute actions
            for b in range(batch_size):
                if not batch_done[b]:
                    action = int(actions[b].item())
                    routes[b].append(action)
                    
                    if action == 0:  # Return to depot
                        remaining_capacity[b] = capacities[b]
                    else:  # Visit customer
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
        
        # Ensure all routes end at depot
        for b in range(batch_size):
            if len(routes[b]) == 0 or routes[b][-1] != 0:
                routes[b].append(0)
        
        # Combine log probabilities and entropy across steps
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        
        return routes, combined_log_probs, combined_entropy
    
    def _create_action_mask(self, visited, demands_batch, remaining_capacity, routes, instances, batch_size, max_nodes):
        """Create mask for invalid actions."""
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        for b in range(batch_size):
            n_nodes = len(instances[b]['coords'])
            
            # Mask padding nodes
            if n_nodes < max_nodes:
                mask[b, n_nodes:] = True
            
            # Mask visited customers (but not depot)
            mask[b, 1:] = mask[b, 1:] | visited[b, 1:]
            
            # Mask customers that exceed capacity
            capacity_mask = demands_batch[b] > remaining_capacity[b]
            mask[b] = mask[b] | capacity_mask
            
            # Allow depot if not currently there or if all customers visited or capacity constraint
            currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
            n_customers = n_nodes - 1
            customers_visited = visited[b, 1:n_customers+1].all() if n_customers > 0 else True
            
            # Depot logic
            if currently_at_depot and not customers_visited:
                # If at depot and customers remain unvisited, can't stay at depot
                mask[b, 0] = True
            else:
                # Allow returning to depot
                mask[b, 0] = False
        
        return mask
    
    def _apply_dynamic_updates(self, embeddings: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """Apply conservative dynamic updates to embeddings."""
        if not self.use_dynamic_embedding:
            return embeddings
            
        # Extract state information
        remaining_capacity = state.get('remaining_capacity', torch.ones(embeddings.size(0), 1))
        step_ratio = state.get('step_ratio', torch.zeros(embeddings.size(0), 1))
        avg_distance = state.get('avg_distance_to_depot', torch.ones(embeddings.size(0), 1))
        
        # Normalize state features
        capacity_norm = remaining_capacity / self.vehicle_capacity
        distance_norm = torch.clamp(avg_distance / 10.0, 0, 1)  # Assume max distance ~10
        
        # Create state vector
        state_vec = torch.cat([
            capacity_norm,
            step_ratio.unsqueeze(-1) if step_ratio.dim() == 1 else step_ratio,
            distance_norm
        ], dim=-1)  # [batch_size, 3]
        
        # Generate state embedding
        state_emb = self.state_embedding(state_vec)  # [batch_size, embed_dim//4]
        
        # Conservative gating - only update a small portion of the embedding
        gate_weight = torch.sigmoid(self.state_gate)
        
        # Apply update only to the first quarter of embedding dimensions
        update_dim = self.embed_dim // 4
        update = state_emb.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        
        # Conservative residual update
        embeddings_updated = embeddings.clone()
        embeddings_updated[:, :, :update_dim] = (
            (1 - gate_weight) * embeddings[:, :, :update_dim] + 
            gate_weight * update
        )
        
        return embeddings_updated
    
    def _get_attention_mask(self, hidden: torch.Tensor, state: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        """Generate attention mask based on current state."""
        if state is None:
            return None
            
        batch_size, n_nodes = hidden.shape[:2]
        
        # Create mask for visited nodes
        visited = state.get('visited', torch.zeros(batch_size, n_nodes, dtype=torch.bool))
        
        # Allow attention to all nodes, but mask will be applied in pointer network
        return None  # Simplified - no attention masking
    
    def sample_action(self, x: torch.Tensor, state: Dict[str, Any], 
                      temperature: float = 1.0, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        action_probs, value = self.forward(x, state)
        
        if greedy:
            # Apply mask and select best valid action
            mask = state.get('action_mask', torch.ones_like(action_probs, dtype=torch.bool))
            masked_probs = action_probs.masked_fill(~mask, -float('inf'))
            action = masked_probs.argmax(dim=-1)
            log_prob = F.log_softmax(masked_probs / temperature, dim=-1).gather(-1, action.unsqueeze(-1))
        else:
            # Sample from distribution
            mask = state.get('action_mask', torch.ones_like(action_probs, dtype=torch.bool))
            masked_probs = action_probs.masked_fill(~mask, -float('inf'))
            
            # Apply temperature
            logits = masked_probs / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample action
            action = torch.multinomial(probs, 1).squeeze(-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1))
        
        return action, log_prob.squeeze(-1)


class SimplifiedTransformerLayer(nn.Module):
    """Simplified transformer layer with single or dual attention heads."""
    
    def __init__(self, embed_dim: int, n_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class SimplifiedPointerNetwork(nn.Module):
    """Simplified pointer network for action selection."""
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Pointer mechanism
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.value_proj = nn.Linear(embed_dim, 1)
        
        # Context vector (learned)
        self.context = nn.Parameter(torch.randn(hidden_dim) * 0.1)
    
    def forward(self, embeddings: torch.Tensor, state: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Generate action probabilities using pointer mechanism.
        
        Args:
            embeddings: Node embeddings [batch_size, n_nodes, embed_dim]
            state: Current state information
            
        Returns:
            action_probs: Action probabilities [batch_size, n_nodes]
        """
        batch_size, n_nodes, _ = embeddings.shape
        
        # Project embeddings
        queries = self.query_proj(embeddings)  # [batch_size, n_nodes, hidden_dim]
        keys = self.key_proj(embeddings)       # [batch_size, n_nodes, hidden_dim]
        
        # Use context vector as query for all nodes
        context = self.context.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        
        # Compute attention scores
        scores = torch.matmul(queries, context.transpose(-2, -1)).squeeze(-1)  # [batch_size, n_nodes]
        
        # Apply scaling
        scores = scores / math.sqrt(self.hidden_dim)
        
        # Apply masking if available
        if state is not None and 'action_mask' in state:
            mask = state['action_mask']  # [batch_size, n_nodes]
            scores = scores.masked_fill(~mask, -float('inf'))
        
        # Convert to probabilities
        action_probs = F.softmax(scores, dim=-1)
        
        return action_probs


def create_simplified_dgt_rl_model(config: Dict[str, Any]) -> SimplifiedDynamicGraphTransformer:
    """Factory function to create a simplified DGT-RL model."""
    
    # Override config for small problem instances
    small_config = config.copy()
    small_config.update({
        'embed_dim': min(config.get('embed_dim', 128), 64),
        'n_heads': min(config.get('n_heads', 8), 2),
        'n_layers': min(config.get('n_layers', 6), 3),
        'ff_hidden_dim': min(config.get('ff_hidden_dim', 512), 128),
        'pointer_hidden_dim': min(config.get('pointer_hidden_dim', 128), 64),
        'dropout': max(config.get('dropout', 0.1), 0.1),  # Ensure some dropout
        'residual_gate_init': 0.1,  # Conservative initialization
    })
    
    return SimplifiedDynamicGraphTransformer(small_config)


# Model registration for easy access
MODEL_REGISTRY = {
    'simplified_dgt_rl': create_simplified_dgt_rl_model
}


if __name__ == '__main__':
    # Test the simplified model
    config = {
        'embed_dim': 64,
        'n_heads': 2,
        'n_layers': 3,
        'ff_hidden_dim': 128,
        'dropout': 0.1,
        'vehicle_capacity': 15,
    }
    
    model = create_simplified_dgt_rl_model(config)
    
    # Test forward pass
    batch_size, n_nodes = 4, 16  # 15 customers + 1 depot
    x = torch.randn(batch_size, n_nodes, 2)
    
    state = {
        'remaining_capacity': torch.rand(batch_size, 1) * 15,
        'step_ratio': torch.rand(batch_size),
        'avg_distance_to_depot': torch.rand(batch_size, 1) * 5,
        'action_mask': torch.ones(batch_size, n_nodes, dtype=torch.bool)
    }
    
    action_probs, value = model(x, state)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test action sampling
    action, log_prob = model.sample_action(x, state, temperature=1.0, greedy=False)
    print(f"Sampled action shape: {action.shape}")
    print(f"Log probability shape: {log_prob.shape}")
