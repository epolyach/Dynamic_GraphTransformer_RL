"""
GAT Model for CVRP with improved convergence.

This version includes:
1. Fixed pointer attention without aggressive tanh + 10x scaling
2. Proper gradient flow management
3. Standard attention scaling
4. Edge-aware GAT layers
5. Residual connections and batch normalization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional, Any


class EdgeGATConv(nn.Module):
    """
    Custom GAT layer that includes edge features in the computation of attention coefficients.
    """
    
    def __init__(self, node_channels: int, hidden_dim: int, edge_dim: int, 
                 negative_slope: float = 0.2, dropout: float = 0.6):
        super(EdgeGATConv, self).__init__()
        self.node_channels = node_channels
        self.hidden_dim = hidden_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.fc = nn.Linear(node_channels, hidden_dim)
        self.att_vector = nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Xavier initialization"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with edge-aware attention (vectorized for efficiency).
        
        Args:
            x: Node features [batch_size * num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        # Project node features
        x = self.fc(x)
        
        if edge_index.size(1) == 0:
            return x  # No edges, return projected features
        
        # Vectorized message passing
        row, col = edge_index[0], edge_index[1]
        
        # Get source and target node features for all edges at once
        x_i = x[row]  # [num_edges, hidden_dim]
        x_j = x[col]  # [num_edges, hidden_dim]
        
        # Concatenate features for attention computation
        x_cat = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [num_edges, hidden_dim*2 + edge_dim]
        
        # Compute attention coefficients for all edges
        alpha = self.att_vector(x_cat)  # [num_edges, hidden_dim]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Prepare messages
        messages = x_j * alpha  # [num_edges, hidden_dim]
        
        # Aggregate messages (sum over incoming edges for each node)
        num_nodes = x.size(0)
        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)
        
        return out


class ResidualEdgeGATEncoder(nn.Module):
    """
    GAT encoder with residual connections and edge feature processing.
    """
    
    def __init__(self, node_input_dim: int, edge_input_dim: int, hidden_dim: int, 
                 edge_dim: int, layers: int = 4, negative_slope: float = 0.2, 
                 dropout: float = 0.6):
        super(ResidualEdgeGATEncoder, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.layers = layers
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Input projections with batch norm
        self.fc_node = nn.Linear(node_input_dim, hidden_dim)
        self.fc_edge = nn.Linear(edge_input_dim, edge_dim)
        self.bn_node = nn.BatchNorm1d(hidden_dim)
        self.bn_edge = nn.BatchNorm1d(edge_dim)
        
        # Stack of EdgeGATConv layers
        self.edge_gat_layers = nn.ModuleList([
            EdgeGATConv(hidden_dim, hidden_dim, edge_dim, negative_slope, dropout) 
            for _ in range(layers)
        ])
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Xavier initialization"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        if hasattr(self.fc_edge, 'bias'):
            nn.init.constant_(self.fc_edge.bias, 0)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, demand: torch.Tensor, 
                batch_size: int) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Node coordinates [batch_size * num_nodes, 2]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge distances [num_edges, 1]
            demand: Node demands [batch_size * num_nodes, 1]
            batch_size: Number of graphs in batch
        
        Returns:
            Node embeddings [batch_size, num_nodes, hidden_dim]
        """
        # Add demand as node feature (concatenates demand)
        x = torch.cat([x, demand], dim=-1)
        
        # Node and edge embedding with batch norm
        x = self.fc_node(x)
        # Handle batch norm dimension
        if x.size(0) > 1:
            x = self.bn_node(x)
        
        if edge_attr.numel() > 0:
            edge_attr = self.fc_edge(edge_attr)
            if edge_attr.size(0) > 1:
                edge_attr = self.bn_edge(edge_attr)
        
        # Apply EdgeGAT layers with residual connections
        for edge_gat_layer in self.edge_gat_layers:
            x_next = edge_gat_layer(x, edge_index, edge_attr)
            x = x + x_next  # Residual connection
        
        # Reshape to batch format
        num_nodes = x.size(0) // batch_size
        x = x.reshape(batch_size, num_nodes, self.hidden_dim)
        
        return x


class TransformerAttention(nn.Module):
    """
    Multi-head transformer attention for state processing.
    """
    
    def __init__(self, n_heads: int, cat: int, input_dim: int, hidden_dim: int, 
                 attn_dropout: float = 0.1, dropout: float = 0):
        super(TransformerAttention, self).__init__()
        
        # Assert hidden_dim divisible by n_heads
        if hidden_dim % n_heads != 0:
            raise ValueError(f'hidden_dim({hidden_dim}) should be divisible by n_heads({n_heads}).')
        
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)
        
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        
        # Linear projections
        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Xavier initialization"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Multi-head attention forward pass.
        
        Args:
            x: Query tensor [batch_size, n_queries, input_dim * cat]
            y: Key/Value tensor [batch_size, n_keys, input_dim]
            mask: Attention mask [batch_size, n_keys]
        
        Returns:
            Attention output [batch_size, n_queries, hidden_dim]
        """
        batch_size, n_queries, _ = x.size()
        n_keys = y.size(1)
        
        # Project to Q, K, V
        Q = self.w(x).reshape(batch_size, n_queries, self.n_heads, self.head_dim)
        K = self.k(y).reshape(batch_size, n_keys, self.n_heads, self.head_dim)
        V = self.v(y).reshape(batch_size, n_keys, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, n_heads, n_queries, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.norm
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, n_queries, -1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        # Compute attention weights
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().reshape(batch_size, n_queries, self.hidden_dim)
        out = self.fc(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out


class ImprovedPointerAttention(nn.Module):
    """
    Fixed pointer attention mechanism with better gradient flow.
    Removes the problematic tanh + 10x scaling that causes gradient saturation.
    """
    
    def __init__(self, n_heads: int, input_dim: int, hidden_dim: int):
        super(ImprovedPointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Standard scaling factor (like in transformers)
        self.scale = 1 / math.sqrt(hidden_dim)
        
        # Linear projections
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        # cat=3 because state_t has 3 concatenated hidden_dim vectors
        self.mhalayer = TransformerAttention(n_heads, 3, input_dim, hidden_dim)
        
        # Optional: Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights with better defaults for transformers."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # Use Xavier uniform but with gain for ReLU
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, state_t: torch.Tensor, context: torch.Tensor, 
                mask: torch.Tensor, T: float) -> torch.Tensor:
        """
        Compute pointer attention scores with improved scaling.
        
        Args:
            state_t: Current state [batch_size, 1, input_dim * 3]
            context: Node embeddings [batch_size, n_nodes, input_dim]
            mask: Feasibility mask [batch_size, n_nodes]
            T: Temperature for softmax
        
        Returns:
            Probability distribution [batch_size, n_nodes]
        """
        # Apply multi-head attention
        x = self.mhalayer(state_t, context, mask)  # [batch_size, 1, hidden_dim]
        
        batch_size, n_nodes, input_dim = context.size()
        
        # Apply layer norm for stability (optional but helpful)
        # x has shape [batch_size, 1, hidden_dim], squeeze the middle dimension for layer norm
        x_squeezed = x.squeeze(1)  # [batch_size, hidden_dim]
        x_norm = self.layer_norm(x_squeezed)  # [batch_size, hidden_dim]
        Q = x_norm.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Project keys
        K = self.k(context).reshape(batch_size, n_nodes, -1)
        
        # Compute scaled dot-product attention (standard transformer approach)
        # This replaces the problematic tanh(compatibility) * 10
        compatibility = torch.matmul(Q, K.transpose(1, 2)) * self.scale
        compatibility = compatibility.squeeze(1)
        
        # Apply mask
        compatibility = compatibility.masked_fill(mask.bool(), float("-inf"))
        
        # Handle edge case where all values are masked
        all_masked = (compatibility == float("-inf")).all(dim=-1, keepdim=True)
        if all_masked.any():
            # If all are masked, unmask the depot (index 0) as fallback
            compatibility = torch.where(
                all_masked.expand_as(compatibility),
                torch.zeros_like(compatibility),
                compatibility
            )
            compatibility[:, 0] = torch.where(
                all_masked.squeeze(-1),
                torch.zeros_like(compatibility[:, 0]),
                compatibility[:, 0]
            )
        
        # Apply temperature-controlled softmax
        scores = F.softmax(compatibility / T, dim=-1)
        
        # Safety check for NaN (should be rare with the fixes)
        if torch.isnan(scores).any():
            # Fallback: uniform distribution over unmasked nodes
            valid_mask = ~mask.bool()
            scores = valid_mask.float()
            scores = scores / scores.sum(dim=-1, keepdim=True)
        
        return scores


class GATDecoder(nn.Module):
    """
    GAT decoder with improved gradient flow and stability.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GATDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Use improved pointer attention
        self.pointer = ImprovedPointerAttention(8, input_dim, hidden_dim)
        
        # Linear layers for state processing
        self.fc = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)  # +1 for capacity
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Optional: Add residual connections and layer norm for stability
        self.use_residual = True
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights appropriately."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, encoder_inputs: torch.Tensor, pool: torch.Tensor, 
                capacity: torch.Tensor, demand: torch.Tensor, 
                max_steps: int, temperature: float, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate routes using pointer attention with improved stability.
        
        Args:
            encoder_inputs: Node embeddings [batch_size, n_nodes, hidden_dim]
            pool: Graph embedding [batch_size, hidden_dim]
            capacity: Vehicle capacities [batch_size, 1]
            demand: Node demands [batch_size, n_nodes]
            max_steps: Maximum steps
            temperature: Temperature for sampling
            greedy: Whether to use greedy selection
        
        Returns:
            actions: Selected nodes [batch_size, seq_len]
            log_p: Log probabilities [batch_size]
        """
        device = encoder_inputs.device
        batch_size = encoder_inputs.size(0)
        seq_len = encoder_inputs.size(1)
        
        # Initialize visited mask to track which nodes have been visited
        visited = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Initialize dynamic state
        max_capacity = capacity.clone()
        dynamic_capacity = capacity.clone().expand(batch_size, -1).to(device)
        demands = demand.to(device)
        
        # Track current node index
        index = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Collect actions and log probabilities
        log_ps = []
        actions = []
        
        # IMPORTANT: Add depot (0) as the first action to ensure route starts at depot
        actions.append(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        
        _input = encoder_inputs[:, 0, :]  # Start from depot
        
        for i in range(max_steps):
            # Create mask for infeasible nodes
            # Mask nodes that exceed capacity
            capacity_mask = demands > dynamic_capacity.expand_as(demands)
            
            # Combine with visited mask (don't revisit customers, but can return to depot)
            mask = visited.clone()
            mask[:, 0] = False  # Depot is initially available
            
            # Apply capacity constraints
            mask = mask | capacity_mask
            
            # Prevent consecutive depot visits when feasible customers exist
            for b in range(batch_size):
                # Check if previous action was depot
                if index[b] == 0:
                    # Check if any customers are feasible (unvisited and within capacity)
                    any_feasible = False
                    for n in range(1, seq_len):
                        if not visited[b, n] and demands[b, n] <= dynamic_capacity[b, 0]:
                            any_feasible = True
                            break
                    # If feasible customers exist, prevent returning to depot immediately
                    if any_feasible:
                        mask[b, 0] = True  # Mask depot to avoid consecutive depot visits
            
            # If all customers are visited, only allow depot
            customers_visited = visited[:, 1:].all(dim=1)
            for b in range(batch_size):
                if customers_visited[b]:
                    mask[b, 1:] = True  # Mask all customers
                    mask[b, 0] = False  # Only depot available
            
            # State preparation
            _input = torch.cat([_input, dynamic_capacity / max_capacity], dim=1)
            _input = self.fc(_input)
            
            if self.use_residual:
                _input = _input + pool  # Residual connection
            
            # Prepare state for attention
            # Gather current node embeddings and ensure correct dimensions
            current_embeds = encoder_inputs[torch.arange(batch_size), index]  # [batch_size, hidden_dim]
            # Concatenate along last dimension to form [batch_size, 1, hidden_dim*3]
            state_t = torch.cat([_input, encoder_inputs[:, 0, :], current_embeds], dim=-1).unsqueeze(1)
            
            # Get probabilities from pointer attention
            probs = self.pointer(state_t, encoder_inputs, mask, temperature)
            
            # Sample or select greedily
            if greedy:
                action = probs.argmax(dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
                log_ps.append(dist.log_prob(action))
            
            actions.append(action.unsqueeze(1))
            
            # Update state
            index = action
            _input = encoder_inputs[torch.arange(batch_size), index]
            
            # Mark nodes as visited (except depot)
            for b in range(batch_size):
                node_idx = action[b].item()
                if node_idx != 0:  # Don't mark depot as visited
                    visited[b, node_idx] = True
            
            # Update capacity
            dynamic_capacity = dynamic_capacity - demands[torch.arange(batch_size), index].unsqueeze(1)
            
            # Reset capacity when returning to depot
            dynamic_capacity = torch.where(
                (index == 0).unsqueeze(1),
                max_capacity,
                dynamic_capacity
            )
            
            # Check if all instances have completed their routes  
            # An instance is complete when all its customers are visited and it's at the depot
            all_done = True
            for b in range(batch_size):
                if not (visited[b, 1:].all() and index[b] == 0):
                    all_done = False
                    break
            if all_done:
                break
        
        # Stack actions and log probabilities
        actions = torch.cat(actions, dim=1)
        
        if not greedy:
            log_p = torch.stack(log_ps, dim=1).sum(dim=1)
        else:
            log_p = torch.zeros(batch_size, device=device)
        
        return actions, log_p


class GATModel(nn.Module):
    """
    Complete GAT model for CVRP with encoder-decoder architecture.
    """
    
    def __init__(self, node_input_dim: int, edge_input_dim: int, hidden_dim: int,
                 edge_dim: int = 16, layers: int = 4, negative_slope: float = 0.2,
                 dropout: float = 0.6, config: Dict[str, Any] = None):
        super(GATModel, self).__init__()
        
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.config = config or {}
        
        # Encoder
        self.encoder = ResidualEdgeGATEncoder(
            node_input_dim, edge_input_dim, hidden_dim,
            edge_dim, layers, negative_slope, dropout
        )
        
        # Decoder
        self.decoder = GATDecoder(hidden_dim, hidden_dim)
        
        # Graph pooling
        self.pool_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, instances: List[Dict], max_steps: Optional[int] = None,
                temperature: Optional[float] = None, greedy: bool = False,
                config: Optional[Dict] = None) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model with standard interface.
        
        Args:
            instances: List of instance dictionaries with 'coords', 'demands', 'capacity'
            max_steps: Maximum decoding steps
            temperature: Temperature for sampling
            greedy: Whether to use greedy decoding
            config: Configuration dictionary
        
        Returns:
            routes: List of routes for each instance
            log_p: Log probabilities [batch_size]
            entropy: Entropy values [batch_size]
        """
        config = config or {}
        batch_size = len(instances)
        
        if batch_size == 0:
            return [], torch.tensor([]), torch.tensor([])
        
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config.get('inference', {}).get('max_steps_multiplier', 2)
        if temperature is None:
            temperature = config.get('inference', {}).get('default_temperature', 1.0)
        
        # Prepare data and call the original forward method
        device = next(self.parameters()).device
        
        # Convert instances to tensors needed by GAT
        max_nodes = max(len(inst['coords']) for inst in instances)
        total_nodes = batch_size * max_nodes
        
        # Prepare node features (coordinates)
        x = torch.zeros(total_nodes, 2, device=device)
        demand = torch.zeros(total_nodes, 1, device=device)
        capacity = torch.zeros(batch_size, 1, device=device)
        
        # Create edge index for fully connected graph per instance
        edge_list = []
        edge_attr_list = []
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            offset = i * max_nodes
            
            # Fill node features
            coords_tensor = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            x[offset:offset+n_nodes] = coords_tensor
            
            demands_tensor = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            demand[offset:offset+n_nodes, 0] = demands_tensor
            
            capacity[i] = inst['capacity']
            
            # Create fully connected edges for this instance
            for j in range(n_nodes):
                for k in range(n_nodes):
                    if j != k:
                        edge_list.append([offset + j, offset + k])
                        # Compute distance as edge attribute
                        dist = torch.norm(coords_tensor[j] - coords_tensor[k])
                        edge_attr_list.append(dist)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32, device=device).unsqueeze(-1)
        
        # Call the original GAT forward
        actions, log_p = self._gat_forward(
            x, edge_index, edge_attr, demand, capacity,
            max_steps, temperature, greedy, batch_size
        )
        
        # Convert actions to routes format
        routes = []
        for b in range(batch_size):
            route = actions[b].cpu().numpy().tolist()
            # Clean up route - remove consecutive depot visits and trailing depots
            clean_route = []
            prev_node = -1
            for node in route:
                if node >= len(instances[b]['coords']):
                    continue  # Skip padding
                # Skip consecutive depot visits
                if node == 0 and prev_node == 0:
                    continue
                clean_route.append(node)
                prev_node = node
            
            # Ensure route ends at depot
            if len(clean_route) == 0 or clean_route[-1] != 0:
                clean_route.append(0)
            routes.append(clean_route)
        
        # Compute entropy (placeholder for consistency)
        entropy = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        return routes, log_p, entropy
    
    def _gat_forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                     demand: torch.Tensor, capacity: torch.Tensor, max_steps: int,
                     temperature: float, greedy: bool, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Original GAT forward pass.
        """
        # Encode
        node_embeddings = self.encoder(x, edge_index, edge_attr, demand, batch_size)
        
        # Pool for graph-level representation
        pool = self.pool_layer(node_embeddings.mean(dim=1))
        
        # Reshape demand for decoder
        num_nodes = node_embeddings.size(1)
        total_nodes = demand.size(0)
        expected_nodes = batch_size * num_nodes
        
        # Handle potential padding issues
        if total_nodes != expected_nodes:
            # Pad or truncate demand to match expected size
            demand_padded = torch.zeros(expected_nodes, 1, device=demand.device, dtype=demand.dtype)
            actual_size = min(total_nodes, expected_nodes)
            demand_padded[:actual_size] = demand[:actual_size]
            demand = demand_padded
        
        demand_reshaped = demand.reshape(batch_size, num_nodes)
        
        # Decode
        actions, log_p = self.decoder(
            node_embeddings, pool, capacity, demand_reshaped,
            max_steps, temperature, greedy
        )
        
        return actions, log_p
