"""
Legacy GAT model implementation matching the exact architecture from GAT_RL project.
This is a faithful reproduction for benchmarking purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any
from torch.distributions import Categorical


class EdgeGATConv(nn.Module):
    """
    Custom GAT layer that includes edge features in the computation of attention coefficients.
    Exact reproduction of legacy EdgeGATConv from GAT_RL/encoder/EdgeGATConv.py
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
        """Xavier initialization as in legacy"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with edge-aware attention.
        
        Args:
            x: Node features [batch_size * num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        # Project node features
        x = self.fc(x)
        
        # Manual message passing (simplified from torch_geometric)
        num_nodes = x.size(0)
        out = torch.zeros_like(x)
        
        # Process each edge
        for edge_idx in range(edge_index.size(1)):
            i = edge_index[0, edge_idx]
            j = edge_index[1, edge_idx]
            
            # Concatenate features for attention
            x_cat = torch.cat([x[i].unsqueeze(0), x[j].unsqueeze(0), 
                              edge_attr[edge_idx].unsqueeze(0)], dim=-1)
            
            # Compute attention coefficient
            alpha = self.att_vector(x_cat)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            
            # Apply dropout
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            # Aggregate messages (ensure proper dimensions)
            out[i] += (x[j] * alpha).squeeze(0)
        
        return out


class ResidualEdgeGATEncoder(nn.Module):
    """
    GAT encoder with residual connections and edge feature processing.
    Exact reproduction of legacy ResidualEdgeGATEncoder from GAT_RL/encoder/GAT_Encoder.py
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
        """Xavier initialization as in legacy"""
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
        # Add demand as node feature (legacy concatenates demand)
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
    Exact reproduction of legacy TransformerAttention from GAT_RL/decoder/TransformerAttention.py
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
        """Xavier initialization as in legacy"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, state_t: torch.Tensor, context: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            state_t: Current state [batch_size, 1, input_dim * cat]
            context: Context to attend to [batch_size, n_nodes, input_dim]
            mask: Attention mask [batch_size, n_nodes]
        
        Returns:
            Attention output [batch_size, hidden_dim]
        """
        batch_size, n_nodes, input_dim = context.size()
        
        # Compute Q, K, V with multi-head reshape
        Q = self.w(state_t).reshape(batch_size, 1, self.n_heads, -1)
        K = self.k(context).reshape(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).reshape(batch_size, n_nodes, self.n_heads, -1)
        
        # Transpose for multi-head attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Compute compatibility scores
        compatibility = self.norm * torch.matmul(Q, K.transpose(2, 3))
        compatibility = compatibility.squeeze(2)
        
        # Apply mask
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))
        
        # Compute attention scores
        scores = F.softmax(u_i, dim=-1)
        scores = scores.unsqueeze(2)
        
        # Apply attention to values
        out_put = torch.matmul(scores, V)
        out_put = out_put.squeeze(2).reshape(batch_size, self.hidden_dim)
        out_put = self.fc(out_put)
        
        return out_put


class PointerAttention(nn.Module):
    """
    Pointer attention layer with multi-head transformer.
    Exact reproduction of legacy PointerAttention from GAT_RL/decoder/PointerAttention.py
    """
    
    def __init__(self, n_heads: int, input_dim: int, hidden_dim: int):
        super(PointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = TransformerAttention(n_heads, 1, input_dim, hidden_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Xavier initialization as in legacy"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, state_t: torch.Tensor, context: torch.Tensor, 
                mask: torch.Tensor, T: float) -> torch.Tensor:
        """
        Compute pointer attention scores.
        
        Args:
            state_t: Current state [batch_size, 1, input_dim * 3]
            context: Node embeddings [batch_size, n_nodes, input_dim]
            mask: Feasibility mask [batch_size, n_nodes]
            T: Temperature for softmax
        
        Returns:
            Probability distribution [batch_size, n_nodes]
        """
        # Apply multi-head attention
        x = self.mhalayer(state_t, context, mask)
        
        batch_size, n_nodes, input_dim = context.size()
        Q = x.reshape(batch_size, 1, -1)
        K = self.k(context).reshape(batch_size, n_nodes, -1)
        
        # Compute compatibility scores
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))
        compatibility = compatibility.squeeze(1)
        
        # Apply tanh and scale (legacy specific)
        x = torch.tanh(compatibility)
        x = x * 10  # Scale by 10 as in legacy
        
        # Apply mask and compute softmax
        x = x.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(x / T, dim=-1)
        
        return scores


class GAT_Decoder(nn.Module):
    """
    GAT decoder with pointer attention for route generation.
    Exact reproduction of legacy GAT_Decoder from GAT_RL/decoder/GAT_Decoder.py
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GAT_Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Pointer attention with 8 heads as in legacy
        self.pointer = PointerAttention(8, input_dim, hidden_dim)
        
        # Linear layers for state processing
        self.fc = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)  # +1 for capacity
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Xavier initialization as in legacy"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, encoder_inputs: torch.Tensor, pool: torch.Tensor, 
                capacity: torch.Tensor, demand: torch.Tensor, 
                n_steps: int, T: float, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate routes using pointer attention.
        
        Args:
            encoder_inputs: Node embeddings [batch_size, n_nodes, hidden_dim]
            pool: Graph embedding [batch_size, hidden_dim]
            capacity: Vehicle capacities [batch_size, 1]
            demand: Node demands [batch_size, n_nodes]
            n_steps: Maximum steps
            T: Temperature
            greedy: Whether to use greedy selection
        
        Returns:
            actions: Selected nodes [batch_size, seq_len]
            log_p: Log probabilities [batch_size]
        """
        device = encoder_inputs.device
        batch_size = encoder_inputs.size(0)
        seq_len = encoder_inputs.size(1)
        
        # Initialize masks
        mask1 = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
        mask = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
        
        # Initialize dynamic state
        dynamic_capacity = capacity.expand(batch_size, -1).to(device)
        demands = demand.to(device)
        
        # Track visited nodes
        index = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Collect actions and log probabilities
        log_ps = []
        actions = []
        
        for i in range(n_steps):
            # Check if all nodes visited
            if not mask1[:, 1:].eq(0).any():
                break
            
            if i == 0:
                # Start from depot
                _input = encoder_inputs[:, 0, :]
            
            # Prepare decoder input with capacity
            decoder_input = torch.cat([_input, dynamic_capacity], -1)
            decoder_input = self.fc(decoder_input)
            pool_processed = self.fc1(pool.to(device))
            decoder_input = decoder_input + pool_processed
            
            # Update mask for first step
            if i == 0:
                mask, mask1 = self.update_mask(demands, dynamic_capacity, 
                                              index.unsqueeze(-1), mask1, i)
            
            # Get pointer probabilities
            p = self.pointer(decoder_input.unsqueeze(1), encoder_inputs, mask, T)
            
            # Sample or select greedily
            dist = Categorical(p)
            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()
            
            actions.append(index.data.unsqueeze(1))
            log_p = dist.log_prob(index)
            
            # Check if done
            is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
            log_p = log_p * (1. - is_done)
            
            log_ps.append(log_p.unsqueeze(1))
            
            # Update state
            dynamic_capacity = self.update_state(demands, dynamic_capacity, 
                                                index.unsqueeze(-1), capacity[0].item())
            mask, mask1 = self.update_mask(demands, dynamic_capacity, 
                                          index.unsqueeze(-1), mask1, i)
            
            # Get next input
            _input = torch.gather(
                encoder_inputs, 1,
                index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1, 
                                                        encoder_inputs.size(2))
            ).squeeze(1)
        
        # Concatenate results
        log_ps = torch.cat(log_ps, dim=1)
        actions = torch.cat(actions, dim=1)
        log_p = log_ps.sum(dim=1)
        
        return actions, log_p
    
    def update_mask(self, demands, dynamic_capacity, index, mask1, step):
        """Update mask based on capacity constraints"""
        batch_size = demands.size(0)
        mask = mask1.clone()
        
        # Can't visit nodes that exceed capacity
        for b in range(batch_size):
            for n in range(demands.size(1)):
                if demands[b, n] > dynamic_capacity[b]:
                    mask[b, n] = 1
        
        # Can't revisit nodes (except depot)
        if step > 0:
            for b in range(batch_size):
                mask1[b, index[b]] = 1
                mask[b, index[b]] = 1
        
        # Must return to depot when all visited or no capacity
        for b in range(batch_size):
            if mask[b, 1:].all():  # All customers masked
                mask[b, 0] = 0  # Allow depot
        
        return mask, mask1
    
    def update_state(self, demands, dynamic_capacity, index, max_capacity):
        """Update dynamic capacity after visiting nodes"""
        batch_size = demands.size(0)
        
        for b in range(batch_size):
            node = index[b].item()
            if node == 0:  # Depot
                dynamic_capacity[b] = max_capacity
            else:
                dynamic_capacity[b] -= demands[b, node]
        
        return dynamic_capacity


class LegacyGATModel(nn.Module):
    """
    Complete legacy GAT model with encoder and decoder.
    This is the exact architecture from the GAT_RL project.
    """
    
    def __init__(self, node_input_dim: int = 3, edge_input_dim: int = 1, 
                 hidden_dim: int = 128, edge_dim: int = 16, 
                 layers: int = 4, negative_slope: float = 0.2, 
                 dropout: float = 0.6, config: Optional[Dict] = None):
        super(LegacyGATModel, self).__init__()
        
        self.config = config or {}
        self.hidden_dim = hidden_dim
        
        # Initialize encoder and decoder as in legacy
        self.encoder = ResidualEdgeGATEncoder(
            node_input_dim, edge_input_dim, hidden_dim, 
            edge_dim, layers, negative_slope, dropout
        )
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)
    
    def forward(self, instances: List[Dict], max_steps: Optional[int] = None,
                temperature: Optional[float] = None, greedy: bool = False,
                config: Optional[Dict] = None) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """
        Forward pass matching current project interface.
        
        Args:
            instances: List of CVRP instances
            max_steps: Maximum decoding steps
            temperature: Softmax temperature
            greedy: Whether to use greedy decoding
            config: Configuration dictionary
        
        Returns:
            routes: List of routes
            log_probs: Log probabilities
            entropy: Entropy (placeholder for compatibility)
        """
        config = config or self.config
        device = next(self.parameters()).device
        batch_size = len(instances)
        
        # Set defaults
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config.get('inference', {}).get('max_steps_multiplier', 2)
        if temperature is None:
            temperature = config.get('inference', {}).get('default_temperature', 1.0)
        
        # Prepare batch data
        max_nodes = max(len(inst['coords']) for inst in instances)
        
        # Node features (coordinates)
        node_coords = torch.zeros(batch_size * max_nodes, 2, device=device)
        demands = torch.zeros(batch_size * max_nodes, 1, device=device)
        capacities = torch.zeros(batch_size, 1, device=device)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            start_idx = i * max_nodes
            end_idx = start_idx + n_nodes
            
            node_coords[start_idx:end_idx] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            demands[start_idx:end_idx, 0] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            capacities[i] = inst['capacity']
        
        # Create edge index and attributes (fully connected graph)
        edge_index_list = []
        edge_attr_list = []
        
        for i in range(batch_size):
            n_nodes = len(instances[i]['coords'])
            offset = i * max_nodes
            
            # Create fully connected graph for this instance
            for src in range(n_nodes):
                for dst in range(n_nodes):
                    if src != dst:
                        edge_index_list.append([offset + src, offset + dst])
                        
                        # Compute edge distance
                        src_coord = node_coords[offset + src]
                        dst_coord = node_coords[offset + dst]
                        dist = torch.norm(dst_coord - src_coord, dim=-1, keepdim=True)
                        edge_attr_list.append(dist)
        
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, device=device).t()
            edge_attr = torch.stack(edge_attr_list)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.zeros(0, 1, device=device)
        
        # Encode graph
        node_embeddings = self.encoder(node_coords, edge_index, edge_attr, demands, batch_size)
        
        # Compute graph embedding (mean pooling)
        graph_embedding = node_embeddings.mean(dim=1)
        
        # Prepare demands and capacity for decoder
        demand_matrix = demands.reshape(batch_size, max_nodes)
        
        # Decode routes
        actions, log_p = self.decoder(
            node_embeddings, graph_embedding, capacities,
            demand_matrix, max_steps, temperature, greedy
        )
        
        # Convert actions to routes format
        routes = []
        for b in range(batch_size):
            route = [0]  # Start at depot
            for step in range(actions.size(1)):
                node = actions[b, step].item()
                route.append(node)
                if node == 0:  # Return to depot
                    break
            routes.append(route)
        
        # Calculate dummy entropy for interface compatibility
        entropy = torch.zeros(batch_size, device=device)
        
        return routes, log_p, entropy
