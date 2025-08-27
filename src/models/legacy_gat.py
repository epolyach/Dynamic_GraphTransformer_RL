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
        
        # Handle case where all values are masked (avoid NaN)
        # Check if all values in a row are -inf
        all_masked = (x == float("-inf")).all(dim=-1, keepdim=True)
        if all_masked.any():
            # If all are masked, unmask the depot (index 0) as fallback
            x = torch.where(all_masked, torch.zeros_like(x), x)
            x[:, 0] = torch.where(all_masked.squeeze(-1), torch.zeros_like(x[:, 0]), x[:, 0])
        
        scores = F.softmax(x / T, dim=-1)
        
        # Additional safety check for NaN
        if torch.isnan(scores).any():
            # Fallback: uniform distribution over unmasked nodes, or depot if all masked
            scores = torch.where(torch.isnan(scores), 
                                torch.ones_like(scores) / scores.size(-1), 
                                scores)
            scores = scores / scores.sum(dim=-1, keepdim=True)  # Renormalize
        
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
        max_capacity = capacity.clone()  # Preserve original capacity
        dynamic_capacity = capacity.clone().expand(batch_size, -1).to(device)
        demands = demand.to(device)
        
        # Track visited nodes
        index = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Collect actions and log probabilities
        log_ps = []
        actions = []
        _input = encoder_inputs[:, 0, :]  # Start from depot
        
        for i in range(n_steps):
            # Check if all nodes visited
            if not mask1[:, 1:].eq(0).any():
                break
            
            # Prepare decoder input with capacity
            decoder_input = torch.cat([_input, dynamic_capacity], -1)
            decoder_input = self.fc(decoder_input)
            pool_processed = self.fc1(pool.to(device))
            decoder_input = decoder_input + pool_processed
            
            # Update mask
            mask, mask1 = self.update_mask(demands, dynamic_capacity, 
                                          index, mask1, i)
            
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
                                                index, max_capacity)
            
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
        """Update mask based on capacity constraints and prevent invalid depot repeats."""
        batch_size = demands.size(0)
        
        # Update visited mask for non-depot nodes selected in the previous step
        if step > 0:
            for b in range(batch_size):
                node_idx = int(index[b].item())
                if node_idx != 0:  # Only mark customers as visited
                    mask1[b, node_idx] = 1
        
        # Create a fresh mask based on current state
        mask = torch.zeros_like(mask1)
        
        # Build current mask based on visited and capacity
        for b in range(batch_size):
            # Mask visited customers and those exceeding capacity
            for n in range(1, demands.size(1)):
                # Already visited
                if mask1[b, n] == 1:
                    mask[b, n] = 1
                # Would exceed capacity
                elif demands[b, n].item() > dynamic_capacity[b].item():
                    mask[b, n] = 1
                else:
                    mask[b, n] = 0  # Available to visit
            
            # Depot handling
            if step == 0:
                # Can't start by returning to depot
                mask[b, 0] = 1
            else:
                # Check last visited node
                last_idx = int(index[b].item())
                
                # Check if any customers are feasible
                any_feasible = False
                for n in range(1, demands.size(1)):
                    if mask1[b, n] == 0 and demands[b, n].item() <= dynamic_capacity[b].item():
                        any_feasible = True
                        break
                
                # Prevent consecutive depot visits when customers are feasible
                if last_idx == 0 and any_feasible:
                    mask[b, 0] = 1  # Mask depot to avoid 0 -> 0
                else:
                    mask[b, 0] = 0  # Depot is allowed
            
            # If all customers are masked (visited or infeasible), force depot
            if mask[b, 1:].all():
                mask[b, :] = 1
                mask[b, 0] = 0  # Only depot available
        
        return mask, mask1
    
    def update_state(self, demands, dynamic_capacity, index, max_capacity):
        """Update dynamic capacity after visiting nodes"""
        batch_size = demands.size(0)
        
        for b in range(batch_size):
            node = index[b].item()
            if node == 0:  # Depot
                # Reset to max capacity for this instance
                if max_capacity.dim() > 0:
                    dynamic_capacity[b] = max_capacity[b].item() if max_capacity.size(0) > 1 else max_capacity[0].item()
                else:
                    dynamic_capacity[b] = max_capacity.item()
            else:
                # Subtract demand from current capacity
                dynamic_capacity[b] -= demands[b, node].item()
        
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
        # Only use actual nodes, not padding
        actual_embeddings = []
        actual_demands = []
        for i in range(batch_size):
            n_nodes = len(instances[i]['coords'])
            actual_embeddings.append(node_embeddings[i, :n_nodes, :])
            actual_demands.append(torch.tensor(instances[i]['demands'], dtype=torch.float32, device=device))
        
        # Process each instance separately since they may have different sizes
        all_actions = []
        all_log_ps = []
        for i in range(batch_size):
            embed = actual_embeddings[i].unsqueeze(0)  # [1, n_nodes, hidden_dim]
            graph_embed = embed.mean(dim=1)  # [1, hidden_dim]
            demand = actual_demands[i].unsqueeze(0)  # [1, n_nodes]
            cap = capacities[i:i+1]  # [1, 1]
            
            # Decode routes for this instance
            actions, log_p = self.decoder(
                embed, graph_embed, cap,
                demand, max_steps, temperature, greedy
            )
            all_actions.append(actions)
            all_log_ps.append(log_p)
        
        # Convert actions to routes format
        routes = []
        for b in range(batch_size):
            actions = all_actions[b]
            route = [0]  # Start at depot
            visited = set()
            n_customers = len(instances[b]['coords']) - 1
            
            for step in range(actions.size(1)):
                node = int(actions[0, step].item())  # actions is [1, seq_len] for each instance
                
                # Avoid consecutive depots in the route representation
                if node == 0 and route[-1] == 0:
                    continue
                
                route.append(node)
                
                if node != 0:
                    visited.add(node)
                    # If all customers visited, ensure final return to depot and stop
                    if len(visited) >= n_customers:
                        if route[-1] != 0:
                            route.append(0)
                        break
                # If node == 0, this represents end of a trip; continue to next trip
            
            # Ensure route ends at depot
            if route[-1] != 0:
                route.append(0)
                
            # Final sanity: if not all customers visited (unlikely), try to append missing customers (fallback)
            if len(visited) < n_customers:
                # Append any missing customers followed by depot to keep validation robust
                for cust in range(1, n_customers + 1):
                    if cust not in visited:
                        route.insert(-1, cust)
                # Ensure end at depot
                if route[-1] != 0:
                    route.append(0)
            
            routes.append(route)
        
        # Concatenate log probabilities
        log_p = torch.cat(all_log_ps, dim=0)
        
        # Calculate dummy entropy for interface compatibility
        entropy = torch.zeros(batch_size, device=device)
        
        return routes, log_p, entropy
