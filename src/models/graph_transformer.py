"""
Graph Transformer implementation for CVRP
This module implements a Graph Transformer encoder that replaces the GAT architecture
with more expressive multi-head attention mechanisms and positional encodings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encodings for graph nodes based on their spatial coordinates
    """
    def __init__(self, d_model: int, pe_type: str = "sinusoidal", max_distance: float = 100.0):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type
        self.max_distance = max_distance
        
        if pe_type == "learnable":
            # Learnable positional embeddings
            self.pos_embedding = nn.Linear(2, d_model)  # 2D coordinates
        elif pe_type == "distance_based":
            # Distance-based positional encoding
            self.distance_embedding = nn.Linear(1, d_model)
        
    def forward(self, coordinates: Tensor, batch_size: int, num_nodes: int) -> Tensor:
        """
        Generate positional encodings based on node coordinates
        
        Args:
            coordinates: Node coordinates [batch_size * num_nodes, 2]
            batch_size: Number of graphs in batch
            num_nodes: Number of nodes per graph
            
        Returns:
            Positional encodings [batch_size, num_nodes, d_model]
        """
        coords = coordinates.view(batch_size, num_nodes, 2)
        
        if self.pe_type == "sinusoidal":
            # Sinusoidal positional encoding based on coordinates
            pe = torch.zeros(batch_size, num_nodes, self.d_model, device=coordinates.device)
            
            # Normalize coordinates to [0, 1]
            normalized_coords = coords / self.max_distance
            
            # Generate sinusoidal encodings
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=coordinates.device) * 
                               -(math.log(10000.0) / self.d_model))
            
            # X coordinate encodings
            pe[:, :, 0::4] = torch.sin(normalized_coords[:, :, 0:1] * div_term[::2])
            pe[:, :, 1::4] = torch.cos(normalized_coords[:, :, 0:1] * div_term[::2])
            
            # Y coordinate encodings  
            pe[:, :, 2::4] = torch.sin(normalized_coords[:, :, 1:2] * div_term[::2])
            pe[:, :, 3::4] = torch.cos(normalized_coords[:, :, 1:2] * div_term[::2])
            
            return pe
            
        elif self.pe_type == "learnable":
            # Learnable positional embeddings
            return self.pos_embedding(coords)
            
        elif self.pe_type == "distance_based":
            # Distance from depot (assuming depot is at index 0)
            depot_coords = coords[:, 0:1, :]  # [batch_size, 1, 2]
            distances = torch.norm(coords - depot_coords, dim=-1, keepdim=True)  # [batch_size, num_nodes, 1]
            return self.distance_embedding(distances)
        
        else:
            return torch.zeros(batch_size, num_nodes, self.d_model, device=coordinates.device)


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head attention mechanism for graph data
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, 
                edge_weights: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of multi-head attention
        
        Args:
            x: Input features [batch_size, num_nodes, d_model]
            mask: Attention mask [batch_size, num_nodes, num_nodes]
            edge_weights: Edge weights for biasing attention [batch_size, num_nodes, num_nodes]
            
        Returns:
            Output features [batch_size, num_nodes, d_model]
        """
        batch_size, num_nodes, _ = x.size()
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add edge weight bias if provided
        if edge_weights is not None:
            # Expand edge weights to match attention heads
            edge_bias = edge_weights.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores + edge_bias
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.d_model)
        out = self.w_o(out)
        
        return out


class GraphTransformerLayer(nn.Module):
    """
    Single Graph Transformer layer with multi-head attention and feed-forward network
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadGraphAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, 
                edge_weights: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer layer
        
        Args:
            x: Input features [batch_size, num_nodes, d_model]
            mask: Attention mask
            edge_weights: Edge weights for attention bias
            
        Returns:
            Output features [batch_size, num_nodes, d_model]
        """
        # Multi-head attention with residual connection
        attn_out = self.attention(x, mask, edge_weights)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class GraphTransformerEncoder(nn.Module):
    """
    Complete Graph Transformer encoder for CVRP
    """
    def __init__(self, 
                 node_input_dim: int = 3,  # x, y, demand
                 edge_input_dim: int = 1,  # distance
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 pe_type: str = "sinusoidal",
                 pe_dim: int = 64,
                 max_distance: float = 100.0,
                 use_edge_weights: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_edge_weights = use_edge_weights
        
        # Input projections
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim) if use_edge_weights else None
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(pe_dim, pe_type, max_distance)
        self.pe_projection = nn.Linear(pe_dim, hidden_dim) if pe_dim != hidden_dim else nn.Identity()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _compute_edge_weights(self, edge_attr: Tensor, edge_index: Tensor, 
                            batch_size: int, num_nodes: int) -> Optional[Tensor]:
        """
        Convert edge attributes to dense attention bias matrix
        
        Args:
            edge_attr: Edge attributes [num_edges, edge_dim]
            edge_index: Edge indices [2, num_edges]
            batch_size: Number of graphs in batch
            num_nodes: Number of nodes per graph
            
        Returns:
            Dense edge weight matrix [batch_size, num_nodes, num_nodes]
        """
        if not self.use_edge_weights or edge_attr is None:
            return None
            
        device = edge_attr.device
        
        # Determine a stable dtype for edge weights under AMP/autocast.
        # Prefer float32 when autocast is enabled to avoid half/float mismatches in index_put.
        if torch.is_autocast_enabled():
            ew_dtype = torch.float32
        else:
            # Fall back to edge_attr's dtype if floating, else float32
            ew_dtype = edge_attr.dtype if edge_attr.is_floating_point() else torch.float32
        
        # Embed edge attributes and aggregate to scalar scores per edge
        edge_features = self.edge_embedding(edge_attr)
        # Ensure features and scores are in the chosen dtype
        if edge_features.dtype != ew_dtype:
            edge_features = edge_features.to(dtype=ew_dtype)
        edge_scores = edge_features.mean(dim=-1)
        if edge_scores.dtype != ew_dtype:
            edge_scores = edge_scores.to(dtype=ew_dtype)
        
        # Allocate dense matrix with the same dtype as edge_scores
        edge_weights = torch.zeros(batch_size, num_nodes, num_nodes, device=device, dtype=ew_dtype)
        
        # Fill dense matrix
        num_edges_per_graph = edge_index.size(1) // batch_size
        
        for b in range(batch_size):
            start_idx = b * num_edges_per_graph
            end_idx = start_idx + num_edges_per_graph
            
            batch_edge_index = edge_index[:, start_idx:end_idx] - b * num_nodes
            batch_edge_scores = edge_scores[start_idx:end_idx]
            if batch_edge_scores.dtype != edge_weights.dtype:
                batch_edge_scores = batch_edge_scores.to(dtype=edge_weights.dtype)
            edge_weights[b, batch_edge_index[0], batch_edge_index[1]] = batch_edge_scores
        
        return edge_weights
    
    def forward(self, data: Data) -> Tensor:
        """
        Forward pass of Graph Transformer encoder
        
        Args:
            data: PyTorch Geometric data object containing:
                - x: Node features [batch_size * num_nodes, node_input_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge attributes [num_edges, edge_input_dim]
                - demand: Node demands [batch_size * num_nodes, 1]
                - batch: Batch indices [batch_size * num_nodes]
                
        Returns:
            Node embeddings [batch_size, num_nodes, hidden_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        demand = data.demand if hasattr(data, 'demand') else torch.zeros(x.size(0), 1, device=x.device)
        
        # Get batch information
        batch_size = data.num_graphs
        total_nodes = x.size(0)
        num_nodes = total_nodes // batch_size
        
        # Combine node features with demand
        node_features = torch.cat([x, demand], dim=-1)  # [total_nodes, node_input_dim]
        
        # Embed nodes
        x = self.node_embedding(node_features)  # [total_nodes, hidden_dim]
        x = x.view(batch_size, num_nodes, self.hidden_dim)  # [batch_size, num_nodes, hidden_dim]
        
        # Add positional encodings
        coordinates = data.x[:, :2]  # Assuming first 2 features are coordinates
        pos_enc = self.pos_encoding(coordinates, batch_size, num_nodes)
        pos_enc = self.pe_projection(pos_enc)
        x = x + pos_enc
        
        # Compute edge weights for attention bias
        edge_weights = self._compute_edge_weights(edge_attr, edge_index, batch_size, num_nodes)
        # Match dtype with activations under AMP to avoid dtype mismatches
        if edge_weights is not None and edge_weights.dtype != x.dtype:
            edge_weights = edge_weights.to(dtype=x.dtype)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=None, edge_weights=edge_weights)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x  # [batch_size, num_nodes, hidden_dim]


if __name__ == "__main__":
    # Test the Graph Transformer encoder
    import torch_geometric
    from torch_geometric.data import Batch
    
    # Create dummy data
    batch_size = 2
    num_nodes = 10
    
    # Create node features (x, y coordinates)
    coordinates = torch.rand(batch_size * num_nodes, 2) * 100  # Random coordinates
    demands = torch.rand(batch_size * num_nodes, 1) * 10  # Random demands
    
    # Create edge indices (complete graph for simplicity)
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
    data = Data(x=coordinates, edge_index=edge_index, edge_attr=edge_attr, 
                demand=demands, batch=batch)
    data.num_graphs = batch_size
    
    # Test encoder
    encoder = GraphTransformerEncoder(
        node_input_dim=3,  # x, y, demand
        edge_input_dim=1,  # distance
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    )
    
    print(f"Input shape: {coordinates.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    
    output = encoder(data)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {num_nodes}, 128]")
    print("Graph Transformer encoder test passed!")
