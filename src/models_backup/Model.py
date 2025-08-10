import torch
from torch import nn
from .graph_transformer import GraphTransformerEncoder
from .GAT_Decoder import GAT_Decoder


class Model(nn.Module):
    """Updated Model using Graph Transformer instead of GAT encoder"""
    
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout):
        super(Model, self).__init__()
        
        # Replace GAT encoder with Graph Transformer encoder
        # Map old parameters to Graph Transformer parameters
        self.encoder = GraphTransformerEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_heads=8,  # Default number of attention heads
            num_layers=layers,
            dropout=dropout,
            pe_type="sinusoidal",  # Use sinusoidal positional encoding
            pe_dim=min(hidden_dim, 64),  # Positional encoding dimension
            max_distance=100.0,  # Maximum distance for normalization
            use_edge_weights=True
        )
        
        # Keep the same decoder for compatibility
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)

    def forward(self, data, n_steps, greedy, T):
        """Forward pass using Graph Transformer encoder"""
        # data.x: node features, data.edge_attr: edge features, data.edge_index: edge indices
        
        # Use Graph Transformer encoder (returns shape: (batch_size, num_nodes, hidden_dim))
        x = self.encoder(data)
        
        # Compute the graph embedding > mean of all node embeddings per feature dimension
        graph_embedding = x.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Prepare demand and capacity tensors
        batch_size = data.num_graphs
        demand = data.demand.reshape(batch_size, -1).float().to(data.x.device)
        capacity = data.capacity.reshape(batch_size, -1).float().to(data.x.device)
        
        # Call the decoder with [B, N, H] embeddings
        actions, log_p = self.decoder(x, graph_embedding, capacity, demand, n_steps, T, greedy)
        
        return actions, log_p
