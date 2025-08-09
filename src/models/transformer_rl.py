"""
Static Transformer + RL model: Graph Transformer encoder + RL decoder without dynamic updates.
Serves as a clean, detachable module for the static RL variant.
"""
import torch
from torch import nn

from .graph_transformer import GraphTransformerEncoder
from .GAT_Decoder import GAT_Decoder


class StaticRLGraphTransformer(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 1,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        pe_type: str = "sinusoidal",
        pe_dim: int = 64,
        max_distance: float = 100.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = GraphTransformerEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pe_type=pe_type,
            pe_dim=pe_dim,
            max_distance=max_distance,
            use_edge_weights=True,
        )
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)

    def forward(self, data, n_steps: int, greedy: bool = False, T: float = 1.0):
        device = data.x.device
        batch_size = data.num_graphs
        x = self.encoder(data)
        graph_embedding = x.mean(dim=1)
        demand = data.demand.reshape(batch_size, -1).float().to(device)
        capacity_full = data.capacity.reshape(batch_size, -1).float().to(device)
        capacity = capacity_full[:, :1]
        actions, log_p = self.decoder(x, graph_embedding, capacity, demand, n_steps, T, greedy)
        return actions, log_p

