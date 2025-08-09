"""
PointerRLModel: Graph Transformer encoder + Pointer (GAT) decoder trained with RL.
This separates the Pointer+RL variant into a dedicated, easily detachable module.
"""
from typing import Tuple
import torch
from torch import nn

from .graph_transformer import GraphTransformerEncoder
from .GAT_Decoder import GAT_Decoder


class PointerRLModel(nn.Module):
    """
    Pointer + RL model:
    - Graph Transformer encoder produces node embeddings
    - Pointer decoder (GAT_Decoder) selects next nodes step-by-step
    - Trained via REINFORCE using log-probabilities from the decoder
    """

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

        # Encoder: Graph Transformer
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

        # Decoder: Pointer attention-based decoder
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)

    @torch.no_grad()
    def infer(self, data, n_steps: int, T: float = 1.0):
        """Greedy inference convenience wrapper (no gradients)."""
        return self.forward(data, n_steps=n_steps, greedy=True, T=T)

    def forward(self, data, n_steps: int, greedy: bool = False, T: float = 1.0):
        """
        Forward pass with RL decoder (no dynamic updates inside this model).

        Args:
            data: PyG Data object with fields x, edge_index, edge_attr, demand, capacity, batch
            n_steps: number of decoding steps
            greedy: whether to use greedy decoding
            T: temperature for sampling

        Returns:
            actions: Tensor [B, S] of selected node indices per step
            log_p: Tensor [B] total log-probability of the trajectory
        """
        device = data.x.device
        batch_size = data.num_graphs

        # Encode graph
        x = self.encoder(data)  # [B, N, H]

        # Graph embedding (pooling)
        graph_embedding = x.mean(dim=1)  # [B, H]

        # Prepare demand/capacity for decoder
        demand = data.demand.reshape(batch_size, -1).float().to(device)
        capacity_full = data.capacity.reshape(batch_size, -1).float().to(device)
        capacity = capacity_full[:, :1]  # decoder expects [B,1] remaining capacity

        # Decode
        actions, log_p = self.decoder(
            x, graph_embedding, capacity, demand, n_steps, T, greedy
        )

        return actions, log_p
