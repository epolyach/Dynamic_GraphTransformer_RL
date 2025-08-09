"""
Greedy baseline model: Graph Transformer encoder + greedy decoding (no RL, no dynamic updates).
"""
import torch
from torch import nn

from .graph_transformer import GraphTransformerEncoder


class GreedyGraphTransformerBaseline(nn.Module):
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
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data, n_steps: int, greedy: bool = True, T: float = 1.0):
        device = data.x.device
        batch_size = data.num_graphs
        total_nodes = data.x.size(0)
        num_nodes = total_nodes // batch_size

        x = self.encoder(data)
        x_proj = self.output_projection(x)

        coordinates = data.x[:, :2].view(batch_size, num_nodes, 2)
        demands = data.demand.view(batch_size, num_nodes)
        capacity = data.capacity.view(batch_size, num_nodes)[:, 0]

        actions, log_probs = self._greedy_routing(x_proj, coordinates, demands, capacity, n_steps, device)
        return actions, log_probs

    def _greedy_routing(self, embeddings, coordinates, demands, capacity, n_steps, device):
        batch_size, num_nodes, hidden_dim = embeddings.shape
        actions = []
        log_probs = []
        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        remaining_capacity = capacity.clone()

        for step in range(n_steps):
            current_embeddings = embeddings[torch.arange(batch_size), current_node]
            similarities = torch.bmm(current_embeddings.unsqueeze(1), embeddings.transpose(1, 2)).squeeze(1)
            current_coords = coordinates[torch.arange(batch_size), current_node]
            distances = torch.norm(coordinates - current_coords.unsqueeze(1), dim=2)
            scores = similarities - 0.1 * distances

            visited_mask = visited.clone(); visited_mask[:, 0] = False
            scores.masked_fill_(visited_mask, -float('inf'))
            capacity_mask = demands > remaining_capacity.unsqueeze(1); capacity_mask[:, 0] = False
            scores.masked_fill_(capacity_mask, -float('inf'))

            next_node = torch.argmax(scores, dim=1)
            actions.append(next_node.unsqueeze(-1))
            log_probs.append(torch.zeros(batch_size, 1, device=device))

            visited[torch.arange(batch_size), next_node] = True
            is_depot = (next_node == 0)
            remaining_capacity = torch.where(
                is_depot,
                capacity,
                remaining_capacity - demands[torch.arange(batch_size), next_node]
            )
            current_node = next_node
            if visited[:, 1:].all(dim=1).all():
                break

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return actions, log_probs

