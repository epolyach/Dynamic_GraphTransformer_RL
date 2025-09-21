import torch
import torch.nn as nn
from typing import List

class GraphTransformerGreedy(nn.Module):
    """Pure Greedy Attention Baseline (no RL updates)."""
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_type = "GT-Greedy"
        self.num_heads = num_heads
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * feedforward_multiplier,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, instances, max_steps=None, temperature=None, greedy=True, config=None):
        batch_size = len(instances)
        if batch_size == 0:
            return [], torch.tensor([]), torch.tensor([])
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        max_nodes = max(len(inst['coords']) for inst in instances)
        device = next(self.parameters()).device
        node_features = torch.zeros(batch_size, max_nodes, 3, device=device)
        demands_batch = torch.zeros(batch_size, max_nodes, device=device)
        capacities = torch.zeros(batch_size, device=device)
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            capacities[i] = inst['capacity']
        x = self.node_embedding(node_features)
        for layer in self.transformer_layers:
            x = layer(x)
        return self._greedy_routes(x, demands_batch, capacities, max_steps, instances)

    def _greedy_routes(self, node_embeddings, demands_batch, capacities, max_steps, instances):
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        device = node_embeddings.device
        routes = [[] for _ in range(batch_size)]
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        current_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)
        for b in range(batch_size):
            routes[b].append(0)
        K_all = self.attn_k(node_embeddings)
        scale = float(self.hidden_dim) ** 0.5
        for step in range(max_steps):
            done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords'])
                customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                at_depot = (routes[b][-1] == 0)
                if customers_visited and at_depot:
                    done_mask[b] = True
            if done_mask.all():
                break
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            for b in range(batch_size):
                if done_mask[b]:
                    actions[b] = 0
                    continue
                actual_nodes = len(instances[b]['coords'])
                mask = torch.zeros(max_nodes, dtype=torch.bool, device=device)
                mask |= visited[b]
                if actual_nodes < max_nodes:
                    mask[actual_nodes:] = True
                infeasible = demands_batch[b] > remaining_capacity[b]
                infeasible[0] = False
                mask |= infeasible
                feasible_unvisited_exists = (~mask[1:actual_nodes]).any() if actual_nodes > 1 else False
                if feasible_unvisited_exists:
                    mask[0] = True
                else:
                    mask[0] = False
                currently_at_depot = (routes[b][-1] == 0)
                if mask.all() and not currently_at_depot:
                    mask[0] = False
                if mask.all():
                    actions[b] = 0
                    continue
                curr = routes[b][-1]
                q = self.attn_q(node_embeddings[b, curr:curr+1, :])
                k = K_all[b, :, :]
                scores = (q @ k.t()).squeeze(0) / scale
                scores = scores.masked_fill(mask, -1e9)
                next_node = int(torch.argmax(scores).item())
                actions[b] = next_node
            for b in range(batch_size):
                if done_mask[b]:
                    continue
                a = int(actions[b].item())
                routes[b].append(a)
                if a == 0:
                    remaining_capacity[b] = capacities[b]
                else:
                    visited[b, a] = True
                    remaining_capacity[b] -= demands_batch[b, a]
                current_nodes[b] = a
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        logp = torch.zeros(batch_size, dtype=torch.float32, device=device)
        ent = torch.zeros(batch_size, dtype=torch.float32, device=device)
        return routes, logp, ent

