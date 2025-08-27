import torch
import torch.nn as nn
from typing import List

class DynamicGraphTransformerNetwork(nn.Module):
    """Dynamic Graph Transformer with adaptive state updates (RL-capable)."""
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config):
        super().__init__()
        self.hidden_dim = hidden_dim
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
        dgt_config = config['model']['dynamic_graph_transformer'] if config else {
            'state_features': 4,
            'residual_gate_init': -2.19722458,
            'update_input_multiplier': 2,
        }
        state_features = dgt_config['state_features']
        self.state_encoder = nn.Linear(state_features, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.res_gate = nn.Parameter(torch.tensor(dgt_config['residual_gate_init']))
        update_multiplier = dgt_config['update_input_multiplier']
        self.dynamic_update = nn.Sequential(
            nn.Linear(hidden_dim * update_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        pointer_multiplier = 3  # node + context + state
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * pointer_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, instances, max_steps=None, temperature=None, greedy=False, config=None):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * config['inference']['max_steps_multiplier']
        if temperature is None:
            temperature = config['inference']['default_temperature']
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3)
        demands_batch = torch.zeros(batch_size, max_nodes)
        capacities = torch.zeros(batch_size)
        distances_batch = torch.zeros(batch_size, max_nodes, max_nodes)
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']
            distances_batch[i, :n_nodes, :n_nodes] = torch.tensor(inst['distances'], dtype=torch.float32)
        embedded = self.node_embedding(node_features)
        x = embedded
        for layer in self.transformer_layers:
            x = layer(x)
        return self._generate_routes_dynamic(x, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances, config)

    def _generate_routes_dynamic(self, node_embeddings, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances=None, config=None):
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        current_nodes = torch.zeros(batch_size, dtype=torch.long)
        for b in range(batch_size):
            routes[b].append(0)
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        for step in range(max_steps):
            for b in range(batch_size):
                if not batch_done[b]:
                    actual_nodes = len(instances[b]['coords']) if instances and b < len(instances) else max_nodes
                    customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                    currently_at_depot = current_nodes[b].item() == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            if batch_done.all():
                break
            capacity_used = (capacities - remaining_capacity) / capacities
            step_progress = torch.full((batch_size,), step / max_steps)
            visited_count = visited.float().sum(dim=1) / max_nodes
            distance_from_depot = torch.zeros(batch_size)
            for b in range(batch_size):
                current_pos = current_nodes[b].item()
                distance_from_depot[b] = distances_batch[b, current_pos, 0]
            state_features = torch.stack([
                capacity_used, step_progress, visited_count, distance_from_depot
            ], dim=1)
            state_encoding = self.state_encoder(state_features)
            dynamic_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)
            normed = self.pre_norm(node_embeddings)
            update_input = torch.cat([normed, dynamic_context], dim=-1)
            delta = self.dynamic_update(update_input)
            gate = torch.sigmoid(self.res_gate)
            updated_embeddings = node_embeddings + gate * delta
            global_context = updated_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            state_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)
            pointer_input = torch.cat([updated_embeddings, global_context, state_context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            cap_mask = demands_batch > remaining_capacity.unsqueeze(1)
            mask = visited | cap_mask
            pad_mask = torch.zeros_like(mask)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords']) if instances and b < len(instances) else max_nodes
                if actual_nodes < max_nodes:
                    pad_mask[b, actual_nodes:] = True
                pad_mask[b, 0] = False
            mask = mask | pad_mask
            currently_at_depot_vec = torch.tensor([len(r) > 0 and r[-1] == 0 for r in routes])
            if currently_at_depot_vec.any():
                mask[currently_at_depot_vec, 0] = True
            all_masked = mask.all(dim=1)
            need_allow_depot = all_masked & (~currently_at_depot_vec)
            if need_allow_depot.any():
                mask[need_allow_depot, 0] = False
            done_mask = all_masked & currently_at_depot_vec
            batch_done[done_mask] = True
            masked_score_value = config['inference']['masked_score_value']
            scores = scores.masked_fill(mask, masked_score_value)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            log_prob_epsilon = config['inference']['log_prob_epsilon']
            log_probs = torch.log(probs + log_prob_epsilon)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
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
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    current_nodes[b] = action
                    if action == 0:
                        remaining_capacity[b] = capacities[b]
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            all_done = True
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords']) if instances and b < len(instances) else max_nodes
                customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                currently_at_depot = current_nodes[b].item() == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            if all_done:
                break
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        return routes, combined_log_probs, combined_entropy

