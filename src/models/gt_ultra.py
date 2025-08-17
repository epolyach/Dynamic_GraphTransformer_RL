import torch
import torch.nn as nn
from typing import List

class GraphTransformerUltra(nn.Module):
    """Ultra-lightweight Graph Transformer with 64-dim hidden size.
    
    Key optimizations:
    - Small hidden dimension (64 instead of 128)
    - Fewer attention heads (2 instead of 4)
    - Minimal layers (2 instead of 4)
    - ~90K parameters (4x smaller than GAT)
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config):
        super().__init__()
        # Override to match DGT-Ultra settings (~30K parameters)
        self.hidden_dim = 48
        self.num_heads = 3
        self.num_layers = 2
        
        self.node_embedding = nn.Linear(input_dim, self.hidden_dim)
        
        # Minimal attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, self.num_heads, dropout=dropout, batch_first=True)
            for _ in range(self.num_layers)
        ])
        
        # Minimal normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Global context aggregation
        self.global_attention = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
        
        # Simple pointer network
        self.pointer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
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

        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities[i] = inst['capacity']

        # Initial embedding
        embedded = self.node_embedding(node_features)
        x = embedded
        
        # Lightweight attention layers
        for attention, norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attention(x, x, x)
            x = norm(x + attn_out)
        
        # Global context
        global_context, _ = self.global_attention(x, x, x)
        enhanced_embeddings = x + global_context
        
        return self._generate_routes(enhanced_embeddings, node_features, demands_batch, capacities, 
                                   max_steps, temperature, greedy, instances, config)

    def _generate_routes(self, node_embeddings, node_features, demands_batch, capacities, 
                        max_steps, temperature, greedy, instances, config):
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes: List[List[int]] = [[] for _ in range(batch_size)]
        all_log_probs = []
        all_entropies = []
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        for b in range(batch_size):
            routes[b].append(0)
        
        batch_done = torch.zeros(batch_size, dtype=torch.bool)
        
        for step in range(max_steps):
            # Check for completion
            for b in range(batch_size):
                if not batch_done[b]:
                    customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            if batch_done.all():
                break
            
            # Simple global context
            context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            pointer_input = torch.cat([node_embeddings, context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Standard masking logic (same as other models)
            cap_mask = demands_batch > remaining_capacity.unsqueeze(1)
            mask = visited | cap_mask
            
            pad_mask = torch.zeros_like(mask)
            for b in range(batch_size):
                actual_nodes = len(instances[b]['coords'])
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
            
            # Apply masking and compute probabilities
            masked_score_value = config['inference']['masked_score_value']
            scores = scores.masked_fill(mask, masked_score_value)
            logits = scores / temperature
            probs = torch.softmax(logits, dim=-1)
            
            log_prob_epsilon = config['inference']['log_prob_epsilon']
            log_probs = torch.log(probs + log_prob_epsilon)
            step_entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
            
            # Action selection
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
            
            # Update state
            for b in range(batch_size):
                if not batch_done[b]:
                    action = int(actions[b].item())
                    routes[b].append(action)
                    if action == 0:
                        remaining_capacity[b] = capacities[b]
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Final completion check
            all_done = True
            for b in range(batch_size):
                customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Ensure all routes have at least depot
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        # Combine log probabilities and entropy
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size)
        combined_entropy = torch.stack(all_entropies, dim=1).sum(dim=1) if all_entropies else torch.zeros(batch_size)
        
        return routes, combined_log_probs, combined_entropy
