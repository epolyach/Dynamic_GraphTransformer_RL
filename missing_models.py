
class GraphTransformerNetwork(nn.Module):
    """Pipeline 2: Graph Transformer with multi-head self-attention"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Graph-level aggregation
        self.graph_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3, device=device)
        demands_batch = torch.zeros(batch_size, max_nodes, device=device)
        capacities = torch.zeros(batch_size, device=device)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            capacities[i] = inst['capacity']
        
        # Embed nodes
        embedded = self.node_embedding(node_features)  # [B, N, H]
        
        # Apply transformer layers
        x = embedded
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Graph-level attention for global context
        graph_context, _ = self.graph_attention(x, x, x)
        enhanced_embeddings = x + graph_context  # Residual connection
        
        return self._generate_routes(enhanced_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances)
    
    def _generate_routes(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances):
        # Same routing logic as baseline but with enhanced embeddings
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        # Track which batches are done to avoid processing them
        batch_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            # Check which batches are done (all customers visited AND at depot)
            for b in range(batch_size):
                if not batch_done[b]:  # Only check if not already done
                    customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            # If all batches are done, break
            if batch_done.all():
                break
            
            # Dynamic context based on current state
            context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            
            # Compute pointer scores with enhanced embeddings
            pointer_input = torch.cat([node_embeddings, context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                # Don't allow staying at depot if already at depot
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                # Safety: if all nodes masked and we're not at depot, allow depot
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                # If we're at depot and all customers visited, we should have terminated above
                elif mask[b].all() and currently_at_depot:
                    # Mark this batch as done to avoid further processing
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            selected_log_probs = torch.zeros(batch_size, device=device)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        probs = torch.softmax(scores[b] / temperature, dim=-1)
                        actions[b] = torch.multinomial(probs, 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            
            # Update state only for batches that aren't done
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    
                    if action == 0:  # Return to depot
                        remaining_capacity[b] = capacities[b]
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Check termination: all customers visited AND currently at depot
            all_done = True
            for b in range(batch_size):
                customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Routes should already end at depot due to termination condition
        # Only add depot if route is empty (shouldn't happen)
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size, device=device)
        return routes, combined_log_probs

class DynamicGraphTransformerNetwork(nn.Module):
    """Pipeline 4: Dynamic Graph Transformer with adaptive updates"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Dynamic update components
        self.state_encoder = nn.Linear(4, hidden_dim)  # capacity_used, step, visited_count, distance_from_depot
        self.dynamic_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # node + context + state
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3, device=device)
        demands_batch = torch.zeros(batch_size, max_nodes, device=device)
        capacities = torch.zeros(batch_size, device=device)
        distances_batch = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            capacities[i] = inst['capacity']
            distances_batch[i, :n_nodes, :n_nodes] = torch.tensor(inst['distances'], dtype=torch.float32, device=device)
        
        # Initial embedding
        embedded = self.node_embedding(node_features)
        
        # Apply transformer layers
        x = embedded
        for layer in self.transformer_layers:
            x = layer(x)
        
        return self._generate_routes_dynamic(x, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances)
    
    def _generate_routes_dynamic(self, node_embeddings, node_features, demands_batch, capacities, distances_batch, max_steps, temperature, greedy, instances=None):
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        current_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)  # Current position
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        # Track which batches are done to avoid processing them
        batch_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            # Check which batches are done (all customers visited AND at depot)
            for b in range(batch_size):
                if not batch_done[b]:  # Only check if not already done
                    if instances and b < len(instances):
                        actual_nodes = len(instances[b]['coords'])
                    else:
                        actual_nodes = max_nodes
                    customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                    currently_at_depot = current_nodes[b].item() == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            # If all batches are done, break
            if batch_done.all():
                break
            
            # Dynamic state features
            capacity_used = (capacities - remaining_capacity) / capacities
            step_progress = torch.full((batch_size,), step / max_steps, device=device)
            visited_count = visited.float().sum(dim=1) / max_nodes
            
            # Distance from depot
            distance_from_depot = torch.zeros(batch_size, device=device)
            for b in range(batch_size):
                current_pos = current_nodes[b].item()
                distance_from_depot[b] = distances_batch[b, current_pos, 0]
            
            # Encode dynamic state
            state_features = torch.stack([
                capacity_used, step_progress, visited_count, distance_from_depot
            ], dim=1)  # [B, 4]
            
            state_encoding = self.state_encoder(state_features)  # [B, H]
            
            # Update node embeddings based on current state
            dynamic_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)  # [B, N, H]
            update_input = torch.cat([node_embeddings, dynamic_context], dim=-1)
            node_updates = self.dynamic_update(update_input)
            
            # Apply dynamic updates
            updated_embeddings = node_embeddings + 0.1 * node_updates  # Residual with small learning rate
            
            # Enhanced context with state information
            global_context = updated_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            state_context = state_encoding.unsqueeze(1).expand(-1, max_nodes, -1)
            
            # Enhanced pointer scores
            pointer_input = torch.cat([updated_embeddings, global_context, state_context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask: visited nodes + capacity constraints
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                # Don't allow staying at depot if already at depot
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                # Safety: if all nodes masked and we're not at depot, allow depot
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                # If we're at depot and all customers visited, we should have terminated above
                elif mask[b].all() and currently_at_depot:
                    # Mark this batch as done to avoid further processing
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
            # Sample actions only for batches that aren't done
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            selected_log_probs = torch.zeros(batch_size, device=device)
            
            for b in range(batch_size):
                if not batch_done[b]:  # Only sample for batches that aren't done
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        probs = torch.softmax(scores[b] / temperature, dim=-1)
                        actions[b] = torch.multinomial(probs, 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            
            # Update state only for batches that aren't done
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    current_nodes[b] = action
                    
                    if action == 0:  # Return to depot
                        remaining_capacity[b] = capacities[b]
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Check termination: all customers visited AND currently at depot
            all_done = True
            for b in range(batch_size):
                if instances and b < len(instances):
                    actual_nodes = len(instances[b]['coords'])
                else:
                    actual_nodes = max_nodes
                customers_visited = visited[b, 1:actual_nodes].all() if actual_nodes > 1 else True
                currently_at_depot = current_nodes[b].item() == 0
                
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Routes should already end at depot due to termination condition
        # Only add depot if route is empty (shouldn't happen)
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size, device=device)
        return routes, combined_log_probs

class GraphAttentionTransformer(nn.Module):
    """Pipeline 5: Graph Attention Transformer with Edge Features"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Edge feature embedding (distance)
        self.edge_embedding = nn.Linear(1, hidden_dim // 4)
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final attention aggregation
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Enhanced pointer network
        self.pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
        batch_size = len(instances)
        if max_steps is None:
            max_steps = len(instances[0]['coords']) * 2
        
        # Convert to tensor format
        max_nodes = max(len(inst['coords']) for inst in instances)
        node_features = torch.zeros(batch_size, max_nodes, 3, device=device)
        demands_batch = torch.zeros(batch_size, max_nodes, device=device)
        capacities = torch.zeros(batch_size, device=device)
        distances_batch = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
        
        for i, inst in enumerate(instances):
            n_nodes = len(inst['coords'])
            node_features[i, :n_nodes, :2] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            node_features[i, :n_nodes, 2] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            capacities[i] = inst['capacity']
            distances_batch[i, :n_nodes, :n_nodes] = torch.tensor(inst['distances'], dtype=torch.float32, device=device)
        
        # Initial node embedding
        node_embeds = self.node_embedding(node_features)  # [B, N, H]
        
        # Apply GAT layers with residual connections
        x = node_embeds
        for gat_layer, layer_norm in zip(self.gat_layers, self.layer_norms):
            # Self-attention with residual
            attn_out, _ = gat_layer(x, x, x)
            x = layer_norm(x + attn_out)
        
        # Global context attention
        global_context, _ = self.global_attention(x, x, x)
        enhanced_embeddings = x + global_context
        
        return self._generate_routes_gat(enhanced_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances)
    
    def _generate_routes_gat(self, node_embeddings, node_features, demands_batch, capacities, max_steps, temperature, greedy, instances):
        # Same routing logic but with GAT-enhanced embeddings
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        routes = [[] for _ in range(batch_size)]
        all_log_probs = []
        
        remaining_capacity = capacities.clone()
        visited = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        
        # ALWAYS START AT DEPOT
        for b in range(batch_size):
            routes[b].append(0)
        
        batch_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            # Check termination
            for b in range(batch_size):
                if not batch_done[b]:
                    customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                    currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                    if customers_visited and currently_at_depot:
                        batch_done[b] = True
            
            if batch_done.all():
                break
            
            # Enhanced context with attention
            context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, max_nodes, -1)
            
            # Compute pointer scores
            pointer_input = torch.cat([node_embeddings, context], dim=-1)
            scores = self.pointer(pointer_input).squeeze(-1)
            
            # Apply mask
            mask = visited.clone()
            for b in range(batch_size):
                for n in range(max_nodes):
                    if demands_batch[b, n] > remaining_capacity[b]:
                        mask[b, n] = True
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if currently_at_depot:
                    mask[b, 0] = True
                
                if mask[b].all() and not currently_at_depot:
                    mask[b, 0] = False
                elif mask[b].all() and currently_at_depot:
                    batch_done[b] = True
            
            scores = scores.masked_fill(mask, float('-inf'))
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            
            # Sample actions
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            selected_log_probs = torch.zeros(batch_size, device=device)
            
            for b in range(batch_size):
                if not batch_done[b]:
                    if greedy:
                        actions[b] = log_probs[b].argmax()
                    else:
                        probs = torch.softmax(scores[b] / temperature, dim=-1)
                        actions[b] = torch.multinomial(probs, 1).squeeze()
                    
                    selected_log_probs[b] = log_probs[b, actions[b]]
            
            all_log_probs.append(selected_log_probs)
            
            # Update state
            for b in range(batch_size):
                if not batch_done[b]:
                    action = actions[b].item()
                    routes[b].append(action)
                    
                    if action == 0:
                        remaining_capacity[b] = capacities[b]
                    else:
                        visited[b, action] = True
                        remaining_capacity[b] -= demands_batch[b, action]
            
            # Final termination check
            all_done = True
            for b in range(batch_size):
                customers_visited = visited[b, 1:len(instances[b]['coords'])].all()
                currently_at_depot = len(routes[b]) > 0 and routes[b][-1] == 0
                if not (customers_visited and currently_at_depot):
                    all_done = False
                    break
            
            if all_done:
                break
        
        for b in range(batch_size):
            if len(routes[b]) == 0:
                routes[b].append(0)
        
        combined_log_probs = torch.stack(all_log_probs, dim=1).sum(dim=1) if all_log_probs else torch.zeros(batch_size, device=device)
        return routes, combined_log_probs

