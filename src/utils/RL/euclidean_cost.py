import torch

def euclidean_cost(static, actions, batch):
    """
    Compute the total route cost based on the actions (node indices) and the static coordinates of the nodes.
    Args:
        static: torch.Tensor, shape [total_nodes, 2], containing the (x, y) coordinates of nodes for a batched Data.
        actions: torch.Tensor, shape [batch_size, steps] or [batch_size, steps, 1], node indices of the route.
        batch: PyG Data object with attributes x and either num_graphs or batch vector.
    """
    # Infer batch size and nodes per graph
    if hasattr(batch, 'num_graphs') and batch.num_graphs is not None:
        batch_size = int(batch.num_graphs)
    elif hasattr(batch, 'batch') and batch.batch is not None:
        batch_size = int(batch.batch.max().item() + 1)
    else:
        batch_size = 1
    total_nodes = batch.x.size(0)
    num_nodes = int(total_nodes // batch_size)

    # Normalize actions shape to [B, S]
    if actions.dim() == 3 and actions.size(-1) == 1:
        actions = actions.squeeze(-1)

    # Reshape and transpose the static coordinates to match expected dims
    static = static.reshape(-1, num_nodes, 2)
    static = static.transpose(2, 1)  # [B, 2, N]

    # Get the coordinates of nodes in the visit order
    idx = actions.unsqueeze(1).expand(-1, static.size(1), -1)  # [B, 2, S]
    tour = torch.gather(static, 2, idx).permute(0, 2, 1)  # [B, S, 2]

    # Add the depot at start and end
    start = static.data[:, :, 0].unsqueeze(1)  # [B, 1, 2]
    y = torch.cat((start, tour, start), dim=1)  # [B, S+2, 2]

    # Euclidean distances between consecutive points
    diff = y[:, :-1] - y[:, 1:]
    tour_len = torch.sqrt(torch.sum(diff * diff, dim=2))  # [B, S+1]

    # Sum distances per batch
    total_tour_len = tour_len.sum(1).detach()

    # Penalize depot visits
    depot_visits = (actions == 0).sum(dim=1)
    depot_penalty = 0.3 * (depot_visits)
    total_tour_len += depot_penalty

    return total_tour_len.detach()
