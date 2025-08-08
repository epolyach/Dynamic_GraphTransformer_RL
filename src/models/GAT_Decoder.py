import logging
import torch
from torch import nn
from torch.distributions import Categorical
from .PointerAttention import PointerAttention
from .mask_capacity import update_state, update_mask

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class GAT_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GAT_Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.pointer = PointerAttention(8, input_dim, hidden_dim)

        # +1 to adjust for the concatenated capacity
        self.fc = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize parameters with Xavier for weights and zeros for biases.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Typically applies to weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:  # Check if it's a bias term
                nn.init.constant_(param, 0)  # Initialize biases to zero

    def forward(self, encoder_inputs, pool, capacity, demand, n_steps, T, greedy, feasible_mask=None):
        """
        Args:
            encoder_inputs: Tensor of shape (batch_size, n_nodes, hidden_dim)
            pool: Graph embedding of shape (batch_size, hidden_dim)
            capacity: (batch_size, 1) or (batch_size,) current remaining capacity per batch or initial cap
            demand: (batch_size, n_nodes)
            feasible_mask: Optional[Tensor] of shape (batch_size, n_nodes), 1 for feasible, 0 for infeasible
        """
        # Ensure expected shapes
        if encoder_inputs.dim() != 3:
            raise ValueError(f"encoder_inputs must be [B, N, H], got {encoder_inputs.shape}")

        device = encoder_inputs.device  # ensure the tensors are on the same device

        batch_size = encoder_inputs.size(0)
        seq_len = encoder_inputs.size(1)

        # Initialize the mask and mask1
        mask1 = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
        mask = encoder_inputs.new_zeros(batch_size, seq_len, device=device)

        # Normalize capacity shapes
        if capacity.dim() == 1:
            dynamic_capacity = capacity.unsqueeze(1).expand(batch_size, 1).to(device)
        else:
            dynamic_capacity = capacity.to(device)
        demands = demand.to(device)

        # Initialize the index tensor to keep track of the visited nodes
        index = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Initialize the log probabilities and actions tensors
        log_ps = []
        actions = []

        for i in range(n_steps):
            # If all non-depot nodes are marked done in mask1, break
            if not mask1[:, 1:].eq(0).any():
                break

            # First input is depot embedding
            if i == 0:
                _input = encoder_inputs[:, 0, :]  # depot (batch_size, hidden_dim)

            # pool + cat(first_node,current_node)
            decoder_input = torch.cat([_input, dynamic_capacity], -1)
            decoder_input = self.fc(decoder_input)
            pool_proj = self.fc1(pool.to(device))
            decoder_input = decoder_input + pool_proj

            # If it is the first step, update the mask to avoid visiting the depot again
            if i == 0:
                mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)

            # Combine internal mask with external feasibility constraints (hard-mask infeasible)
            combined_mask = mask
            if feasible_mask is not None:
                ext_infeasible = (~feasible_mask.bool()).clone()
                ext_infeasible[:, 0] = False  # Always allow depot
                combined_mask = mask | ext_infeasible

            # Compute the probability distribution with combined masking
            p = self.pointer(decoder_input, encoder_inputs, combined_mask, T)

            # Calculate the probability distribution for sampling
            dist = Categorical(p)
            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()

            actions.append(index.data.unsqueeze(1))
            log_p = dist.log_prob(index)
            is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
            log_p = log_p * (1. - is_done)

            log_ps.append(log_p.unsqueeze(1))

            # Update capacity and masks
            base_capacity = capacity[0].item() if capacity.dim() > 0 else float(dynamic_capacity[0, 0].item())
            dynamic_capacity = update_state(demands, dynamic_capacity, index.unsqueeze(-1), base_capacity)
            mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)

            # Gather the embedding of the selected node for next step
            _input = torch.gather(
                encoder_inputs, 1,
                index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, encoder_inputs.size(2))
            ).squeeze(1)

        # Concatenate the actions and log probabilities
        if log_ps:
            log_ps = torch.cat(log_ps, dim=1)
            log_p = log_ps.sum(dim=1)  # (batch_size,)
            actions = torch.cat(actions, dim=1)
        else:
            log_p = torch.zeros(batch_size, device=device)
            actions = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        return actions, log_p
