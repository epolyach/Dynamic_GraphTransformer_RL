import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.TransformerAttention import TransformerAttention

class PointerAttention(nn.Module):
    """
    This class is the single head attention layer for the pointer network.
    """
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(PointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = TransformerAttention(n_heads, 1, input_dim, hidden_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """
        This function initializes the parameters of the encoder.
        It's using the Xavier initialization over Orthogonal initialization because it's more suitable for the ReLU activation function applied to the output of the attention layer.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Typically applies to weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:  # Check if it's a bias term
                nn.init.constant_(param, 0)  # Initialize biases to zero


    def forward(self, state_t, context, mask, T):
        '''
        This function computes the attention scores, applies the mask, computes the nodes probabilities and returns them as a softmax score.
        - Applies a clipping to the attention scores to avoid numerical instability.
        
        Args:
        - state_t: The current state of the model. (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        - context: The context to attend to. (batch_size,n_nodes,input_dim)
        - mask: The mask to apply to the attention scores. (batch_size,n_nodes)
        - T: The temperature for the softmax function.
        
        returns:
        - softmax_score: The softmax scores of the attention layer. (batch_size, n_nodes)
        '''
        
        # Check for NaN in inputs
        if torch.isnan(state_t).any():
            # print(f"[DEBUG] NaN detected in state_t: {torch.isnan(state_t).sum()} values")
            state_t = torch.nan_to_num(state_t, nan=0.0)
        
        if torch.isnan(context).any():
            # print(f"[DEBUG] NaN detected in context: {torch.isnan(context).sum()} values")
            context = torch.nan_to_num(context, nan=0.0)

        x = self.mhalayer(state_t, context, mask)
        
        # Check for NaN after mhalayer
        if torch.isnan(x).any():
            # print(f"[DEBUG] NaN detected after mhalayer: {torch.isnan(x).sum()} values")
            x = torch.nan_to_num(x, nan=0.0)

        batch_size, n_nodes, input_dim = context.size()
        Q = x.reshape(batch_size, 1, -1)
        K = self.k(context).reshape(batch_size, n_nodes, -1)
        
        # Check for NaN in K
        if torch.isnan(K).any():
            # print(f"[DEBUG] NaN detected in K: {torch.isnan(K).sum()} values")
            K = torch.nan_to_num(K, nan=0.0)
        
        # Compute the compatibility scores
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))  # Size: (batch_size, 1, n_nodes)
        compatibility = compatibility.squeeze(1)
        
        # Check for NaN in compatibility
        if torch.isnan(compatibility).any():
            # print(f"[DEBUG] NaN detected in compatibility: {torch.isnan(compatibility).sum()} values")
            compatibility = torch.nan_to_num(compatibility, nan=0.0)

        # Non-linear transformation with clipping for stability
        x = torch.tanh(compatibility)
        
        # Scaling the values to avoid numerical instability (reduced scale)
        x = x * 5.0  # Reduced from 10 to 5 for better stability
        
        # Apply the mask
        x = x.masked_fill(mask.bool(), float("-inf"))

        # Ensure at least one feasible option per row (fallback to depot)
        all_masked = torch.isinf(x).all(dim=-1)
        if all_masked.any():
            # Set all to -inf then depot (index 0) to 0 for those rows
            x = x.clone()
            x[all_masked] = float("-inf")
            x[all_masked, 0] = 0.0
        
        # Clamp temperature to avoid extreme values
        T = max(0.1, min(10.0, T))
        
        # Compute the softmax scores with numerical stability
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_stable = x - x_max  # Subtract max for numerical stability
        
        # Check if all values are -inf
        all_neg_inf = torch.isinf(x_stable).all(dim=-1)
        if all_neg_inf.any():
            # Create uniform distribution for rows with all -inf
            scores = torch.zeros_like(x)
            scores[all_neg_inf] = 1.0 / x.size(-1)  # Uniform distribution
            scores[~all_neg_inf] = F.softmax(x_stable[~all_neg_inf] / T, dim=-1)
        else:
            scores = F.softmax(x_stable / T, dim=-1)
        
        # Final NaN check
        if torch.isnan(scores).any():
            print(f"[ERROR] NaN detected in final scores! Replacing with uniform distribution")
            nan_rows = torch.isnan(scores).any(dim=-1)
            scores[nan_rows] = 1.0 / scores.size(-1)  # Uniform distribution for NaN rows
            
        return scores
