"""
Balanced Pointer Attention mechanism for GAT model.
This version uses controlled scaling without the extreme tanh + 10x combination.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedPointerAttention(nn.Module):
    """
    Pointer attention mechanism with balanced scaling.
    Avoids the problematic tanh + 10x scaling while maintaining sufficient expressiveness.
    """
    
    def __init__(self, n_heads: int, input_dim: int, hidden_dim: int):
        super(BalancedPointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Use moderate scaling factor
        self.scale = 1 / math.sqrt(hidden_dim / n_heads)
        
        # Linear projections
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Optional: Learnable temperature parameter
        self.log_temp = nn.Parameter(torch.zeros(1))
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights with appropriate scaling."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # Use Kaiming initialization for better gradient flow
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, state_t: torch.Tensor, context: torch.Tensor, 
                mask: torch.Tensor, external_temp: float = 1.0) -> torch.Tensor:
        """
        Compute pointer attention scores with balanced scaling.
        
        Args:
            state_t: Current state [batch_size, 1, input_dim]
            context: Node embeddings [batch_size, n_nodes, input_dim]
            mask: Feasibility mask [batch_size, n_nodes]
            external_temp: External temperature for exploration
        
        Returns:
            Probability distribution [batch_size, n_nodes]
        """
        batch_size, n_nodes, _ = context.size()
        
        # Project queries and keys
        Q = self.w(state_t)  # [batch_size, 1, hidden_dim]
        K = self.k(context)  # [batch_size, n_nodes, hidden_dim]
        V = self.v(context)  # [batch_size, n_nodes, hidden_dim]
        
        # Reshape for multi-head attention
        head_dim = self.hidden_dim // self.n_heads
        Q = Q.view(batch_size, 1, self.n_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, n_nodes, self.n_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, n_nodes, self.n_heads, head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(2)  # [batch_size, n_heads, n_nodes]
        
        # Average over heads
        scores = scores.mean(dim=1)  # [batch_size, n_nodes]
        
        # Apply controlled non-linearity (less extreme than tanh + 10x)
        # Option 1: Moderate scaling with tanh
        scores = 5.0 * torch.tanh(scores / 2.0)
        
        # Option 2: Direct scaling without tanh (comment out Option 1 to use)
        # scores = scores * 2.0
        
        # Apply mask
        scores = scores.masked_fill(mask.bool(), float("-inf"))
        
        # Handle edge case where all values are masked
        all_masked = (scores == float("-inf")).all(dim=-1, keepdim=True)
        if all_masked.any():
            # Unmask depot as fallback
            scores = torch.where(
                all_masked.expand_as(scores),
                torch.zeros_like(scores),
                scores
            )
            scores[:, 0] = torch.where(
                all_masked.squeeze(-1),
                torch.ones_like(scores[:, 0]),  # Prefer depot when all masked
                scores[:, 0]
            )
        
        # Apply temperature (combine learned and external)
        learned_temp = torch.exp(self.log_temp)
        effective_temp = external_temp * learned_temp
        
        # Compute probabilities
        probs = F.softmax(scores / effective_temp, dim=-1)
        
        # Safety check for NaN
        if torch.isnan(probs).any():
            # Fallback: uniform distribution over unmasked nodes
            valid_mask = ~mask.bool()
            probs = valid_mask.float()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        
        return probs


class LegacyCompatiblePointerAttention(nn.Module):
    """
    Pointer attention that closely matches the legacy implementation
    but with improved numerical stability.
    """
    
    def __init__(self, n_heads: int, input_dim: int, hidden_dim: int):
        super(LegacyCompatiblePointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Match legacy normalization
        self.norm = 1 / math.sqrt(hidden_dim)
        
        # Single key projection like legacy
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Multi-head attention layer for state processing
        # Assuming TransformerAttention exists in your codebase
        from .gat import TransformerAttention
        self.mhalayer = TransformerAttention(n_heads, 3, input_dim, hidden_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Xavier initialization like legacy."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, state_t: torch.Tensor, context: torch.Tensor, 
                mask: torch.Tensor, T: float) -> torch.Tensor:
        """
        Legacy-compatible forward pass with improved stability.
        
        Args:
            state_t: Current state [batch_size, 1, input_dim*3]
            context: Node embeddings [batch_size, n_nodes, input_dim]
            mask: Feasibility mask [batch_size, n_nodes]
            T: Temperature for softmax
        
        Returns:
            Probability distribution [batch_size, n_nodes]
        """
        # Process state through multi-head attention
        x = self.mhalayer(state_t, context, mask)
        
        batch_size, n_nodes, _ = context.size()
        Q = x.reshape(batch_size, 1, -1)
        K = self.k(context).reshape(batch_size, n_nodes, -1)
        
        # Compute compatibility scores
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))
        compatibility = compatibility.squeeze(1)
        
        # Apply the legacy transformation with gradient-friendly modifications
        # Option 1: Reduce the scaling factor
        x = torch.tanh(compatibility) * 5.0  # Reduced from 10 to 5
        
        # Option 2: Use a different activation (comment out Option 1)
        # x = F.gelu(compatibility) * 3.0  # GELU is smoother than tanh
        
        # Apply mask
        x = x.masked_fill(mask.bool(), float("-inf"))
        
        # Add numerical stability check
        max_val = x.max(dim=-1, keepdim=True)[0]
        x = x - max_val  # Subtract max for numerical stability
        
        # Compute softmax with temperature
        scores = F.softmax(x / T, dim=-1)
        
        return scores
