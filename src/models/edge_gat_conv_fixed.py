"""
Fixed EdgeGATConv implementation using PyTorch Geometric's MessagePassing framework.
This implementation properly handles attention computation with softmax normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeGATConvFixed(MessagePassing):
    """
    Custom GAT layer that includes edge features in the computation of attention coefficients.
    Uses PyTorch Geometric's MessagePassing framework for proper aggregation.
    """
    
    def __init__(self, node_channels: int, hidden_dim: int, edge_dim: int, 
                 negative_slope: float = 0.2, dropout: float = 0.6, concat: bool = True):
        super(EdgeGATConvFixed, self).__init__(aggr='add' if concat else 'mean')
        self.node_channels = node_channels
        self.hidden_dim = hidden_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat
        
        self.fc = nn.Linear(node_channels, hidden_dim)
        self.att_vector = nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Xavier initialization"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, size=None) -> torch.Tensor:
        """
        Forward pass with proper message passing.
        
        Args:
            x: Node features [num_nodes, node_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        # Project node features
        x = self.fc(x)
        
        # Use PyTorch Geometric's propagate method
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
    
    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        """
        Compute messages with attention coefficients.
        This is called by the propagate method.
        """
        # Concatenate source, target, and edge features
        x_cat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Compute attention coefficients
        alpha = self.att_vector(x_cat)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Apply softmax normalization across all incoming edges for each node
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Return weighted messages
        return x_j * alpha
    
    def update(self, aggr_out):
        """Update node embeddings."""
        return aggr_out
