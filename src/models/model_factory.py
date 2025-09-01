"""
Model factory for creating CVRP model instances.

This factory provides a centralized way to create model instances with consistent
configurations and parameter handling.
"""

import torch.nn as nn
from typing import Dict, Any

from src.models.gat import GATModel
from src.models.gt import GraphTransformer
from src.models.greedy_gt import GraphTransformerGreedy
from src.models.dgt import DynamicGraphTransformerNetwork


def create_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """
    Create a model instance based on the model name and configuration.
    
    Args:
        model_name: Name of the model to create
        config: Configuration dictionary containing model parameters
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    # Extract model parameters from config
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    dropout = config['model']['transformer_dropout']
    ff_mult = config['model']['feedforward_multiplier']
    
    # GAT-specific parameters (with defaults for backward compatibility)
    gat_edge_dim = config['model'].get('gat_edge_dim', 16)
    gat_dropout = config['model'].get('gat_dropout', 0.6)
    gat_negative_slope = config['model'].get('gat_negative_slope', 0.2)
    
    # Create model based on name
    if model_name == 'GAT+RL':
        # GAT model with improved convergence
        # Uses GAT-specific parameters from config
        return GATModel(
            node_input_dim=input_dim,
            edge_input_dim=1,  # Edge features are distances
            hidden_dim=hidden_dim,
            edge_dim=gat_edge_dim,  # From config (default: 16)
            layers=num_layers,  # From config
            negative_slope=gat_negative_slope,  # From config (default: 0.2)
            dropout=gat_dropout,  # From config (default: 0.6)
            config=config
        )
    elif model_name == 'GT-Greedy':
        return GraphTransformerGreedy(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'GT+RL':
        return GraphTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'DGT+RL':
        return DynamicGraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    
    else:
        raise ValueError(f'Unknown model name: {model_name}. Supported models: '
                        f'GAT+RL, GT-Greedy, GT+RL, DGT+RL')


def get_supported_models():
    """Get list of supported model names."""
    return [
        'GAT+RL',  # GAT with improved convergence
        'GT-Greedy',  # Greedy baseline
        'GT+RL',  # Advanced Graph Transformer
        'DGT+RL'  # Dynamic Graph Transformer (ultimate model)
    ]


class ModelFactory:
    """Factory class for creating CVRP models."""
    
    @staticmethod
    def create_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
        """Create a model instance."""
        return create_model(model_name, config)
    
    @staticmethod
    def get_supported_models():
        """Get list of supported model names."""
        return get_supported_models()
