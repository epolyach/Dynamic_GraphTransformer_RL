"""
Model factory for creating CVRP model instances.

This factory provides a centralized way to create model instances with consistent
configurations and parameter handling.
"""

import torch.nn as nn
from typing import Dict, Any

from src.models.legacy_gat import LegacyGATModel
from src.models.gt import GraphTransformerNetwork
from src.models.greedy_gt import GraphTransformerGreedy
from src.models.dgt import DynamicGraphTransformerNetwork
from src.models.gt_lite import GraphTransformerLite
from src.models.gt_ultra import GraphTransformerUltra
from src.models.gt_super import GraphTransformerSuper
from src.models.dgt_lite import DynamicGraphTransformerLite
from src.models.dgt_ultra import DynamicGraphTransformerUltra
from src.models.dgt_super import DynamicGraphTransformerSuper


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
    edge_div = config['model']['edge_embedding_divisor']
    
    # Create model based on name
    if model_name == 'GAT+RL':
        # Legacy GAT model with exact architecture from GAT_RL project
        return LegacyGATModel(
            node_input_dim=input_dim,
            edge_input_dim=1,  # Edge features are distances
            hidden_dim=hidden_dim,
            edge_dim=16,  # Legacy default
            layers=4,  # Legacy uses 4 layers
            negative_slope=0.2,  # Legacy default
            dropout=0.6,  # Legacy default
            config=config
        )
    elif model_name == 'GT-Greedy':
        return GraphTransformerGreedy(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'GT+RL':
        return GraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'DGT+RL':
        return DynamicGraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    
    # Lightweight variants
    elif model_name == 'GT-Lite+RL':
        return GraphTransformerLite(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'GT-Ultra+RL':
        return GraphTransformerUltra(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'GT-Super+RL':
        return GraphTransformerSuper(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'DGT-Lite+RL':
        return DynamicGraphTransformerLite(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'DGT-Ultra+RL':
        return DynamicGraphTransformerUltra(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif model_name == 'DGT-Super+RL':
        return DynamicGraphTransformerSuper(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    
    else:
        raise ValueError(f'Unknown model name: {model_name}. Supported models: '
                        f'GAT+RL (Legacy GAT), GT-Greedy, GT+RL, DGT+RL, '
                        f'GT-Lite+RL, GT-Ultra+RL, GT-Super+RL, '
                        f'DGT-Lite+RL, DGT-Ultra+RL, DGT-Super+RL')


def get_supported_models():
    """Get list of supported model names."""
    return [
        'GAT+RL',  # Legacy GAT with edge features and 8-head pointer attention
        'GT-Greedy', 'GT+RL', 'DGT+RL',
        'GT-Lite+RL', 'GT-Ultra+RL', 'GT-Super+RL',
        'DGT-Lite+RL', 'DGT-Ultra+RL', 'DGT-Super+RL'
    ]
