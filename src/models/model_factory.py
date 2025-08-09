"""
Model Factory for Dynamic Graph Transformer
Creates different model variants based on configuration
"""

import torch
from torch import nn
from typing import Dict, Any, Union
import yaml

from .Model import Model
from .DynamicGraphTransformerModel import DynamicGraphTransformerModel
from .pointer_rl import PointerRLModel


class ModelFactory:
    """
    Factory class for creating different model variants based on configuration
    """
    
    @staticmethod
    def create_model(config: Union[Dict[str, Any], str], **kwargs) -> nn.Module:
        """
        Create a model based on configuration
        
        Args:
            config: Either a config dictionary or path to config file
            **kwargs: Additional parameters to override config
            
        Returns:
            Instantiated model
        """
        # Load config if path is provided
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Extract model config
        model_config = config.get('model', {})
        
        # Override with kwargs
        model_config.update(kwargs)
        
        # Get model type
        model_type = model_config.get('type', 'dynamic_transformer')
        
        if model_type == 'basic_transformer':
            return ModelFactory._create_basic_transformer(model_config)
        elif model_type == 'dynamic_transformer':
            return ModelFactory._create_dynamic_transformer(model_config)
        elif model_type == 'pointer_rl':
            return ModelFactory._create_pointer_rl(model_config)
        elif model_type == 'legacy':
            return ModelFactory._create_legacy_model(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_basic_transformer(config: Dict[str, Any]) -> Model:
        """Create basic Graph Transformer model (replacement for GAT)"""
        encoder_config = config.get('encoder', {})
        
        return Model(
            node_input_dim=encoder_config.get('node_input_dim', 3),
            edge_input_dim=encoder_config.get('edge_input_dim', 1),
            hidden_dim=encoder_config.get('hidden_dim', 128),
            edge_dim=encoder_config.get('hidden_dim', 128),  # For compatibility
            layers=encoder_config.get('num_layers', 6),
            negative_slope=0.2,  # Not used in transformer, for compatibility
            dropout=encoder_config.get('dropout', 0.1)
        )

    @staticmethod
    def _create_pointer_rl(config: Dict[str, Any]) -> PointerRLModel:
        """Create Pointer + RL model (Graph Transformer encoder + Pointer decoder)."""
        encoder_config = config.get('encoder', {})
        pe_config = config.get('positional_encoding', {})
        
        return PointerRLModel(
            node_input_dim=encoder_config.get('node_input_dim', 3),
            edge_input_dim=encoder_config.get('edge_input_dim', 1),
            hidden_dim=encoder_config.get('hidden_dim', 128),
            num_heads=encoder_config.get('num_heads', 8),
            num_layers=encoder_config.get('num_layers', 6),
            dropout=encoder_config.get('dropout', 0.1),
            pe_type=pe_config.get('pe_type', 'sinusoidal'),
            pe_dim=pe_config.get('pe_dim', 64),
            max_distance=pe_config.get('max_distance', 100.0),
        )
    
    @staticmethod
    def _create_dynamic_transformer(config: Dict[str, Any]) -> DynamicGraphTransformerModel:
        """Create Dynamic Graph Transformer model with graph updates"""
        encoder_config = config.get('encoder', {})
        pe_config = config.get('positional_encoding', {})
        dynamic_config = config.get('dynamic_updates', {})
        problem_config = config.get('problem', {})
        
        return DynamicGraphTransformerModel(
            # Graph Transformer parameters
            node_input_dim=encoder_config.get('node_input_dim', 3),
            edge_input_dim=encoder_config.get('edge_input_dim', 1),
            hidden_dim=encoder_config.get('hidden_dim', 128),
            num_heads=encoder_config.get('num_heads', 8),
            num_layers=encoder_config.get('num_layers', 6),
            dropout=encoder_config.get('dropout', 0.1),
            
            # Positional encoding parameters
            pe_type=pe_config.get('pe_type', 'sinusoidal'),
            pe_dim=pe_config.get('pe_dim', 64),
            max_distance=pe_config.get('max_distance', 100.0),
            
            # Dynamic updates parameters
            use_dynamic_updates=dynamic_config.get('enabled', True),
            update_frequency=dynamic_config.get('update_frequency', 1),
            
            # Vehicle parameters
            vehicle_capacity=problem_config.get('vehicle_capacity', 50.0)
        )
    
    @staticmethod
    def _create_legacy_model(config: Dict[str, Any]) -> Model:
        """Create legacy GAT-based model for comparison"""
        # This would need the original GAT implementation
        # For now, return the basic transformer as fallback
        return ModelFactory._create_basic_transformer(config)
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'model_type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        # Add model-specific information
        if hasattr(model, 'encoder'):
            if hasattr(model.encoder, 'num_heads'):
                info['num_attention_heads'] = model.encoder.num_heads
            if hasattr(model.encoder, 'num_layers'):
                info['num_transformer_layers'] = model.encoder.num_layers
            if hasattr(model.encoder, 'hidden_dim'):
                info['hidden_dimension'] = model.encoder.hidden_dim
        
        if hasattr(model, 'use_dynamic_updates'):
            info['dynamic_updates_enabled'] = model.use_dynamic_updates
        
        return info


def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None, **kwargs) -> nn.Module:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file (optional)
        **kwargs: Additional parameters
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get config from checkpoint first
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path is not None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("No config found in checkpoint and no config_path provided")
    
    # Create model
    model = ModelFactory.create_model(config, **kwargs)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


if __name__ == "__main__":
    # Test model factory
    import tempfile
    import os
    
    # Create test config
    test_config = {
        'model': {
            'type': 'dynamic_transformer',
            'encoder': {
                'node_input_dim': 3,
                'edge_input_dim': 1,
                'hidden_dim': 128,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1
            },
            'positional_encoding': {
                'pe_type': 'sinusoidal',
                'pe_dim': 64,
                'max_distance': 100.0
            },
            'dynamic_updates': {
                'enabled': True,
                'update_frequency': 1
            }
        },
        'problem': {
            'vehicle_capacity': 50.0
        }
    }
    
    print("Testing Model Factory...")
    
    # Test creating dynamic transformer
    model1 = ModelFactory.create_model(test_config)
    info1 = ModelFactory.get_model_info(model1)
    print(f"Dynamic Transformer Model:")
    for key, value in info1.items():
        print(f"  {key}: {value}")
    print()
    
    # Test creating basic transformer
    test_config['model']['type'] = 'basic_transformer'
    model2 = ModelFactory.create_model(test_config)
    info2 = ModelFactory.get_model_info(model2)
    print(f"Basic Transformer Model:")
    for key, value in info2.items():
        print(f"  {key}: {value}")
    print()
    
    # Test saving and loading config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_config_path = f.name
    
    try:
        model3 = ModelFactory.create_model(temp_config_path)
        print("Successfully created model from config file!")
        
        # Test checkpoint save/load
        checkpoint = {
            'model_state_dict': model3.state_dict(),
            'config': test_config
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_checkpoint_path = f.name
        
        try:
            model4 = load_model_from_checkpoint(temp_checkpoint_path)
            print("Successfully loaded model from checkpoint!")
        finally:
            os.unlink(temp_checkpoint_path)
            
    finally:
        os.unlink(temp_config_path)
    
    print("Model Factory test completed successfully!")
