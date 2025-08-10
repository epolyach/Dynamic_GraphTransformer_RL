"""
Dynamic Graph Transformer Models Package

This package has been cleaned to contain only essential, actively used models.
All experimental and unused models have been moved to models_backup/.

Core models are now defined inline in run_comparative_study.py for better maintainability.
This directory is reserved for future modular model implementations.

Available in models_backup/:
- DynamicGraphTransformerModel.py
- EdgeGATConv.py  
- GAT_Encoder.py
- GAT_Decoder.py
- Model.py
- PointerAttention.py
- TransformerAttention.py
- ablation_models.py
- dynamic_updater.py
- graph_transformer.py
- mask_capacity.py
- model_factory.py
"""

# This package is currently empty - all models are inline in run_comparative_study.py
# Add modular model implementations here as needed

__version__ = "2.0.0"
__all__ = []
