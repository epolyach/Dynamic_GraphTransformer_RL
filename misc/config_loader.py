#!/usr/bin/env python3
"""
Configuration loader for CVRP benchmarks.
Ensures both CPU and GPU benchmarks use identical parameters.
"""

import json
import os
from typing import Dict, Any

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def get_instance_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract instance generation parameters from config."""
    instance_config = config["instance_generation"]
    return {
        "capacity": instance_config["capacity"],
        "demand_range": [instance_config["demand_min"], instance_config["demand_max"]],
        "coord_range": instance_config["coord_range"]
    }

def get_benchmark_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract benchmark parameters from config."""
    return config["benchmark_settings"]

def validate_config(config: Dict[str, Any]) -> None:
    """Validate that config contains required parameters."""
    required_sections = ["instance_generation", "benchmark_settings", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    instance_params = ["capacity", "demand_min", "demand_max", "coord_range"]
    for param in instance_params:
        if param not in config["instance_generation"]:
            raise ValueError(f"Missing required instance parameter: {param}")
    
    print(f"✅ Config validation passed")
    print(f"   - Capacity: {config['instance_generation']['capacity']}")
    print(f"   - Demand range: [{config['instance_generation']['demand_min']}, {config['instance_generation']['demand_max']}]")
    print(f"   - Coordinate range: [0, {config['instance_generation']['coord_range']}] normalized to [0, 1]")

if __name__ == "__main__":
    # Test config loading
    config = load_config()
    validate_config(config)
    print("Config loaded successfully!")
