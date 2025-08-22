#!/usr/bin/env python3
"""
Download and parse standard CVRP benchmark instances with known optimal solutions.
This will help us validate our exact algorithms.
"""

import requests
import numpy as np
import os
import re
import urllib3
from typing import Dict, List, Tuple, Optional

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url: str, filename: str) -> bool:
    """Download a file from URL"""
    try:
        print(f"Downloading {filename} from {url}")
        response = requests.get(url, verify=False)  # Skip SSL verification
        response.raise_for_status()
        
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def parse_vrp_instance(filename: str) -> Optional[Dict]:
    """Parse a VRP instance in TSPLIB format"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        instance = {}
        section = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('EOF'):
                continue
                
            if ':' in line and not line.startswith(' '):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'NAME':
                    instance['name'] = value
                elif key == 'TYPE':
                    instance['type'] = value
                elif key == 'DIMENSION':
                    instance['dimension'] = int(value)
                elif key == 'CAPACITY':
                    instance['capacity'] = int(value)
                elif key == 'EDGE_WEIGHT_TYPE':
                    instance['edge_weight_type'] = value
            
            elif line in ['NODE_COORD_SECTION', 'DEMAND_SECTION', 'DEPOT_SECTION']:
                section = line
                instance[section] = []
            
            elif section:
                if section == 'NODE_COORD_SECTION':
                    parts = line.split()
                    if len(parts) == 3:
                        node_id, x, y = parts
                        instance[section].append([int(node_id), float(x), float(y)])
                
                elif section == 'DEMAND_SECTION':
                    parts = line.split()
                    if len(parts) == 2:
                        node_id, demand = parts
                        instance[section].append([int(node_id), int(demand)])
                
                elif section == 'DEPOT_SECTION':
                    if line.strip() != '-1':
                        instance[section].append(int(line))
        
        return instance
    
    except Exception as e:
        print(f"❌ Failed to parse {filename}: {e}")
        return None

def convert_to_our_format(vrp_instance: Dict) -> Dict:
    """Convert TSPLIB format to our internal format"""
    
    # Extract demands first to determine size
    demand_data = vrp_instance.get('DEMAND_SECTION', [])
    demands_dict = {node_id: demand for node_id, demand in demand_data}
    
    if not demands_dict:
        print(f"⚠️ No demand data found for {vrp_instance.get('name', 'Unknown')}")
        return None
    
    max_node = max(demands_dict.keys())
    n = max_node + 1
    
    demands = np.zeros(n, dtype=int)
    for node_id, demand in demands_dict.items():
        demands[node_id] = demand
    
    # Handle coordinates or explicit distances
    coords_data = vrp_instance.get('NODE_COORD_SECTION', [])
    
    if coords_data:
        # Case 1: Coordinates provided (EUC_2D)
        coords_dict = {node_id: (x, y) for node_id, x, y in coords_data}
        coords = np.zeros((n, 2))
        for node_id, (x, y) in coords_dict.items():
            coords[node_id] = [x, y]
        
        # Compute distance matrix (Euclidean)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = np.linalg.norm(coords[i] - coords[j])
    else:
        # Case 2: Explicit distances (skip instances without coordinates for now)
        print(f"⚠️ Skipping {vrp_instance.get('name', 'Unknown')}: EXPLICIT distances not yet supported")
        return None
        
        # TODO: Parse EDGE_WEIGHT_SECTION for explicit distances
        # coords = np.random.rand(n, 2) * 100  # Generate dummy coordinates
        # distances = ... # Parse explicit distance matrix
    
    return {
        'name': vrp_instance.get('name', 'Unknown'),
        'coords': coords,
        'demands': demands,
        'distances': distances,
        'capacity': vrp_instance.get('capacity', 100),
        'num_customers': n - 1,  # Excluding depot
        'instance_type': 'benchmark'
    }

def download_benchmark_instances():
    """Download standard CVRP benchmark instances"""
    
    # Create benchmarks directory
    os.makedirs('benchmarks', exist_ok=True)
    os.chdir('benchmarks')
    
    # Known benchmark instances with optimal solutions (N=10-31 for testing exact algorithms)
    instances = [
        # Very small instances for exact algorithm testing
        # Format: (name, optimal_cost, description)
        # Note: Some might not exist, we'll try common ones
        
        # Augerat instances (A-series)
        {'name': 'A-n32-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp', 'optimal_cost': 784, 'description': '32 customers, 5 vehicles'},
        {'name': 'A-n33-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n33-k5.vrp', 'optimal_cost': 661, 'description': '33 customers, 5 vehicles'},
        {'name': 'A-n33-k6', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n33-k6.vrp', 'optimal_cost': 742, 'description': '33 customers, 6 vehicles'},
        {'name': 'A-n34-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n34-k5.vrp', 'optimal_cost': 778, 'description': '34 customers, 5 vehicles'},
        {'name': 'A-n36-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n36-k5.vrp', 'optimal_cost': 799, 'description': '36 customers, 5 vehicles'},
        {'name': 'A-n37-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n37-k5.vrp', 'optimal_cost': 669, 'description': '37 customers, 5 vehicles'},
        {'name': 'A-n37-k6', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n37-k6.vrp', 'optimal_cost': 949, 'description': '37 customers, 6 vehicles'},
        {'name': 'A-n38-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n38-k5.vrp', 'optimal_cost': 730, 'description': '38 customers, 5 vehicles'},
        {'name': 'A-n39-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n39-k5.vrp', 'optimal_cost': 822, 'description': '39 customers, 5 vehicles'},
        {'name': 'A-n39-k6', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n39-k6.vrp', 'optimal_cost': 831, 'description': '39 customers, 6 vehicles'},
        
        # B-series instances (smaller capacity, good for testing)
        {'name': 'B-n31-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/B/B-n31-k5.vrp', 'optimal_cost': 672, 'description': '31 customers, 5 vehicles'},
        {'name': 'B-n34-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/B/B-n34-k5.vrp', 'optimal_cost': 788, 'description': '34 customers, 5 vehicles'},
        {'name': 'B-n35-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/B/B-n35-k5.vrp', 'optimal_cost': 955, 'description': '35 customers, 5 vehicles'},
        {'name': 'B-n38-k6', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/B/B-n38-k6.vrp', 'optimal_cost': 805, 'description': '38 customers, 6 vehicles'},
        {'name': 'B-n39-k5', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/B/B-n39-k5.vrp', 'optimal_cost': 549, 'description': '39 customers, 5 vehicles'},
        
        # P-series instances (from Pearn et al., smaller problems)
        {'name': 'P-n16-k8', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n16-k8.vrp', 'optimal_cost': 450, 'description': '16 customers, 8 vehicles'},
        {'name': 'P-n19-k2', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n19-k2.vrp', 'optimal_cost': 212, 'description': '19 customers, 2 vehicles'},
        {'name': 'P-n20-k2', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n20-k2.vrp', 'optimal_cost': 216, 'description': '20 customers, 2 vehicles'},
        {'name': 'P-n21-k2', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n21-k2.vrp', 'optimal_cost': 211, 'description': '21 customers, 2 vehicles'},
        {'name': 'P-n22-k2', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n22-k2.vrp', 'optimal_cost': 216, 'description': '22 customers, 2 vehicles'},
        {'name': 'P-n22-k8', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n22-k8.vrp', 'optimal_cost': 603, 'description': '22 customers, 8 vehicles'},
        {'name': 'P-n23-k8', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/P/P-n23-k8.vrp', 'optimal_cost': 529, 'description': '23 customers, 8 vehicles'},
        
        # E-series instances (Eilon et al., smaller problems)
        {'name': 'E-n13-k4', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/E/E-n13-k4.vrp', 'optimal_cost': 247, 'description': '13 customers, 4 vehicles'},
        {'name': 'E-n22-k4', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/E/E-n22-k4.vrp', 'optimal_cost': 375, 'description': '22 customers, 4 vehicles'},
        {'name': 'E-n23-k3', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/E/E-n23-k3.vrp', 'optimal_cost': 569, 'description': '23 customers, 3 vehicles'},
        {'name': 'E-n30-k3', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/E/E-n30-k3.vrp', 'optimal_cost': 534, 'description': '30 customers, 3 vehicles'},
        {'name': 'E-n31-k7', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/E/E-n31-k7.vrp', 'optimal_cost': 379, 'description': '31 customers, 7 vehicles'},
        
        # F-series instances (Fisher, small problems)
        {'name': 'F-n25-k4', 'url': 'http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/F/F-n25-k4.vrp', 'optimal_cost': 1079, 'description': '25 customers, 4 vehicles'},
    ]
    
    downloaded_instances = []
    
    print("="*60)
    print("DOWNLOADING CVRP BENCHMARK INSTANCES")
    print("="*60)
    
    for instance in instances:
        filename = f"{instance['name']}.vrp"
        
        if download_file(instance['url'], filename):
            # Parse the instance
            vrp_data = parse_vrp_instance(filename)
            if vrp_data:
                # Convert to our format
                converted = convert_to_our_format(vrp_data)
                if converted:
                    instance['data'] = converted
                    downloaded_instances.append(instance)
                    
                    print(f"  📊 {instance['name']}: {converted['num_customers']} customers, "
                          f"capacity {converted['capacity']}, optimal cost {instance['optimal_cost']}")
                else:
                    print(f"  ⚠️ Skipped {instance['name']}: conversion failed")
            else:
                print(f"  ❌ Failed to parse {filename}")
        
        print()
    
    os.chdir('..')  # Return to parent directory
    
    print(f"✅ Successfully downloaded {len(downloaded_instances)} benchmark instances")
    return downloaded_instances

if __name__ == "__main__":
    instances = download_benchmark_instances()
