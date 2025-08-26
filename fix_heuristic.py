#!/usr/bin/env python3
"""Fix heuristic_or.py to handle more vehicles and add debug output"""

with open('solvers/heuristic_or.py', 'r') as f:
    content = f.read()

# Add verbose output
old_line = "    for n_vehicles in range(min_vehicles, max_vehicles + 1):"
new_line = """    if verbose:
        print(f"Heuristic solver: n_customers={n_customers}, total_demand={total_demand}, capacity={capacity}")
        print(f"  Will try {min_vehicles} to {max_vehicles} vehicles")
    
    for n_vehicles in range(min_vehicles, max_vehicles + 1):
        if verbose:
            print(f"  Trying {n_vehicles} vehicles...", end=" ")"""

content = content.replace(old_line, new_line)

# Add success/failure message
old_fail = '            continue'
new_fail = """            if verbose:
                print("No solution found")
            continue"""

content = content.replace('            continue', new_fail, 1)

with open('solvers/heuristic_or.py', 'w') as f:
    f.write(content)

print("Fixed heuristic_or.py")
