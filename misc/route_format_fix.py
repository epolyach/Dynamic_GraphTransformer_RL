def clean_route_format(vehicle_routes):
    """Remove depot (0) from start and end of routes for validation"""
    cleaned_routes = []
    for route in vehicle_routes:
        if not route:
            continue
        # Remove depot from start and end
        cleaned_route = [node for node in route if node != 0]
        if cleaned_route:  # Only add non-empty routes
            cleaned_routes.append(cleaned_route)
    return cleaned_routes
