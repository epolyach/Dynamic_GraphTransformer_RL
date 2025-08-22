function validate_cvrp_solution(problem_matrix, solution_string)
% VALIDATE_CVRP_SOLUTION - Visualize CVRP problem and validate solution
%
% Usage:
%   problem_matrix = [0.08 0.28 0.41 0.67 0.23 0.36 0.07;
%                     0.40 0.48 0.61 0.63 0.94 0.38 0.40;
%                     0 7 6 10 5 2 9];
%   solution_string = 'exact_milp cost/route:   1.9583 [0, 6, 0, 1, 5, 2, 3, 4, 0]';
%   validate_cvrp_solution(problem_matrix, solution_string);

    % Parse the problem matrix
    x_coords = problem_matrix(1, :);
    y_coords = problem_matrix(2, :);
    demands = problem_matrix(3, :);
    n_nodes = length(x_coords);
    
    % Parse the solution string
    [solver_name, reported_total_cost, route] = parse_solution_string(solution_string);
    
    % Calculate Euclidean distance matrix
    dist_matrix = zeros(n_nodes, n_nodes);
    for i = 1:n_nodes
        for j = 1:n_nodes
            dist_matrix(i, j) = sqrt((x_coords(i) - x_coords(j))^2 + (y_coords(i) - y_coords(j))^2);
        end
    end
    
    % Calculate actual cost from the route
    actual_cost = 0;
    for i = 1:(length(route) - 1)
        from_node = route(i) + 1;  % Convert to 1-based indexing
        to_node = route(i + 1) + 1;
        actual_cost = actual_cost + dist_matrix(from_node, to_node);
    end
    
    % Calculate cost per customer for comparison
    num_customers = sum(demands > 0) - 1;  % Exclude depot
    actual_cost_per_customer = actual_cost / num_customers;
    reported_cost_per_customer = reported_total_cost / num_customers;
    
    % Create visualization
    figure('Position', [100, 100, 800, 600]);
    hold on;
    
    % Plot depot (node 0)
    depot_idx = find(demands == 0, 1);
    plot(x_coords(depot_idx), y_coords(depot_idx), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
    
    % Plot customers
    customer_indices = find(demands > 0);
    plot(x_coords(customer_indices), y_coords(customer_indices), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'blue');
    
    % Plot the route
    colors = ['g', 'm', 'c', 'k', 'y'];  % Different colors for different vehicle routes
    vehicle_num = 1;
    
    for i = 1:(length(route) - 1)
        from_node = route(i) + 1;  % Convert to 1-based indexing
        to_node = route(i + 1) + 1;
        
        % If returning to depot, use different line style
        if route(i + 1) == 0 && i < length(route) - 1
            % End of a vehicle route
            plot([x_coords(from_node), x_coords(to_node)], [y_coords(from_node), y_coords(to_node)], ...
                 [colors(mod(vehicle_num-1, length(colors)) + 1), '--'], 'LineWidth', 2);
            vehicle_num = vehicle_num + 1;
        else
            % Regular route segment
            plot([x_coords(from_node), x_coords(to_node)], [y_coords(from_node), y_coords(to_node)], ...
                 colors(mod(vehicle_num-1, length(colors)) + 1), 'LineWidth', 2);
        end
        
        % Add arrow to show direction
        dx = x_coords(to_node) - x_coords(from_node);
        dy = y_coords(to_node) - y_coords(from_node);
        if abs(dx) > 1e-10 || abs(dy) > 1e-10
            mid_x = x_coords(from_node) + 0.7 * dx;
            mid_y = y_coords(from_node) + 0.7 * dy;
            arrow_scale = 0.03;
            quiver(mid_x, mid_y, dx * arrow_scale, dy * arrow_scale, 0, ...
                   'Color', colors(mod(vehicle_num-1, length(colors)) + 1), 'LineWidth', 1.5, 'MaxHeadSize', 0.8);
        end
    end
    
    % Add node labels and demands
    for i = 1:n_nodes
        if demands(i) == 0
            % No label for depot - the star marker is sufficient
        else
            text(x_coords(i) + 0.02, y_coords(i) + 0.02, sprintf('%d (d=%d)', i-1, demands(i)), 'FontSize', 9);
        end
    end
    
    grid on;
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    title(sprintf('%s CVRP Solution', solver_name));
    axis([0 1 0 1]);
    axis square;
    legend({'Depot', 'Customers'}, 'Location', 'best');
    
    % Add cost information as text
    total_cost_diff = abs(actual_cost - reported_total_cost);
    cost_match = total_cost_diff < 1e-4;
    
    % Single line cost comparison at bottom
    if cost_match
        text(0.02, 0.02, sprintf('Costs: Reported %.4f | Calculated %.4f | PASS', reported_total_cost, actual_cost), ...
             'Units', 'normalized', 'FontSize', 11, 'BackgroundColor', 'green', 'FontWeight', 'bold', 'Color', 'white');
    else
        text(0.02, 0.02, sprintf('Costs: Reported %.4f | Calculated %.4f | FAIL (diff=%.6f)', reported_total_cost, actual_cost, total_cost_diff), ...
             'Units', 'normalized', 'FontSize', 11, 'BackgroundColor', 'red', 'FontWeight', 'bold', 'Color', 'white');
    end
    
    % Print results to console
    fprintf('\n=== CVRP Solution Validation ===\n');
    fprintf('Solver: %s\n', solver_name);
    fprintf('Route: %s\n', mat2str(route));
    fprintf('Number of customers: %d\n', num_customers);
    fprintf('Reported total cost: %.6f\n', reported_total_cost);
    fprintf('Calculated total cost: %.6f\n', actual_cost);
    fprintf('Reported cost per customer: %.6f\n', reported_cost_per_customer);
    fprintf('Calculated cost per customer: %.6f\n', actual_cost_per_customer);
    fprintf('Total cost difference: %.6f\n', total_cost_diff);
    if cost_match
        fprintf('Cost validation: PASS\n');
    else
        fprintf('Cost validation: FAIL\n');
    end
    fprintf('================================\n\n');
    
end

function [solver_name, total_cost, route] = parse_solution_string(solution_string)
% Parse solution string like "exact_milp cost/route: 1.9583 [0, 6, 0, 1, 5, 2, 3, 4, 0]"
    
    % Extract solver name (everything before " cost/route:")
    cost_pos = strfind(solution_string, ' cost/route:');
    solver_name = solution_string(1:cost_pos-1);
    
    % Extract total cost
    bracket_pos = strfind(solution_string, '[');
    cost_part = solution_string(cost_pos+12:bracket_pos-2);  % +12 for " cost/route:" length
    total_cost = str2double(strtrim(cost_part));
    
    % Extract route
    route_part = solution_string(bracket_pos:end);
    % Remove brackets and parse as numbers
    route_part = strrep(route_part, '[', '');
    route_part = strrep(route_part, ']', '');
    route_part = strrep(route_part, ',', ' ');
    route_numbers = str2num(route_part);
    route = route_numbers;
    
end

% Example usage:
% problem_matrix = [0.08 0.28 0.41 0.67 0.23 0.36 0.07;
%                   0.40 0.48 0.61 0.63 0.94 0.38 0.40;
%                   0 7 6 10 5 2 9];
% solution_string = 'exact_milp cost/route:   1.9583 [0, 6, 0, 1, 5, 2, 3, 4, 0]';
% validate_cvrp_solution(problem_matrix, solution_string);
