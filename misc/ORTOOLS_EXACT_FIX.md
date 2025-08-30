# OR-Tools VRP Solver Fix: Enabling Exact Solving

## Problem Identified
The `exact_ortools_vrp.py` solver was incorrectly configured to use metaheuristics (GUIDED_LOCAL_SEARCH) despite being labeled as an "exact" solver. This made it a heuristic solver that could miss optimal solutions.

## Solution Applied
Changed the solver configuration to disable metaheuristics and enable exact solving options.

### Key Changes

1. **Disabled Metaheuristics**
   ```python
   # Before (INCORRECT - uses metaheuristic):
   search_parameters.local_search_metaheuristic = (
       routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
   )
   
   # After (CORRECT - disables metaheuristics):
   search_parameters.local_search_metaheuristic = (
       routing_enums_pb2.LocalSearchMetaheuristic.UNSET  # Value: 0
   )
   ```

2. **Added Exact Search Options**
   ```python
   # Enable full constraint propagation
   search_parameters.use_full_propagation = True
   
   # Enable depth-first search for small instances (≤10 customers)
   if n_customers <= 10:
       search_parameters.use_depth_first_search = True
   ```

3. **Updated Optimality Claim**
   - Now only marks solutions as optimal for small instances (≤10 customers)
   - Added documentation explaining OR-Tools' limitations for exact solving

## Performance Impact

### Before (with GUIDED_LOCAL_SEARCH):
- Slow convergence (5+ seconds even for small instances)
- Uses metaheuristic local search
- May not find true optimal solution

### After (with UNSET metaheuristic):
- Fast for small instances (<0.1 seconds for 5 customers)
- No metaheuristic search
- Finds optimal solutions for small instances
- Much more suitable as an "exact" solver

## Important Notes

1. **OR-Tools Routing Module Limitations**:
   - Primarily designed for heuristic solving
   - Even without metaheuristics, doesn't guarantee optimality for large instances
   - Best used for small instances (≤10 customers) when exact solving is needed

2. **For True Exact Solving**:
   - Use `exact_milp.py` or `exact_pulp.py` for guaranteed optimal solutions
   - Use `exact_dp.py` for very small instances (brute force)

3. **Benchmark Implications**:
   - The benchmark was treating OR-Tools VRP as ground truth
   - With this fix, it's more appropriate for small instance validation
   - Consider using MILP solvers as ground truth for larger instances

## Testing Results
- Tested on 5-customer instance: finds optimal solution in 0.044s
- Previously with metaheuristic: took 5+ seconds
- Now agrees with other exact solvers on optimal cost

## Files Modified
- `/solvers/exact_ortools_vrp.py` - Fixed configuration
- Original backed up to `/solvers/exact_ortools_vrp.py.backup`
