# CVRP Solver Improvements Summary

## Issues Identified and Fixed

### 1. exact_milp.py Was Not Truly Exact
**Problem:**
- Used CP-SAT as a fallback solver (constraint programming, not MILP)
- Accepted FEASIBLE solutions as OPTIMAL when time limit was reached
- Caused rare but significant validation errors (3-5% cost differences)

**Solution:**
- Replaced exact_milp with exact_dp in benchmarks
- exact_dp uses brute force enumeration - guarantees true optimality
- Limited to N≤10 (computationally feasible range for brute force)

### 2. Route Format Inconsistency
**Problem:**
- calculate_route_cost() expects routes WITHOUT depot nodes
- Some solvers were passing routes WITH depot nodes included
- Caused incorrect cost calculations

**Solution:**
- Fixed exact_dp to return clean routes (depot nodes removed)
- Ensured consistent route format across all solvers

### 3. Duplicate Node Coordinates
**Problem:**
- Generator sometimes created instances with nodes at identical positions
- Example: nodes 1 and 5 both at position (0.30, 0.16)
- Caused solver issues and validation errors

**Solution:**
- Added duplicate coordinate detection after instance generation
- Automatically regenerates instances with different seed if duplicates found
- Prevents degenerate problem instances

## Current Solver Configuration

### benchmark_exact_cpu.py now uses:
1. **exact_dp** - Brute force dynamic programming (N≤10 only)
   - Truly exact, guaranteed optimal
   - Automatically skipped for N>10

2. **exact_ortools_vrp** - OR-Tools routing solver
   - Good solutions but not guaranteed optimal for large instances
   - Works for all N

3. **heuristic_or** - Heuristic solver
   - Fast approximate solutions
   - Works for all N

## Files Modified
- `benchmark_exact_cpu.py` - Updated to use exact_dp, added duplicate check
- `solvers/exact_dp.py` - Fixed route format issue
- `plot_cpu_benchmark.py` - Updated to reference exact_dp

## Testing Results
- All changes tested successfully
- No validation errors on test runs
- Duplicate detection working correctly
- Benchmarks running smoothly with new configuration
