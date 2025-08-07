#!/usr/bin/env python3

def format_comparison_table(results, naive_cost_per_customer):
    """
    Create a nicely formatted comparison table with proper alignment and 3-digit precision.
    """
    # Sample data structure matching what we have
    sample_results = {
        'Pointer+RL': {
            'parameters': 21057,
            'training_time': 16.9,
            'final_val_cost': 2.586,
            'final_train_cost': 2.671
        },
        'GT-Greedy': {
            'parameters': 92161,
            'training_time': 12.6,
            'final_val_cost': 2.599,
            'final_train_cost': 2.741
        },
        'GT+RL': {
            'parameters': 92161,
            'training_time': 14.6,
            'final_val_cost': 2.601,
            'final_train_cost': 2.720
        },
        'DGT+RL': {
            'parameters': 92353,
            'training_time': 17.1,
            'final_val_cost': 2.600,
            'final_train_cost': 2.690
        },
        'GAT+RL': {
            'parameters': 58785,
            'training_time': 14.3,
            'final_val_cost': 2.600,
            'final_train_cost': 2.710
        }
    }
    
    # Use either provided results or sample data
    data = results if results else sample_results
    naive_baseline = naive_cost_per_customer if naive_cost_per_customer else 0.2225
    
    # Calculate column widths with corrections
    model_width = max(len("Model"), max(len(model) for model in data.keys()))
    params_width = max(len("Parameters"), max(len(f"{data[model]['parameters']:,}") for model in data.keys()))
    time_width = max(len("Time (s)") - 1, max(len(f"{data[model]['training_time']:.1f}s") for model in data.keys()))  # Remove 1 space
    train_cost_width = max(len("Train Cost"), max(len(f"{data[model]['final_train_cost']:.3f}") for model in data.keys()))
    val_cost_width = max(len("Val Cost"), max(len(f"{data[model]['final_val_cost']:.3f}") for model in data.keys()))
    val_per_cust_width = max(len("Val/Cust"), max(len(f"{data[model]['final_val_cost']/20:.3f}") for model in data.keys()))
    improvement_width = max(len("Improv %") - 1, max(len(f"{(1 - (data[model]['final_val_cost']/20)/naive_baseline)*100:.2f}%") for model in data.keys()))  # Remove 1 space
    
    # Create table - add 1 dash to time and improvement columns for proper alignment
    header = f"| {'Model':<{model_width}} | {'Parameters':>{params_width}} | {'Time (s)':>{time_width}} | {'Train Cost':>{train_cost_width}} | {'Val Cost':>{val_cost_width}} | {'Val/Cust':>{val_per_cust_width}} | {'Improv %':>{improvement_width}} |"
    separator = f"|{'-'*(model_width+2)}|{'-'*(params_width+2)}|{'-'*(time_width+3)}|{'-'*(train_cost_width+2)}|{'-'*(val_cost_width+2)}|{'-'*(val_per_cust_width+2)}|{'-'*(improvement_width+3)}|"
    
    print(header)
    print(separator)
    
    for model_name, metrics in data.items():
        val_per_customer = metrics['final_val_cost'] / 20
        improvement = (1 - val_per_customer / naive_baseline) * 100
        
        row = f"| {model_name:<{model_width}} | {metrics['parameters']:>{params_width},} | {metrics['training_time']:>{time_width}.1f}s | {metrics['final_train_cost']:>{train_cost_width}.3f} | {metrics['final_val_cost']:>{val_cost_width}.3f} | {val_per_customer:>{val_per_cust_width}.3f} | {improvement:>{improvement_width}.2f}% |"
        print(row)
    
    return header, separator

def create_table_formatting_function():
    """
    Return the formatted table creation code for integration into main script.
    """
    
    function_code = '''
def create_formatted_results_table(results_dict, naive_cost_per_customer):
    """
    Create a nicely formatted comparison table with proper alignment and 3-digit precision.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        naive_cost_per_customer: Naive baseline cost per customer for improvement calculation
    
    Returns:
        str: Formatted table string
    """
    if not results_dict:
        return "No results to display"
    
    # Calculate column widths dynamically
    model_width = max(len("Model"), max(len(model) for model in results_dict.keys()))
    params_width = max(len("Parameters"), max(len(f"{results_dict[model]['parameters']:,}") for model in results_dict.keys()))
    time_width = max(len("Time (s)"), max(len(f"{results_dict[model]['training_time']:.1f}s") for model in results_dict.keys()))
    train_cost_width = max(len("Train Cost"), max(len(f"{results_dict[model]['final_train_cost']:.3f}") for model in results_dict.keys()))
    val_cost_width = max(len("Val Cost"), max(len(f"{results_dict[model]['final_val_cost']:.3f}") for model in results_dict.keys()))
    val_per_cust_width = max(len("Val/Cust"), max(len(f"{results_dict[model]['final_val_cost']/20:.3f}") for model in results_dict.keys()))
    improvement_width = max(len("Improv %"), max(len(f"{(1 - (results_dict[model]['final_val_cost']/20)/naive_cost_per_customer)*100:.2f}%") for model in results_dict.keys()))
    
    # Create table components
    header = f"| {'Model':<{model_width}} | {'Parameters':>{params_width}} | {'Time (s)':>{time_width}} | {'Train Cost':>{train_cost_width}} | {'Val Cost':>{val_cost_width}} | {'Val/Cust':>{val_per_cust_width}} | {'Improv %':>{improvement_width}} |"
    separator = f"|{'-'*(model_width+2)}|{'-'*(params_width+2)}|{'-'*(time_width+2)}|{'-'*(train_cost_width+2)}|{'-'*(val_cost_width+2)}|{'-'*(val_per_cust_width+2)}|{'-'*(improvement_width+2)}|"
    
    # Build table rows
    table_lines = [header, separator]
    
    for model_name, metrics in results_dict.items():
        val_per_customer = metrics['final_val_cost'] / 20
        improvement = (1 - val_per_customer / naive_cost_per_customer) * 100
        
        row = f"| {model_name:<{model_width}} | {metrics['parameters']:>{params_width},} | {metrics['training_time']:>{time_width}.1f}s | {metrics['final_train_cost']:>{train_cost_width}.3f} | {metrics['final_val_cost']:>{val_cost_width}.3f} | {val_per_customer:>{val_per_cust_width}.3f} | {improvement:>{improvement_width}.2f}% |"
        table_lines.append(row)
    
    return "\\n".join(table_lines)
'''
    
    return function_code

if __name__ == "__main__":
    print("🧪 Testing table formatting with sample data...")
    print("=" * 120)
    
    # Test with sample data
    format_comparison_table(None, None)
    
    print("\n" + "=" * 120)
    print("✅ Table formatting test completed!")
    print("\nFunction code for integration:")
    print("-" * 60)
    print(create_table_formatting_function())
