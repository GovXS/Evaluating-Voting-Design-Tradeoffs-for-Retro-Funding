import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import multiprocessing as mp
from copy import deepcopy

# Add the directory containing the VotingModel to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')  # Adjust this to point to the correct folder
sys.path.append(project_root)
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics
from model.VotingRules import VotingRules

# Define a function to process a single round of evaluation
def process_round(model, desired_increase, round_num):
    model_copy = deepcopy(model)  # Independent copy of the model to avoid state sharing
    eval_metrics_copy = EvalMetrics(model_copy)  # Independent EvalMetrics instance
    
    # Simulate the next step (round)
    model_copy.step()
    
    # Evaluate control results for the current desired increase for this round
    control_results_df = eval_metrics_copy.evaluate_control(1, desired_increase)  # Evaluate only for one round
    
    # Add round number to the results for tracking
    control_results_df['round'] = round_num
    
    return control_results_df

# Function to run control evaluation across rounds in parallel for a specific desired increase
def run_parallel_control_evaluation(model, desired_increase, num_rounds, num_workers=4):
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Use pool.starmap to parallelize the process_round function
        results = pool.starmap(process_round, [(deepcopy(model), desired_increase, round_num) for round_num in range(1, num_rounds + 1)])

    # Combine all results into a single DataFrame
    combined_results = pd.concat(results, ignore_index=True)
    return combined_results

# Main execution
if __name__ == '__main__':
    # Initialize simulation parameters
    num_voters = 40
    num_projects = 145
    total_op_tokens = 8e6
    num_rounds = 5
    voter_type = 'mallows_model'
    quorum = 17
    
    # Initialize the model
    model = VotingModel(voter_type=voter_type, num_voters=num_voters, num_projects=num_projects, total_op_tokens=total_op_tokens)
    
    # Initialize the evaluation metrics
    model.step()
    eval_metrics = EvalMetrics(model)

    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    output_dir = os.path.join(current_dir, '..', 'data', 'vm_data')  # Define relative path

    # Parameters for control sweep
    min_increase = 1
    max_increase = 30
    iterations = 30  # Number of different desired increase percentages to try
    desired_increase_values = np.linspace(min_increase, max_increase, iterations)

    # Create a DataFrame to hold the results
    all_control_results = pd.DataFrame()

    # Iterate over each desired_increase_percentage
    for i, desired_increase in enumerate(desired_increase_values, 1):
        print(f"Evaluating control for desired_increase_percentage: {desired_increase} ({i}/{iterations})")
        
        # Run the parallel control evaluation for this desired increase across rounds
        num_workers = mp.cpu_count()  # Use the available CPU cores
        control_results = run_parallel_control_evaluation(model, desired_increase, num_rounds, num_workers)
        
        # Add the desired increase percentage to the results
        control_results['desired_increase_percentage'] = desired_increase
        
        # Append to the overall results
        all_control_results = pd.concat([all_control_results, control_results], ignore_index=True)

    # Display the results after the parallel execution
    print(all_control_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the results to a CSV file
    output_path = os.path.join(output_dir, f'control_experiment_results_{timestamp}_{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}.csv')
    all_control_results.to_csv(output_path, index=False)

    # Display the results
    print(all_control_results.head(100))
    print(f"Results saved to {output_path}")
