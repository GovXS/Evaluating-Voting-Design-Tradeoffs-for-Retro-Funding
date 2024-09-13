# Import libraries
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')  # Adjust this to point to the correct folder
sys.path.append(project_root)
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics

# Add the directory containing the VotingModel to the Python path

from model.VotingRules import VotingRules
from datetime import datetime
import multiprocessing as mp
from copy import deepcopy

# Define a function to process a single iteration
def process_iteration(model, desired_increase, num_rounds):
    model_copy = deepcopy(model)  # Independent copy of the model to avoid state sharing
    eval_metrics_copy = EvalMetrics(model_copy)  # Independent EvalMetrics instance
    control_results_constant_desired_increase_df = eval_metrics_copy.evaluate_control(num_rounds, desired_increase)

    # Calculate the average control results over all rounds
    avg_control_results = control_results_constant_desired_increase_df.mean()

    # Log both the percentage and absolute amount of the desired increase
    avg_control_results['desired_increase'] = desired_increase

    # Convert the Series to a DataFrame for concatenation
    avg_control_results_df = avg_control_results.to_frame().T

    return avg_control_results_df

# Function to run control evaluation in parallel
def run_parallel_control_evaluation(model, desired_increase_values, num_rounds, num_workers=4):
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Use pool.starmap to parallelize the process_iteration function
        results = pool.starmap(process_iteration, [(deepcopy(model), desired_increase, num_rounds) for desired_increase in desired_increase_values])

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
    iterations = 30
    desired_increase_values = np.linspace(min_increase, max_increase, iterations)

    # Run the parallel control evaluation
    num_workers = mp.cpu_count()  # Use the available CPU cores
    control_results = run_parallel_control_evaluation(model, desired_increase_values, num_rounds, num_workers)

    # Display the results after the parallel execution
    print(control_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the results to a CSV file
    control_results.to_csv(os.path.join(output_dir, f'control_experiment_results_{timestamp}_{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}.csv'), index=False)

    # Display the results
    print(control_results.head(100))