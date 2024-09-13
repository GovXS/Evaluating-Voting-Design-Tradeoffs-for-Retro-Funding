import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import multiprocessing as mp
from copy import deepcopy

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')  # Adjust this to point to the correct folder
sys.path.append(project_root)
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics
from model.VotingRules import VotingRules

# Define a function to process a single iteration
def process_bribery_iteration(model, desired_increase_percentage, num_rounds):
    model_copy = deepcopy(model)  # Independent copy of the model to avoid state sharing
    eval_metrics_copy = EvalMetrics(model_copy)  # Independent EvalMetrics instance
    bribery_results_df = eval_metrics_copy.evaluate_bribery_optimized(num_rounds, desired_increase_percentage)

    # Calculate the average bribery cost for each voting rule over all rounds
    avg_bribery_costs = bribery_results_df.mean()

    # Convert the result to a DataFrame and add the desired_increase_percentage column
    avg_bribery_costs_df = avg_bribery_costs.to_frame().T
    avg_bribery_costs_df['desired_increase_percentage'] = desired_increase_percentage

    return avg_bribery_costs_df

# Function to run bribery evaluation in parallel
def run_parallel_bribery_evaluation(model, desired_increase_percentages, num_rounds, num_workers=4):
    with mp.Pool(processes=num_workers) as pool:
        # Use pool.starmap to parallelize the process_bribery_iteration function
        results = pool.starmap(process_bribery_iteration, [(deepcopy(model), desired_increase_percentage, num_rounds) for desired_increase_percentage in desired_increase_percentages])

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

    # Parameters for bribery sweep
    min_increase = 1
    max_increase = 30
    iterations = 30
    desired_increase_percentages = np.linspace(min_increase, max_increase, iterations)

    # Run the parallel bribery evaluation
    num_workers = mp.cpu_count()  # Use the available CPU cores
    bribery_results = run_parallel_bribery_evaluation(model, desired_increase_percentages, num_rounds, num_workers)

    # Display the results after the parallel execution
    print(bribery_results)

    # Save the results to a CSV file
    output_path = os.path.join(output_dir, f'bribery_experiment_results_{timestamp}_{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}.csv')
    bribery_results.to_csv(output_path, index=False)

    print("Bribery experiment completed")
    print(f"Results saved to {output_path}")
