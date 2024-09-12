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

# Initialize simulation parameters
num_voters = 40
num_projects = 145
total_op_tokens = 8e6
voter_type = 'mallows_model'
quorum = 17

# Define the directory for output
output_dir = os.path.join(current_dir, '..', 'data', 'vm_data')
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define a function to process a single round of evaluation
def process_round(round_num, num_rounds, model_params):
    # Initialize a new model for each process to ensure independence
    model = VotingModel(**model_params)
    model.step()

    # Initialize a new EvalMetrics instance for each process
    eval_metrics = EvalMetrics(model)
    
    # Perform the VEV evaluation
    vev_results = eval_metrics.evaluate_vev(num_rounds=num_rounds)
    vev_results['round'] = round_num  # Track the round number

    return vev_results

# Function to run the simulation in parallel
def run_parallel_simulation(num_rounds, num_workers=4, model_params=None):
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Run the simulation for each round in parallel
        results = pool.starmap(process_round, [(round_num, num_rounds, model_params) for round_num in range(1, num_rounds + 1)])

    # Combine all results into a single DataFrame
    combined_results = pd.concat(results, ignore_index=True)
    return combined_results

# Main execution
if __name__ == '__main__':
    num_workers = mp.cpu_count()  # Use the number of available CPU cores for parallel processing
    num_rounds = 30  # Define the number of rounds

    # Prepare model parameters to pass them to each worker
    model_params = {
        'voter_type': voter_type,
        'num_voters': num_voters,
        'num_projects': num_projects,
        'total_op_tokens': total_op_tokens
    }

    print(f"Running {num_rounds} rounds in parallel using {num_workers} workers...")

    # Run the simulation in parallel and get combined results
    all_results = run_parallel_simulation(num_rounds, num_workers, model_params)
    # Save the combined results to a CSV file
    output_file = os.path.join(output_dir, f'vev_parallel_run_results_{timestamp}_{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}.csv')
    all_results.to_csv(output_file, index=False)
    
    print("Fund Allocations for each project")
    model = VotingModel(**model_params)
    model.step()
    allocation_df = model.compile_fund_allocations()
    print(allocation_df.head(10))

    print(f"Experiment completed and results saved to {output_file}.")
    print(all_results.head(100))
