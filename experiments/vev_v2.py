# Import libraries
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import multiprocessing as mp  # Import multiprocessing for parallel processing

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')  # Adjust this to point to the correct folder
sys.path.append(project_root)
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics
from model.VotingRules import VotingRules

# Initialize simulation parameters
num_voters = 144
num_projects = 600
total_op_tokens = 30e6
num_rounds = 100
voter_type = 'mallows_model'
quorum=17

# Initialize the model
model = VotingModel(voter_type=voter_type, num_voters=num_voters, num_projects=num_projects, total_op_tokens=total_op_tokens)

# Initialize the evaluation metrics
model.step()
eval_metrics = EvalMetrics(model)

# Define the directory for output
output_dir = os.path.join(current_dir, '..', 'data', 'vm_data')
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Define a function to process a single round of evaluation
def process_round(round_num):
    # In each round, the model and evaluation are run
    model.step()  # Advance the simulation by one step
    vev_results = eval_metrics.evaluate_vev(num_rounds=num_rounds)  # Evaluate for this round
    vev_results['round'] = round_num  # Track the round number
    #vev_results['project_max_vev'] = vev_results['project_max_vev'] / total_op_tokens  # Normalize the results
    return vev_results

# Function to run the simulation in parallel
def run_parallel_simulation(num_rounds, num_workers=4):
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Run the simulation for each round in parallel
        results = pool.map(process_round, range(1, num_rounds + 1))  # Parallel execution
    
    # Combine all results into a single DataFrame
    combined_results = pd.concat(results, ignore_index=True)
    return combined_results

# Main execution
if __name__ == '__main__':
    num_workers = mp.cpu_count()  # Use the number of available CPU cores for parallel processing
    print(f"Running {num_rounds} rounds in parallel using {num_workers} workers...")

    # Run the simulation in parallel and get combined results
    all_results = run_parallel_simulation(num_rounds, num_workers)

    # Save the combined results to a CSV file
    output_file = os.path.join(output_dir, f'vev_parallel_run_results_{timestamp}_{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}.csv')
    all_results.to_csv(output_file, index=False)
    
    print("Fund Allocations for each project")
    allocation_df = model.compile_fund_allocations()
    #allocation_df.to_csv(os.path.join(output_dir, 'allocation_df.csv'), index=False)
    print(allocation_df.head(10))

    print(f"Experiment completed and results saved to {output_file}.")
    print(all_results.head(100))
