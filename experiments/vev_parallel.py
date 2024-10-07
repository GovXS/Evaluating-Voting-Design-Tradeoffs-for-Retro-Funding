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
import experiments.experiments_config as experiments_config

# Initialize simulation parameters
num_voters = experiments_config.num_voters#40
num_projects = experiments_config.num_projects#145
total_op_tokens = experiments_config.total_op_tokens#8e6
num_rounds = experiments_config.num_rounds#5
voter_type = experiments_config.voter_type#'mallows_model'
quorum = experiments_config.quorum#17

 # Parameters for bribery evaluation
min_increase = experiments_config.min_increase#1
max_increase = experiments_config.max_increase#30
iterations = experiments_config.iterations#30
experiment_description=experiments_config.experiment_description#'running robustness with r4 data'

# Initialize the model
model = VotingModel(voter_type=voter_type, num_voters=num_voters, num_projects=num_projects, total_op_tokens=total_op_tokens)

# Initialize the evaluation metrics
model.step()
eval_metrics = EvalMetrics(model)

# Define the directory for output
output_dir = os.path.join(current_dir, '..', 'data', 'experiment_results', f'{experiment_description}')

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)


# Define a function to process a single round of evaluation
def process_round(round_num):
    # In each round, the model and evaluation are run
    model.step()  # Advance the simulation by one step
    vev_results = eval_metrics.evaluate_vev_optimized(1)  # Evaluate for this round
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
    output_file = os.path.join(output_dir, f'vev_results_{timestamp}.csv')
    all_results.to_csv(output_file, index=False)
    
    print("Fund Allocations for each project")
    allocation_df = model.compile_fund_allocations()
    allocation_df.to_csv(os.path.join(output_dir, 'allocation_df.csv'), index=False)
    print(allocation_df.head(10))

    print(f"Experiment completed and results saved to {output_file}.")
    print(all_results.head(100))

    # Save the experiment parameters to a text file
    parameters = {
        "experiment_description":experiment_description,
        "num_voters": num_voters,
        "num_projects": num_projects,
        "total_op_tokens": total_op_tokens,
        "num_rounds per iteration": num_rounds,
        "voter_type": voter_type,
        "quorum": quorum,
        "timestamp": timestamp
    }

    script_file_name = os.path.splitext(os.path.basename(__file__))[0]

    # Set the path for the parameter file, including the script file name
    param_file_path = os.path.join(output_dir, f'{script_file_name}_experiment_parameters_{timestamp}.txt')

    # Write the parameters to the text file
    with open(param_file_path, 'w') as f:
        for key, value in parameters.items():
            f.write(f'{key}: {value}\n')