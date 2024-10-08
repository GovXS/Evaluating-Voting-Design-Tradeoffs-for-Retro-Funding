# Import libraries
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')  # Adjust this to point to the correct folder
sys.path.append(project_root)
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics
import experiments_config

# Add the directory containing the VotingModel to the Python path

from model.VotingRules import VotingRules

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
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current file's directory
output_dir = os.path.join(current_dir, '..', 'data', 'experiment_results', f'{experiment_description}_{timestamp}')

    # Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

allocation_df=model.compile_fund_allocations()
allocation_df.to_csv(os.path.join(output_dir, 'allocation_df.csv'), index=False)
print("Fund Allocations for each project")
print(allocation_df.head(10))


vev_results = eval_metrics.evaluate_vev(num_rounds)
vev_results['project_max_vev']=vev_results['project_max_vev']/total_op_tokens
vev_results.to_csv(os.path.join(output_dir, f'vev_results_{timestamp}.csv'), index=False)

print(vev_results.head(100))
print("Experiment Completed")


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