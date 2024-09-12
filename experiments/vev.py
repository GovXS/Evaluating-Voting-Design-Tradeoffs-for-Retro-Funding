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

# Add the directory containing the VotingModel to the Python path

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
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current file's directory
output_dir = os.path.join(current_dir, '..', 'data', 'vm_data')  # Define relative path

allocation_df=model.compile_fund_allocations()
allocation_df.to_csv(os.path.join(output_dir, 'allocation_df.csv'), index=False)
print("Fund Allocations for each project")
print(allocation_df.head(10))


vev_results = eval_metrics.evaluate_vev(num_rounds)
vev_results['project_max_vev']=vev_results['project_max_vev']/total_op_tokens
vev_results.to_csv(os.path.join(output_dir, f'vev_results_{timestamp}_{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}.csv'), index=False)

print(vev_results.head(100))
print("Experiment Completed")
