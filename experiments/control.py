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

# Initialize simulation parameters
num_voters = 40
num_projects = 145
total_op_tokens = 8e6
num_rounds = 5
voter_type = 'mallows_model'
quorum=17
# Initialize the model
model = VotingModel(voter_type=voter_type, num_voters=num_voters, num_projects=num_projects, total_op_tokens=total_op_tokens)

# Initialize the evaluation metrics
model.step()
eval_metrics = EvalMetrics(model)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current file's directory
output_dir = os.path.join(current_dir, '..', 'data', 'experiment_results', f'{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}')

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

allocation_df=model.compile_fund_allocations()
allocation_df.to_csv(os.path.join(output_dir, 'allocation_df.csv'), index=False)
print("Fund Allocations for each project")
print(allocation_df.head(10))



# %%

# Parameters for control sweep
min_increase = 1
max_increase = 30
iterations = 30

# Create a DataFrame to store the results of each iteration
control_results = pd.DataFrame()

# Generate 30 values of desired_increase from 1 to 30
desired_increase_values = np.linspace(min_increase, max_increase, iterations)

# Iterate through each desired_increase value
for i, desired_increase in enumerate(desired_increase_values, 1):
    print(f"Iteration {i}/{iterations} with desired_increase: {desired_increase}")

    # Evaluate control results for the current desired increase
    control_results_constant_desired_increase_df = eval_metrics.evaluate_control(num_rounds, desired_increase)

    # Calculate the average control results over all rounds
    avg_control_results = control_results_constant_desired_increase_df.mean()

    # Log both the percentage and absolute amount of the desired increase
    avg_control_results['desired_increase'] = desired_increase

    # Convert the Series to a DataFrame for concatenation
    avg_control_results_df = avg_control_results.to_frame().T

    # Append the results to the DataFrame using pd.concat
    control_results = pd.concat([control_results, avg_control_results_df], ignore_index=True)

# Display the results after the loop
print(control_results)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the results to a CSV file
control_results.to_csv(os.path.join(output_dir, f'control_experiment_results_{num_projects}_{num_voters}_{total_op_tokens}_{num_rounds*iterations}_{timestamp}.csv'), index=False)


# %%
print(control_results.head(100))


# Save the experiment parameters to a text file
parameters = {
    "num_voters": num_voters,
    "num_projects": num_projects,
    "total_op_tokens": total_op_tokens,
    "num_rounds per iteration": num_rounds,
    "voter_type": voter_type,
    "quorum": quorum,
    "min_increase": min_increase,
    "max_increase": max_increase,
    "iterations": iterations,
    "timestamp": timestamp
}

script_file_name = os.path.splitext(os.path.basename(__file__))[0]

# Set the path for the parameter file, including the script file name
param_file_path = os.path.join(output_dir, f'{script_file_name}_experiment_parameters_{timestamp}.txt')

# Write the parameters to the text file
with open(param_file_path, 'w') as f:
    for key, value in parameters.items():
        f.write(f'{key}: {value}\n')
