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
output_dir = os.path.join(current_dir, '..', 'data', 'vm_data')  # Define relative path

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
control_results.to_csv(os.path.join(output_dir, f'control_experiment_results_{timestamp}_{num_voters}_{num_projects}_{total_op_tokens}_{num_rounds}.csv'), index=False)


# %%
print(control_results.head(100))



