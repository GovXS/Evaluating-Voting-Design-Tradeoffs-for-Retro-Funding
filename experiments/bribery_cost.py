# %%
# Import libraries
import numpy as np
import pandas as pd
import os
import sys
#sys.path.append(os.path.abspath('/Users/idrees/Code/govxs/'))
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics

# Add the directory containing the VotingModel to the Python path

from model.VotingRules import VotingRules

# Initialize simulation parameters
num_voters = 140
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
output_dir = '/Users/idrees/Code/govxs/data/simulation_data'
allocation_df=model.compile_fund_allocations()
allocation_df.to_csv(os.path.join(output_dir, 'allocation_df.csv'), index=False)
allocation_df

# %%
min_increase = 0.01
max_increase = 10
iterations = 10
counter=1

# Create a DataFrame to store the results of each iteration
bribery_results = pd.DataFrame()

# Generate 100 values of desired_increase_percentage from 0.01 to 10
desired_increase_percentages = np.linspace(min_increase, max_increase, iterations)

# Iterate through each desired_increase_percentage
for i, desired_increase_percentage in enumerate(desired_increase_percentages, 1):
    print(f"Iteration {i}/{iterations} with desired_increase_percentage: {desired_increase_percentage}")

    # Evaluate bribery costs for the current desired increase percentage
    bribery_results_df = eval_metrics.evaluate_bribery_optimized(num_rounds, desired_increase_percentage)

    # Calculate the average bribery cost for each voting rule over all rounds
    avg_bribery_costs = bribery_results_df.mean()

    # Convert the result to a DataFrame and add the desired_increase_percentage column
    avg_bribery_costs_df = avg_bribery_costs.to_frame().T
    avg_bribery_costs_df['desired_increase_percentage'] = desired_increase_percentage

    # Append the results to the DataFrame using pd.concat
    bribery_results = pd.concat([bribery_results, avg_bribery_costs_df], ignore_index=True)

# Display the results after the loop
print(bribery_results)
output_path=os.path.join(output_dir, 'bribery_experiment_results_{counter}.csv')

bribery_results.to_csv(output_path, index=False)


# %%
bribery_results.head(100)
counter=counter+1
print("Bribery experiment Completed")
print("Results saved to {}")


