import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import multiprocessing as mp
from copy import deepcopy
import config

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')  # Adjust this to point to the correct folder
script_file_name = os.path.splitext(os.path.basename(__file__))[0]
sys.path.append(project_root)
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics
from model.VotingRules import VotingRules


# Define a function to process a single round of bribery evaluation
def process_bribery_round(model, desired_increase_percentage, round_num):
    model_copy = deepcopy(model)  # Independent copy of the model to avoid state sharing
    eval_metrics_copy = EvalMetrics(model_copy)  # Independent EvalMetrics instance
    
    # Simulate the bribery for the current round
    model_copy.step()  # Advance the simulation for this round
    bribery_results_df = eval_metrics_copy.evaluate_bribery_optimized(1, desired_increase_percentage)  # Evaluate for one round
    
    # Add round number for tracking
    bribery_results_df['round'] = round_num
    return bribery_results_df

# Function to run bribery evaluation across rounds in parallel
def run_parallel_bribery_evaluation(model, num_rounds, desired_increase_percentage, num_workers=4):
    with mp.Pool(processes=num_workers) as pool:
        # Parallelize the round execution using pool.starmap
        results = pool.starmap(process_bribery_round, [(deepcopy(model), desired_increase_percentage, round_num) for round_num in range(1, num_rounds + 1)])

    # Combine all results into a single DataFrame
    combined_results = pd.concat(results, ignore_index=True)
    return combined_results

# Main execution
if __name__ == '__main__':
    # Initialize simulation parameters
    
    num_voters = config.num_voters#40
    num_projects = config.num_projects#145
    total_op_tokens = config.total_op_tokens#8e6
    num_rounds = config.num_rounds#5
    voter_type = config.voter_type#'mallows_model'
    quorum = config.quorum#17
    experiment_description=config.experiment_description#'running robustness with r4 data'
    
    # Initialize the model
    model = VotingModel(voter_type=voter_type, num_voters=num_voters, num_projects=num_projects, total_op_tokens=total_op_tokens)
    
    # Initialize the evaluation metrics
    model.step()
    eval_metrics = EvalMetrics(model)

    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    output_dir = os.path.join(current_dir, '..', 'data', 'experiment_results', f'{experiment_description}')

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Parameters for bribery evaluation
    min_increase = config.min_increase#1
    max_increase = config.max_increase#30
    iterations = config.iterations#30
    desired_increase_percentages = np.linspace(min_increase, max_increase, iterations)

    # Iterate over each desired increase percentage
    bribery_results = pd.DataFrame()

    for i, desired_increase_percentage in enumerate(desired_increase_percentages, 1):
        print(f"Iteration {i}/{iterations} with desired_increase_percentage: {desired_increase_percentage}")

        # Run the parallel bribery evaluation for this desired increase percentage
        num_workers = mp.cpu_count()  # Use the available CPU cores
        bribery_results_for_percentage = run_parallel_bribery_evaluation(model, num_rounds, desired_increase_percentage, num_workers)

        # Calculate the average bribery cost for each voting rule over all rounds
        avg_bribery_costs = bribery_results_for_percentage.mean()

        # Convert the result to a DataFrame and add the desired_increase_percentage column
        avg_bribery_costs_df = avg_bribery_costs.to_frame().T
        avg_bribery_costs_df['desired_increase_percentage'] = desired_increase_percentage

        # Append the results to the final DataFrame
        bribery_results = pd.concat([bribery_results, avg_bribery_costs_df], ignore_index=True)

    # Display the results after the parallel execution
    print(bribery_results)

    # Save the results to a CSV file
    output_path = os.path.join(output_dir, f'bribery_experiment_results_{timestamp}.csv')
    bribery_results.to_csv(output_path, index=False)

    print("Bribery experiment completed")
    print(f"Results saved to {output_path}")

    # Save the experiment parameters to a text file
    parameters = {
        "experiment_description":experiment_description,
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

   

    # Set the path for the parameter file, including the script file name
    param_file_path = os.path.join(output_dir, f'{script_file_name}_experiment_parameters_{timestamp}.txt')

    # Write the parameters to the text file
    with open(param_file_path, 'w') as f:
        for key, value in parameters.items():
            f.write(f'{key}: {value}\n')
