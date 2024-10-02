import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import multiprocessing as mp
from copy import deepcopy
import config

# Add the directory containing the VotingModel to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')  # Adjust this to point to the correct folder
sys.path.append(project_root)
from model.VotingModel import VotingModel
from model.EvalMetrics import EvalMetrics
from model.VotingRules import VotingRules

# Define a function to process a single round of evaluation
def process_round(model, desired_increase, round_num):
    model_copy = deepcopy(model)  # Independent copy of the model to avoid state sharing
    eval_metrics_copy = EvalMetrics(model_copy)  # Independent EvalMetrics instance
    
    # Simulate the next step (round)
    model_copy.step()
    
    # Evaluate control results for the current desired increase for this round
    control_results_df = eval_metrics_copy.evaluate_control_optimized(num_rounds=1, desired_increase=desired_increase)  # Evaluate only for one round
    
    # Add round number to the results for tracking
    control_results_df['round'] = round_num
    
    return control_results_df

# Function to run control evaluation across rounds in parallel for a specific desired increase
def run_parallel_control_evaluation(model, desired_increase, num_rounds, num_workers=4):
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Use pool.starmap to parallelize the process_round function
        results = pool.starmap(process_round, [(deepcopy(model), desired_increase, round_num) for round_num in range(1, num_rounds + 1)])

    # Combine all results into a single DataFrame
    combined_results = pd.concat(results, ignore_index=True)
    return combined_results

# Main execution
if __name__ == '__main__':
    # Initialize simulation parameters
  

    # Initialize simulation parameters
    num_voters = config.num_voters#40
    num_projects = config.num_projects#145
    total_op_tokens = config.total_op_tokens#8e6
    num_rounds = config.num_rounds#5
    voter_type = config.voter_type#'mallows_model'
    quorum = config.quorum#17

    # Parameters for bribery evaluation
    min_increase = config.min_increase#1
    max_increase = config.max_increase#30
    iterations = config.iterations#30
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

     # Number of different desired increase percentages to try
    desired_increase_values = np.linspace(min_increase, max_increase, iterations)

    # Create a DataFrame to hold the results
    all_control_results = pd.DataFrame()

    # Iterate over each desired_increase_percentage
    # Iterate over each desired_increase_percentage
    for i, desired_increase in enumerate(desired_increase_values, 1):
        print(f"Evaluating control for desired_increase_percentage: {desired_increase} ({i}/{iterations})")
        
        # Run the parallel control evaluation for this desired increase across rounds
        num_workers = mp.cpu_count()  # Use the available CPU cores
        control_results = run_parallel_control_evaluation(model, desired_increase, num_rounds, num_workers)
        
        # Calculate the average control results for this percentage
        avg_control_results = control_results.mean()
        
        # Convert the average results to a DataFrame and add the desired_increase_percentage
        avg_control_results_df = avg_control_results.to_frame().T
        avg_control_results_df['desired_increase_percentage'] = desired_increase
        
        # Append the average results for this desired increase to the final results DataFrame
        all_control_results = pd.concat([all_control_results, avg_control_results_df], ignore_index=True)

        

    # Display the results after the parallel execution
    print(all_control_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the results to a CSV file
    output_path = os.path.join(output_dir, f'control_experiment_results_{timestamp}.csv')
    all_control_results.to_csv(output_path, index=False)

    # Display the results
    print(all_control_results.head(100))
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

    script_file_name = os.path.splitext(os.path.basename(__file__))[0]

    # Set the path for the parameter file, including the script file name
    param_file_path = os.path.join(output_dir, f'{script_file_name}_experiment_parameters_{timestamp}.txt')

    # Write the parameters to the text file
    with open(param_file_path, 'w') as f:
        for key, value in parameters.items():
            f.write(f'{key}: {value}\n')
