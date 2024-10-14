# Cost of Bribery Experiment

### Key Components of the Experiment

1. **Cost of Bribery Experiment Setup**:
   - The experiment evaluates bribery costs over a range of **desired increase percentages**. The goal is to measure how many additional votes (tokens) are needed to increase the funding for a specific project by varying percentages. These percentages range from `1%` to `30%` in 30 iterations.
   
   ```python
   min_increase = 1
   max_increase = 30
   iterations = 30
   desired_increase_percentages = np.linspace(min_increase, max_increase, iterations)
   ```

2. **Run the Experiment Loop**:
   - The experiment iterates over each value of `desired_increase_percentage` and, for each value, it runs the `evaluate_bribery()` method of the `EvalMetrics` class to compute the **bribery cost**.
   - The bribery cost for each voting rule is evaluated across multiple rounds (`num_rounds` = 5). The average bribery cost across all rounds is calculated for each voting rule.

   ```python
   for i, desired_increase_percentage in enumerate(desired_increase_percentages, 1):
       print(f"Iteration {i}/{iterations} with desired_increase_percentage: {desired_increase_percentage}")
       
       # Evaluate bribery costs
       bribery_results_df = eval_metrics.evaluate_bribery(num_rounds, desired_increase_percentage)
       
       # Calculate average bribery costs
       avg_bribery_costs = bribery_results_df.mean()
       avg_bribery_costs_df = avg_bribery_costs.to_frame().T
       avg_bribery_costs_df['desired_increase_percentage'] = desired_increase_percentage
       
       # Append results
       bribery_results = pd.concat([bribery_results, avg_bribery_costs_df], ignore_index=True)
   ```

3. **Store the Results**:
   - After each iteration, the average bribery costs for the current desired increase percentage are appended to the `bribery_results` DataFrame. This DataFrame stores the results for all iterations.
   - At the end of the experiment, the results are saved to a CSV file for further analysis.
   
   ```python
   bribery_results.to_csv(output_path, index=False)
   ```

### Explanation of the Key Concepts:

- **Cost of Bribery**: 
   - The **bribery cost** is the number of additional votes (or tokens) required to increase the funding allocated to a specific project by a desired percentage. It is calculated by incrementally adding votes to the target project and recalculating the allocation using different voting rules.
   
- **Desired Increase Percentage**:
   - The experiment tests how the bribery cost changes as the desired increase in funding for a project grows. It starts with small increases (1%) and goes up to larger increases (30%).

- **Iterations**:
   - The experiment runs for 30 different **desired increase percentages**, ranging from 1% to 30%. Each iteration corresponds to evaluating how difficult it is to increase the funding for a project by the specified percentage.

- **Rounds**:
   - In each iteration, the bribery cost is evaluated over 5 rounds to average out any fluctuations in the results due to randomness in the voting model. The average bribery cost for each voting rule is recorded for analysis.

### Experiment Output:

- The experiment produces a **DataFrame** (`bribery_results`) that stores the average bribery costs for each voting rule across multiple desired increase percentages. The results are saved in a CSV file, which includes:
   - The **average bribery cost** for each voting rule across multiple rounds.
   - The **desired increase percentage** corresponding to each bribery cost.

### Sample Flow of Data:

1. **Initial Setup**:
   - The experiment sets up 40 voters, 145 projects, and a total of 8 million tokens.
   
2. **Bribery Evaluation**:
   - The bribery cost is evaluated for each voting rule at various desired increase percentages.
   
3. **Results**:
   - For each iteration (desired increase percentage), the bribery cost is averaged across all rounds and stored.
   - Example:
     ```
     desired_increase_percentage: 5%
     average_bribery_cost for voting_rule_1: 1000 tokens
     average_bribery_cost for voting_rule_2: 1200 tokens
     ```

### Summary:
- The experiment evaluates the **cost of bribery** for different voting rules by simulating multiple rounds and increasing the funding allocation for target projects by various percentages (1% to 30%).
- It records the average bribery cost required to achieve each percentage increase, helping to understand how vulnerable the system is to manipulation as the desired increase grows.
- The results are stored in a CSV file for further analysis, making it easy to examine how different voting rules react to bribery attempts at various levels.


# Cost of Control Experiment

This script is designed to evaluate the **resistance to control** in a voting system using the **control** metric from the `EvalMetrics` class. The experiment is structured to examine how difficult it is to manipulate a voting system by adding or removing voters in order to increase the funding for a specific project by a specified percentage.

### Key Components of the Experiment

1. **Experiment Parameters for Control Sweep**:
   - The experiment aims to evaluate how many voters need to be added or removed to achieve a desired percentage increase in funding for a project. The range of desired increases is specified between **1%** and **30%** in **30 iterations**.
   
   ```python
   min_increase = 1
   max_increase = 30
   iterations = 30
   desired_increase_values = np.linspace(min_increase, max_increase, iterations)
   ```

2. **Main Experiment Loop**:
   - For each **desired increase percentage** in the range from 1% to 30%, the `evaluate_control()` function is called to compute the **resistance to control**.
   - The evaluation is conducted across **several rounds** (as specified by `num_rounds`), and the average results from all rounds are calculated.

   ```python
   for i, desired_increase in enumerate(desired_increase_values, 1):
       print(f"Iteration {i}/{iterations} with desired_increase: {desired_increase}")
       
       # Evaluate control for the current desired increase
       control_results_constant_desired_increase_df = eval_metrics.evaluate_control(num_rounds, desired_increase)
       
       # Calculate the average results over all rounds
       avg_control_results = control_results_constant_desired_increase_df.mean()
       
       # Log the desired increase percentage
       avg_control_results['desired_increase'] = desired_increase
       
       # Convert the Series to DataFrame for appending
       avg_control_results_df = avg_control_results.to_frame().T
       
       # Append the results to the DataFrame
       control_results = pd.concat([control_results, avg_control_results_df], ignore_index=True)
   ```

3. **Key Functions Used:**
   - **`eval_metrics.evaluate_control(num_rounds, desired_increase)`**:
     - This function evaluates how many voters need to be added or removed to achieve the desired increase in funding for a target project.
     - It simulates the process across the specified number of rounds (5 in this case), where the control metric is evaluated for each voting rule and project.
   
   - **Average Control Results**:
     - After evaluating the control metric for each round, the average results are calculated to minimize random fluctuations.
     - The results include the minimum number of voters that need to be added or removed to achieve the desired increase for each voting rule.

4. **Save the Results**:
   - Once all iterations are complete, the results of the experiment are saved to a CSV file. The file name includes parameters like the number of voters, projects, tokens, and rounds, as well as a timestamp to ensure uniqueness.

   ```python
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   control_results.to_csv(os.path.join(output_dir, f'control_experiment_results_{num_projects}_{num_voters}_{total_op_tokens}_{num_rounds*iterations}_{timestamp}.csv'), index=False)
   ```

5. **Display Results**:
   - The experiment prints the top 100 results for quick inspection and debugging purposes.

   ```python
   print(control_results.head(100))
   ```

### Explanation of the Key Concepts:

- **Cost of Control**:
   - This metric measures how difficult it is to manipulate the voting system by adding or removing voters to achieve a desired increase in funding for a specific project. The resistance is quantified by the number of voters that need to be added or removed.
   
- **Desired Increase**:
   - The experiment iterates over various desired percentage increases in funding (from 1% to 30%). For each increase, the script evaluates how many voters need to be added or removed to achieve that increase.
   
- **Iterations**:
   - The experiment runs 30 iterations, each corresponding to a different desired increase percentage. Each iteration evaluates the control metric for different voting rules and projects.

- **Rounds**:
   - In each iteration, the experiment runs over multiple rounds to simulate the voting process multiple times and calculate the average control results. This helps reduce the impact of randomness in the voting model.

### Output of the Experiment:

- The experiment produces a **DataFrame** (`control_results`) that stores the average control metrics for each desired increase percentage. The results are saved in a CSV file, which includes:
   - The **average control cost** (number of voters added/removed) for each voting rule.
   - The **desired increase percentage** corresponding to each control cost.

### Sample Flow of Data:

1. **Initial Setup**:
   - 40 voters, 145 projects, and 8 million tokens are initialized in the voting model.
   
2. **Control Evaluation**:
   - The number of voters that need to be added or removed is evaluated for various desired increase percentages.
   
3. **Results**:
   - Example:
     ```
     desired_increase: 10%
     avg_min_removal_cost for voting_rule_1: 5 voters
     avg_min_addition_cost for voting_rule_1: 4 voters
     ```

### Summary:

- The experiment evaluates the **cost of control** for different voting rules by simulating the number of voters that need to be added or removed to achieve a desired increase in funding for a target project.
- It runs multiple iterations, each corresponding to a different **desired increase percentage** (from 1% to 30%).
- The experiment calculates and saves the **average control cost** (the number of voters added/removed) for each voting rule, making it possible to analyze how resistant each rule is to manipulation through voter addition or removal.
- The results are saved in a CSV file, providing insights into the vulnerability of the voting system to control attempts.



# VEV Experiment

This experiment evaluates the **Voter Extractable Value (VEV)** in a voting system using the `EvalMetrics` class. The **VEV** metric measures how much a single voter can influence the outcome by concentrating a large percentage of their voting power (e.g., 90-99%) on a single project. This experiment aims to quantify the maximum impact that voters can have on project allocations using different voting rules.

### Key Components of the Experiment

1. **Evaluate Voter Extractable Value (VEV)**:
   - The experiment runs the **VEV evaluation** for the specified number of rounds (50 rounds). In each round, the model generates a new voting profile, and for each voter and project, the **maximum L1 distance** is computed.
   - The **L1 distance** measures how much the outcome changes when a voter allocates 90-99% of their tokens to a single project (i.e., extreme voting behavior). The goal is to determine the maximum skewness a voter can cause to the fund allocation using different voting rules.

   ```python
   vev_results = eval_metrics.evaluate_vev(num_rounds)
   ```

### Key Steps in VEV Evaluation:
1. For each voter and project, the vote is modified to allocate 90-99% of the voter's tokens to the project.
2. The resulting allocation is compared with the original allocation using the **L1 distance** to quantify the impact of the voter’s extreme voting behavior.
3. The process is repeated for multiple voting rules to evaluate how resistant each rule is to such extreme behavior.

4. **Normalize the Maximum VEV**:
   - The experiment normalizes the maximum VEV values by dividing them by the total number of tokens (`total_op_tokens`). This normalization ensures that the results are expressed as a fraction of the total voting power, making it easier to compare results across different settings or simulations with varying total tokens.
   
   ```python
   vev_results['project_max_vev'] = vev_results['project_max_vev'] / total_op_tokens
   ```

5. **Save VEV Results**:
   - The VEV results, which include metrics such as the maximum VEV for each voting rule and the corresponding project, are saved in a CSV file.
   - The filename includes parameters like the number of voters, projects, tokens, and rounds, along with a timestamp to ensure unique file names for different experiments.

   ```python
   vev_results.to_csv(os.path.join(output_dir, f'vev_results_{num_projects}_{num_voters}_{total_op_tokens}_{num_rounds}_{timestamp}.csv'), index=False)
   ```

6. **Display Results**:
   - The experiment prints the first 100 rows of the results for quick inspection and debugging.
   
   ```python
   print(vev_results.head(100))
   ```

7. **Completion Message**:
   - A message is printed to indicate that the experiment has been completed.

   ```python
   print("Experiment Completed")
   ```

### Explanation of the Key Concepts:

- **Voter Extractable Value (VEV)**:
   - VEV measures the maximum impact a single voter can have on the allocation outcome by concentrating their voting power on a single project. It is computed as the maximum **L1 distance** between the original allocation and the allocation after the voter modifies their vote to heavily favor one project.
   - This metric helps assess the vulnerability of the voting system to extreme behavior, where a voter tries to "extract" as much value as possible for their preferred project.

- **L1 Distance**:
   - The **L1 distance** quantifies the difference between two allocation vectors by summing the absolute differences in the allocations across all projects. In this context, it measures the change in the outcome caused by the voter's extreme behavior.

- **Rounds**:
   - The experiment runs for 50 rounds. In each round, a new voting profile is generated, and the VEV is calculated for all voters and projects. The results are averaged across these rounds to reduce the impact of randomness in the voting model.

### Output of the Experiment:

- The experiment produces a **DataFrame** (`vev_results`) that stores the VEV for each round and voting rule. The results include:
   - The **maximum VEV** for each voting rule, which represents the largest skewness any voter was able to cause during the simulation.
   - The **project** that was most affected by the voter's behavior, along with the new allocation after the vote modification.

### Sample Flow of Data:

1. **Initial Setup**:
   - The voting model is initialized with 40 voters, 145 projects, and 8 million tokens.
   
2. **VEV Evaluation**:
   - The maximum VEV is calculated for each voting rule by simulating extreme voting behavior, where a voter allocates 90-99% of their tokens to a single project.
   
3. **Results**:
   - Example:
     ```
     round: 1
     voting_rule: 'quadratic'
     project_max_vev: 0.015 (normalized as a fraction of total tokens)
     project_max_original_allocation: 1000 tokens
     project_max_new_allocation: 1250 tokens
     ```

### Summary:

- The experiment evaluates the **Voter Extractable Value (VEV)**, which measures the maximum impact a single voter can have on the outcome by heavily favoring a single project.
- The experiment runs for 50 rounds, and for each round, it calculates the **maximum VEV** for various voting rules.
- The results are normalized and saved in a CSV file, providing insights into how vulnerable the voting system is to extreme behavior and which projects are most affected by such behavior.
- The output shows how much skewness a voter can introduce into the system by concentrating their voting power on a particular project, helping identify potential weaknesses in the voting rules.
