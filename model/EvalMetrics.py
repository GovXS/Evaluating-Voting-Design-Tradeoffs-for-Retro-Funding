import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
  
from joblib import Parallel, delayed

class EvalMetrics:
    def __init__(self, model):
        self.model = model

    def simulate_bribery_generic(self, voting_rule, target_project, desired_increase):
        """
        Generalized bribery simulation for any voting rule.
        
        Parameters:
        - voting_rule: The voting rule method (e.g., 'r1_quadratic', 'r2_mean', etc.)
        - target_project: The project that we are targeting for bribery.
        - desired_increase: The percentage increase we want in the target project's allocation.
        - max_iterations: Maximum number of iterations to prevent infinite loops.
        - tolerance: Minimal acceptable difference between new_funds and target_funds.

        Returns:
        - bribery_cost: The total additional votes required (bribery cost).
        """
        num_voters, num_projects = self.model.voting_matrix.shape
        original_allocation = self.model.allocate_funds(voting_rule)
        original_funds = original_allocation[target_project]
        
        tolerance=0.0001*self.model.total_op_tokens
        # Step 2: Calculate the target funds with the desired increase
        target_funds = original_funds + desired_increase
        
        # Step 3: Initialize the bribery cost and make a copy of the voting matrix
        bribery_cost = 0
        new_voting_matrix = self.model.voting_matrix.copy()
        original_matrix = self.model.voting_matrix.copy()  # Save the original matrix
        max_no_progress_iterations=5
        no_progress_iterations = 0
        prev_funds = original_funds
        #print(f"--- Bribery Simulation Debugging ---")
        #print(f"Original Funds: {original_funds}")
        #print(f"Desired Increase: {desired_increase}")
        #print(f"Target Funds: {target_funds}")  

        try:
            
            while True:
                # Temporarily assign the updated matrix to self.model.voting_matrix
                self.model.voting_matrix = new_voting_matrix
                
                # Recalculate the allocation with the updated voting matrix
                new_allocation = self.model.allocate_funds(voting_rule)
                new_funds = new_allocation[target_project]
                #print(f'{voting_rule}, project {target_project} new_funds: {new_funds}, target_funds: {target_funds}')
                #print(f"New Funds for project {target_project}: {new_funds}")

                # Check if the new allocation for the target project meets or exceeds the target
                if new_funds >= target_funds or abs(new_funds - target_funds) < tolerance:
                    print(f"Target met. Final Funds: {new_funds}, Bribery Cost: {bribery_cost}")
                    break

                
                if abs(new_funds - prev_funds) < tolerance:
                    no_progress_iterations += 1
                    if no_progress_iterations >= max_no_progress_iterations:
                        # No progress after multiple iterations, set bribery cost to a large value
                        bribery_cost = self.model.total_op_tokens
                        print(f"For project {target_project} and voting rule {voting_rule}, the bribery cost is infinite")
                        break
                else:
                    no_progress_iterations = 0  # Reset if there was progress
                
                # Add a small, fixed amount of additional votes (say 1% of the original votes)
                additional_votes = 0.01 * np.sum(original_matrix[:, target_project])
                new_voting_matrix[:, target_project] += additional_votes

                #print(f"Additional Votes: {additional_votes}, Bribery Cost So Far: {bribery_cost}")

                prev_funds = new_funds
                
                # Accumulate the additional votes into the bribery cost
                bribery_cost += additional_votes

        finally:
            # Step 5: Restore the original voting matrix
            self.model.voting_matrix = original_matrix

        return bribery_cost

    
    def evaluate_bribery(self, num_rounds=10, desired_increase_percentage=10):
        """
        Evaluate the bribery costs for increasing the allocation to a project by a desired percentage.
        """
        results = {'round': list(range(1, num_rounds + 1)), 'desired_increase': []}
        
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_bribery_cost'] = []

        # Track the overall progress for the number of rounds
        with tqdm(total=num_rounds, desc="Bribery Evaluation Progress", unit="round") as round_progress_bar:
            for i in range(num_rounds):
                print(f"\n--- Round {i + 1}/{num_rounds} ---")
                self.model.step()  # Simulate the next round
                desired_increase_percentage_current_round = desired_increase_percentage
                results['desired_increase'].append(desired_increase_percentage_current_round)

                # Progress bar for voting rules within a round
                for voting_rule in tqdm(self.model.voting_rules.keys(), desc=f"Round {i + 1}: Processing Voting Rules", leave=False, unit="rule"):
                    original_allocation = self.model.allocate_funds(voting_rule)
                    min_bribery_cost = float('inf')

                    # Track progress for projects
                    for project in range(self.model.num_projects):
                        start_time = time.time()
                        
                        original_funds = original_allocation[project]
                        desired_increase = original_funds * desired_increase_percentage_current_round

                        # Simulate bribery for the current project
                        bribery_cost = self.simulate_bribery_generic(voting_rule, project, desired_increase)
                        elapsed_time = time.time() - start_time

                        # Log progress for each project
                        print(f"[Round {i + 1}] [Project {project + 1}/{self.model.num_projects}] "
                            f"Voting Rule: {voting_rule}, Desired Increase: {desired_increase:.4f}, "
                            f"Bribery Cost: {bribery_cost:.4f}, Elapsed Time: {elapsed_time:.2f}s")

                        # Update the minimum bribery cost for the current voting rule
                        if bribery_cost < min_bribery_cost:
                            min_bribery_cost = bribery_cost

                    # Append the bribery cost for the current voting rule
                    results[f'{voting_rule}_bribery_cost'].append(min_bribery_cost)

                # Update the round progress bar after completing all voting rules
                round_progress_bar.update(1)

        # Convert results to a DataFrame
        final_results = pd.DataFrame(results)
        print("\nAll rounds completed. Final results:\n", final_results)
        return final_results
    

        # Gini Index
    def calculate_gini_index(self, allocation):
        m = len(allocation)
        if m == 0:
            return 0
        allocation_sorted = np.sort(allocation)
        cumulative_allocation = np.cumsum(allocation_sorted)
        numerator = 2 * np.sum((np.arange(1, m + 1) - 1) * allocation_sorted) - m * cumulative_allocation[-1]
        denominator = m * cumulative_allocation[-1]
        return numerator / denominator

    
    def evaluate_gini_index(self, num_rounds=10):
        results = {'round': list(range(1, num_rounds + 1))}
        cumulative_allocations = {voting_rule: [] for voting_rule in self.model.voting_rules.keys()}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_gini_index'] = []
        
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                allocation = self.model.allocate_funds(voting_rule)
                gini_index = self.calculate_gini_index(allocation)
                results[f'{voting_rule}_gini_index'].append(gini_index)
                
                # Store the sorted cumulative allocations for the Lorenz curve
                sorted_cumulative_allocation = np.cumsum(np.sort(allocation)) / np.sum(allocation)
                cumulative_allocations[voting_rule].append(sorted_cumulative_allocation)

        # Average the cumulative allocations across rounds for the Lorenz curve
        averaged_cumulative_allocations = {voting_rule: np.mean(cumulative_allocations[voting_rule], axis=0)
                                           for voting_rule in self.model.voting_rules.keys()}
        
        return pd.DataFrame(results), averaged_cumulative_allocations


    # Ground Truth Alignment
    def generate_ground_truth(self, num_projects):
        # Mallows model ground truth
        ground_truth = np.random.multinomial(self.model.total_op_tokens, [1.0/self.model.num_projects] * self.model.num_projects)

        #ground_truth = np.random.rand(num_projects)
        #ground_truth /= np.sum(ground_truth)
        return ground_truth

    # l1 distance here
    def calculate_hamming_distance(self, x, x_star, top_k):
        x_top_k = np.argsort(-x)[:top_k]
        x_star_top_k = np.argsort(-x_star)[:top_k]
        return np.sum(np.isin(x_top_k, x_star_top_k, invert=True))

    def evaluate_alignment(self, num_rounds=10):
        # Generate ground truth for alignment evaluation
        ground_truth = self.generate_ground_truth(self.model.num_projects)
        top_k = 200  # Number of top projects to compare for Hamming distance 

        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_l1_distance'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                allocation = self.model.allocate_funds(voting_rule)
                allocation_normalized = allocation / np.sum(allocation)
                l1_distance = self.calculate_l1_distance(allocation, ground_truth)
                results[f'{voting_rule}_l1_distance'].append(l1_distance)
        return pd.DataFrame(results)


    # Robustness
    def random_change_vote(self, vote, min_change_param,max_change_param,num_changes=None):
        """Modify the entire vote across all projects by a small, controlled amount."""
        new_vote = vote.copy()
        min_change=min_change_param * self.model.total_op_tokens
        max_change=max_change_param * self.model.total_op_tokens
        num_changes = 10
    
        # Randomly decide how many projects to change if num_changes is not specified
        if num_changes is None:
            num_changes = np.random.randint(1, self.model.num_projects + 1)
        
        # Randomly select which projects to change
        change_indices = np.random.choice(self.model.num_projects, num_changes, replace=False)
    
        # Iterate over all projects and randomly adjust each vote by a small amount
        for i in change_indices:
            change_value = np.random.randint(min_change, max_change)
            new_vote[i] = max(0, new_vote[i] + change_value)  # Ensure vote doesn't go below 0
        
        return new_vote

    def calculate_l1_distance(self, x, x_prime):
        """Calculate the L1 distance (Manhattan distance) between two vectors."""
        return np.sum(np.abs(x - x_prime))

    def evaluate_robustness(self, num_rounds=100,min_change_param=0.001, max_change_param=0.03):
        """
        Evaluate the robustness of the voting system by randomly changing a voter's vote and measuring the impact 
        across different voting rules.
        
        Parameters:
        - min_change_param: Minimum amount by which the vote can change.
        - max_change_param: Maximum amount by which the vote can change.
        - num_rounds: Number of rounds to simulate.

        Returns:
        - robustness_df: DataFrame containing the L1 distances for each voting rule and the changed vote distances.
        """
        # Initialize results dictionary
        robustness_results = {f"{method}_distances": [] for method in self.model.voting_rules.keys()}
        robustness_results["changed_vote_l1_distances"] = []

        # Track the overall progress for the number of rounds
        with tqdm(total=num_rounds, desc="Robustness Evaluation Progress", unit="round") as round_progress_bar:
            for round_num in range(num_rounds):
                print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

                # Randomly select a voter and change their vote
                voter_idx = np.random.randint(0, self.model.num_voters)
                original_vote = self.model.voting_matrix[voter_idx].copy()
                new_vote = self.random_change_vote(original_vote, min_change_param, max_change_param)

                # Calculate the magnitude of the vote change
                change_in_vote = np.sum(np.abs(new_vote - original_vote))
                robustness_results["changed_vote_l1_distances"].append(change_in_vote)

                print(f"[Round {round_num + 1}] Voter {voter_idx} changed their vote (L1 change: {change_in_vote:.4f})")

                # Apply the same vote change across all voting rules
                for method in tqdm(self.model.voting_rules.keys(), desc=f"Processing Voting Rules (Round {round_num + 1})", leave=False, unit="rule"):
                    # Original outcome
                    original_outcome = self.model.allocate_funds(method)
                    
                    # Create a new voting matrix with the modified vote
                    new_voting_matrix = self.model.voting_matrix.copy()
                    new_voting_matrix[voter_idx] = new_vote
                    self.model.voting_matrix = new_voting_matrix

                    # Calculate the outcome with the modified vote
                    start_time = time.time()
                    new_outcome = self.model.allocate_funds(method)
                    elapsed_time = time.time() - start_time

                    # Calculate the L1 distance between the original and new outcomes
                    distance = self.calculate_l1_distance(original_outcome, new_outcome)
                    robustness_results[f"{method}_distances"].append(distance)

                    # Log the result for the current method
                    print(f"[Round {round_num + 1}] [Voting Rule: {method}] L1 Distance: {distance:.4f} (Time: {elapsed_time:.2f}s)")

                    # Restore the original voting matrix for the next round
                    self.model.voting_matrix[voter_idx] = original_vote

                # Update the round progress bar after completing all voting rules
                round_progress_bar.update(1)

        # Convert the results to a DataFrame
        robustness_df = pd.DataFrame(robustness_results)
        robustness_df["round"] = robustness_df.index + 1

        print("\nAll rounds completed. Final results:\n", robustness_df)
        return robustness_df

    # Social Welfare

    def evaluate_social_welfare(self,num_rounds=10):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_social_welfare_avg_l1_distance'] = []

        for round_num in range(num_rounds):
            self.model.step()   
            for voting_rule in self.model.voting_rules.keys():
                outcome = self.model.allocate_funds(voting_rule)
                total_distance = 0
                for i in range(self.model.num_voters):
                    total_distance += self.calculate_l1_distance(outcome, self.model.voting_matrix[i])
                average_distance = total_distance /self. model.num_voters
                results[f'{voting_rule}_social_welfare_avg_l1_distance'].append(average_distance)

        return pd.DataFrame(results)


    def calculate_egalitarian_score(self, allocation, voting_matrix):
        distances = np.linalg.norm(voting_matrix - allocation, ord=1, axis=1)
        return np.max(distances)

    def evaluate_egalitarian_score(self, num_rounds=10):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_egalitarian_score'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                allocation = self.model.allocate_funds(voting_rule)
                egalitarian_score = self.calculate_egalitarian_score(allocation, self.model.voting_matrix)
                results[f'{voting_rule}_egalitarian_score'].append(egalitarian_score)
        return pd.DataFrame(results)

    def simulate_voter_addition(self, project, voting_rule, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        original_allocation = self.model.allocate_funds(voting_rule)
        original_funds = original_allocation[project]
        target_funds = original_funds * (1 + desired_increase / 100)
        added_voters = 0

        #print(f"Voting Rule: {voting_rule}: Original Funds for Project {project}: {original_funds}")
        #print(f"Voting Rule: {voting_rule}: Target Funds for Project {project}: {target_funds}")

        max_additional_voters = num_voters * 0.5  # Limit to avoid infinite loop

        # Create a copy of the original voting matrix to work with
        potential_voting_matrix = self.model.voting_matrix.copy()

        while added_voters < max_additional_voters:
            new_voter = np.zeros(num_projects)
            new_voter[project] = self.model.total_op_tokens  # New voters give all their votes to the target project

            # Update the voting matrix cumulatively with the new voter
            potential_voting_matrix = np.vstack([potential_voting_matrix, new_voter])

            # Temporarily replace the voting matrix
            original_matrix = self.model.voting_matrix
            self.model.voting_matrix = potential_voting_matrix

            try:
                new_allocation = self.model.allocate_funds(voting_rule)
                new_funds = new_allocation[project]

                #print(f"Voting Rule: {voting_rule}: New Funds for Project {project} after adding {added_voters + 1} voters: {new_funds}")

                if new_funds >= target_funds:
                    return added_voters + 1  # Number of voters added
            finally:
                # Always restore the original voting matrix
                self.model.voting_matrix = original_matrix

            added_voters += 1

        return np.inf  # If no solution is found

    def simulate_voter_removal(self, project, voting_rule, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        original_allocation = self.model.allocate_funds(voting_rule)
        original_funds = original_allocation[project]
        target_funds = original_funds * (1 + desired_increase / 100)

        # Make a copy of the voting matrix to modify during the simulation
        potential_voting_matrix = self.model.voting_matrix.copy()

        current_num_voters = num_voters

        #print(f"Voting Rule: {voting_rule} Original Funds for Project {project}: {original_funds}")
        #print(f"Voting Rule: {voting_rule} Target Funds for Project {project}: {target_funds}")

        for i in range(num_voters):
            if current_num_voters == 0:
                break  # No voters left to remove

            # Sort voters by their current influence after each iteration?? reverse the list
            #current_votes = potential_voting_matrix[:, project]
            #voters_sorted_by_influence = np.argsort(current_votes)[::-1]
            current_votes = potential_voting_matrix[:, project]
            voters_sorted_by_influence = np.argsort(current_votes)

            # Remove the most influential voter for other projects
            potential_voting_matrix = np.delete(potential_voting_matrix, voters_sorted_by_influence[0], axis=0)
            current_num_voters -= 1

            # **Exit early if voting matrix is empty**:
            if potential_voting_matrix.shape[0] == 0:
                print("No voters left in the matrix. Cannot proceed.")
                return np.inf  # Exit early as no more voters are left to manipulate the project funds

            # Recalculate the allocation after voter removal
            original_matrix = self.model.voting_matrix
            self.model.voting_matrix = potential_voting_matrix

            try:
                new_allocation = self.model.allocate_funds(voting_rule)
                new_funds = new_allocation[project]

                #print(f"Voting Rule: {voting_rule}: New Funds for Project {project} after removing {i + 1} voters: {new_funds}")

                if new_funds >= target_funds:
                    return i + 1  # Number of voters removed
                
                if new_funds <= original_funds * 0.5:
                    break

            finally:
                # Restore the original voting matrix
                self.model.voting_matrix = original_matrix

        return np.inf  # Not possible to achieve the desired increase by removing voters

    
    def evaluate_control(self, num_rounds=10, desired_increase=20):
        """
        Evaluate the resistance to control by adding or removing voters for a desired percentage increase in funding.
        """
        results = {'round': list(range(1, num_rounds + 1)), 'desired_increase': []}

        # Initialize columns for removal and addition costs for each voting rule
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_min_removal_cost'] = []
            results[f'{voting_rule}_min_addition_cost'] = []

        num_iterations = num_rounds * self.model.num_voters * self.model.num_projects

        # Outer loop for the number of rounds
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            self.model.step()  # Simulate the next round
            results['desired_increase'].append(desired_increase)

            # Track progress for each round
            for voting_rule in tqdm(self.model.voting_rules.keys(), desc=f"Processing Voting Rules (Round {round_num + 1})", unit="rule"):
                min_removal_cost = np.inf
                min_addition_cost = np.inf
                removal_possible = False

                # Track progress for each project
                for project in range(self.model.num_projects):
                    project_start_time = time.time()

                    # Calculate the cost to remove voters
                    removal_cost = self.simulate_voter_removal(project, voting_rule, desired_increase)
                    if removal_cost < np.inf:
                        removal_possible = True
                        min_removal_cost = min(min_removal_cost, removal_cost)

                    # Calculate the cost to add voters
                    addition_cost = self.simulate_voter_addition(project, voting_rule, desired_increase)
                    min_addition_cost = min(min_addition_cost, addition_cost)

                    # Log progress for each voter-project combination
                    elapsed_time = time.time() - project_start_time
                    print(f"[Round {round_num + 1}] [Project {project + 1}/{self.model.num_projects}] "
                        f"Voting Rule: {voting_rule}, Removal Cost: {removal_cost:.4f}, "
                        f"Addition Cost: {addition_cost:.4f}, Time: {elapsed_time:.2f}s")

                # Append the removal and addition costs for the current voting rule
                results[f'{voting_rule}_min_removal_cost'].append(min_removal_cost if removal_possible else "Not Possible")
                results[f'{voting_rule}_min_addition_cost'].append(min_addition_cost)

        # Convert results to a DataFrame
        final_results = pd.DataFrame(results)
        print("\nAll rounds completed. Final results:\n", final_results)
        return final_results
        
    def evaluate_vev(self, num_rounds=100, r_min=90, r_max=99):
        """
        Evaluate the Voter Extractable Value (VEV) for a given voting rule.

        Parameters:
        - num_rounds: Number of instances to compute VEV across different vote profiles.
        - r_min: Minimum percentage allocation to the specific project (default: 90%).
        - r_max: Maximum percentage allocation to the specific project (default: 99%).

        Returns:
        - VEV_results: DataFrame containing the VEV for each instance and voting rule.
        """
        results = {
            'round': [],        # Store round info for each instance and rule
            'voting_rule': [],  # Store the voting rule for each instance
            'max_vev': [],
            'project_max_vev':[],
            'project_max_original_allocation':[],
            'project_max_new_allocation':[],
            'project_max_allocation_percentage':[]       # Store the maximum VEV for each instance
        }
        r_values = np.linspace(r_min / 100, r_max / 100, 5)
        
        for instance in range(1, num_rounds + 1):  # Loop through rounds
            self.model.step()  # Simulate a new vote profile

            for voting_rule in self.model.voting_rules.keys():
                max_vev = float('-inf')  # Track the maximum VEV for this rule

                project_max_vev = float('-inf')
                project_max_new_allocation = float('-inf')
                project_max_original_allocation  = float('-inf')

                # Get the original allocation using the voting rule
                original_allocation = self.model.allocate_funds(voting_rule)
                #project_original_allocation = original_allocation[project]
                num_iterations = self.model.num_voters * self.model.num_projects * 5  
                with tqdm(total=num_iterations) as pbar:
                    for voter in range(self.model.num_voters):
                        for project in range(self.model.num_projects):
                            project_original_allocation = original_allocation[project]
                            # Iterate through r values from r_min to r_max (90% to 99%)
                            for r in r_values:
                                # Create a modified vote for this voter and project
                                start_time = time.time()
                                modified_vote_matrix = self.model.voting_matrix.copy()
                                modified_vote_matrix[voter] = self.modify_vote(voter, project, r)
                                
                                # Apply the modified vote profile to get a new allocation
                                new_allocation = self.model.allocate_funds(voting_rule, modified_vote_matrix)
                                project_new_allocation=new_allocation[project]
                                
                                # Calculate the L1 distance between original and new allocation
                                l1_distance = np.sum(np.abs(original_allocation - new_allocation))
                                project_allocation_difference=project_new_allocation-project_original_allocation

                                elapsed_time = time.time() - start_time
                                print(f"[Round {instance}] [Voter {voter}/{self.model.num_voters}] [Project {project}/{self.model.num_projects}] "
                                f"r={r:.2f} L1={l1_distance:.4f} Allocation Diff={project_allocation_difference:.4f} "
                                f"Max VEV={max_vev:.4f} Elapsed Time: {elapsed_time:.2f}s")


                                # Progress bar update
                                pbar.update(1)
                                # Update the maximum VEV if this is the largest skewness
                                if l1_distance > max_vev:
                                    max_vev = l1_distance
                                if project_allocation_difference > project_max_vev:
                                    project_max_vev = project_allocation_difference
                                    project_max_new_allocation = project_new_allocation
                                    project_max_original_allocation = project_original_allocation
                                #if project_new_allocation > project_max_new_allocation:
                                #    project_max_new_allocation = project_new_allocation
                                #    project_max_original_allocation = project_original_allocation

                # Log the maximum VEV for this instance and voting rule
                results['round'].append(instance)  # Add round number dynamically
                results['voting_rule'].append(voting_rule)  # Add the voting rule
                results['max_vev'].append(max_vev)  # Add the maximum VEV
                results['project_max_vev'].append(project_max_vev)
                results['project_max_original_allocation'].append(project_max_original_allocation)
                results['project_max_new_allocation'].append(project_max_new_allocation)
                results['project_max_allocation_percentage'].append(project_max_new_allocation/self.model.total_op_tokens)

        # Create a DataFrame to store results
        VEV_results = pd.DataFrame(results)
        return VEV_results


    def modify_vote(self, voter, project, r):
        """
        Modify the vote of voter `i` by allocating r% of their funds to project `k`.
        The remainder is distributed equally across the other projects.
        
        Parameters:
        - voter: Index of the voter whose vote will be modified.
        - project: Index of the project where the majority of the funds will go.
        - r: The percentage of total funds allocated to the selected project.

        Returns:
        - modified_vote: The new vote profile for the voter with modified allocations.
        """
        total_funds = self.model.total_op_tokens
        num_projects = self.model.num_projects

        # Allocate r% of funds to the selected project
        modified_vote = np.zeros(num_projects)
        modified_vote[project] = r * total_funds

        # Distribute the remaining (1 - r)% equally across the other projects
        remaining_funds = (1 - r) * total_funds
        for other_project in range(num_projects):
            if other_project != project:
                modified_vote[other_project] = remaining_funds / (num_projects - 1)

        return modified_vote
