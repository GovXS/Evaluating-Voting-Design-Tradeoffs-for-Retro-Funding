import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class EvalMetrics:
    def __init__(self, model):
        self.model = model

    # Bribery Cost
    def simulate_bribery(self, method, target_project, desired_increase):
        if method == "r2_mean":
            return self.simulate_bribery_mean(target_project, desired_increase)
        elif method == "r3_median":
            return self.simulate_bribery_median(target_project, desired_increase)
        elif method == "r1_quadratic":
            return self.simulate_bribery_quadratic(target_project, desired_increase)
        else:
            raise ValueError(f"Unknown method for bribery simulation: {method}")

    def simulate_bribery_mean(self, target_project, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        new_voting_matrix = self.model.voting_matrix.copy()

        original_allocation = self.model.allocate_funds("r2_mean")
        original_funds = original_allocation[target_project]

        target_funds = original_funds + desired_increase
        total_required_votes = target_funds * num_voters

        current_votes = np.sum(new_voting_matrix[:, target_project])
        votes_needed = total_required_votes - current_votes

        bribery_cost = votes_needed
        return bribery_cost

    def simulate_bribery_quadratic(self, target_project, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        new_voting_matrix = self.model.voting_matrix.copy()

        original_allocation = self.model.allocate_funds("r1_quadratic")
        original_funds = original_allocation[target_project]

        target_funds = original_funds + desired_increase
        total_required_votes = (target_funds * np.sum(original_allocation)) ** 0.5

        current_votes = np.sum(new_voting_matrix[:, target_project] ** 2)
        votes_needed = total_required_votes - np.sqrt(current_votes)
        votes_needed = votes_needed ** 2

        bribery_cost = votes_needed
        return bribery_cost

    def simulate_bribery_median(self, target_project, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        new_voting_matrix = self.model.voting_matrix.copy()

        original_allocation = self.model.allocate_funds("r3_median")
        original_funds = original_allocation[target_project]

        target_funds = original_funds + desired_increase
        votes = new_voting_matrix[:, target_project]
        current_median_vote = np.median(votes)

        total_required_votes = target_funds * self.model.num_voters / self.model.total_op_tokens
        votes_needed = total_required_votes - current_median_vote

        bribery_cost = desired_increase
        return bribery_cost

    def evaluate_bribery(self, num_rounds):
        max_bribe = 1e6
        desired_increases = np.linspace(max_bribe / num_rounds, max_bribe, num_rounds)
        results = {'round': list(range(1, num_rounds + 1))}
        results = {'round': list(range(1, num_rounds + 1)), 'desired_increase': []}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_bribery_cost'] = []

        for i in range(num_rounds):
            self.model.step()

            target_project = np.random.randint(0, self.model.num_projects)  # Randomly select a target project
            desired_increase = desired_increases[i]
            results['desired_increase'].append(desired_increase)

            for voting_rule in self.model.voting_rules.keys():
                if voting_rule == "r2_mean":
                    bribery_cost = self.simulate_bribery_mean(target_project, desired_increase)
                elif voting_rule == "r3_median":
                    bribery_cost = self.simulate_bribery_median(target_project, desired_increase)
                elif voting_rule == "r1_quadratic":
                    bribery_cost = self.simulate_bribery_quadratic(target_project, desired_increase)
                else:
                    print(f"Bribery Cost Calculation Function for {voting_rule} is not defined in EvalMetrics")
                    bribery_cost = 0

                results[f'{voting_rule}_bribery_cost'].append(bribery_cost)

        final_results = pd.DataFrame(results)
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

    def evaluate_gini_index(self, num_rounds):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_gini_index'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                allocation = self.model.allocate_funds(voting_rule)
                gini_index = self.calculate_gini_index(allocation)
                results[f'{voting_rule}_gini_index'].append(gini_index)
        return pd.DataFrame(results)

  

    def lorenz_curve(self,allocation):
        """Calculate and plot the Lorenz curve for a given allocation."""
        sorted_allocation = np.sort(allocation)
        cumulative_allocation = np.cumsum(sorted_allocation) / np.sum(sorted_allocation)
        cumulative_allocation = np.insert(cumulative_allocation, 0, 0)  # Insert the origin (0,0)
        return cumulative_allocation

    def plot_lorenz_curves(self,allocations, voting_rules):
        """Plot Lorenz curves for multiple voting rules."""
        plt.figure(figsize=(8, 8))
        
        for voting_rule in voting_rules:
            allocation = allocations[voting_rule]
            lorenz_values = self.lorenz_curve(allocation)
            plt.plot(np.linspace(0, 1, len(lorenz_values)), lorenz_values, label=voting_rule)
        
        # Line of equality
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        
        plt.xlabel('Cumulative Population Proportion')
        plt.ylabel('Cumulative Funding Proportion')
        plt.title('Lorenz Curve for Different Voting Rules')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_and_plot_lorenz_curve(self, num_rounds):
        """Evaluate Gini index and plot Lorenz curves for multiple voting rounds."""
        allocations = {voting_rule: np.zeros(self.model.num_projects) for voting_rule in self.model.voting_rules.keys()}
        
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                allocation = self.model.allocate_funds(voting_rule)
                allocations[voting_rule] += allocation  # Aggregate allocation across rounds
        
        # Calculate the average allocation across rounds
        for voting_rule in allocations:
            allocations[voting_rule] /= num_rounds
        
        # Plot the Lorenz curves
        self.plot_lorenz_curves(allocations, self.model.voting_rules.keys())


    # Ground Truth Alignment
    def generate_ground_truth(self, num_projects):
        ground_truth = np.random.rand(num_projects)
        ground_truth /= np.sum(ground_truth)
        return ground_truth

    def calculate_hamming_distance(self, x, x_star, top_k):
        x_top_k = np.argsort(-x)[:top_k]
        x_star_top_k = np.argsort(-x_star)[:top_k]
        return np.sum(np.isin(x_top_k, x_star_top_k, invert=True))

    def evaluate_alignment(self, num_rounds):
        # Generate ground truth for alignment evaluation
        ground_truth = self.generate_ground_truth(self.model.num_projects)
        top_k = 200  # Number of top projects to compare for Hamming distance 

        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_hamming_distance'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                allocation = self.model.allocate_funds(voting_rule)
                allocation_normalized = allocation / np.sum(allocation)
                hamming_distance = self.calculate_hamming_distance(allocation_normalized, ground_truth, top_k)
                results[f'{voting_rule}_hamming_distance'].append(hamming_distance)
        return pd.DataFrame(results)

    # Group Strategyproofness
    def utility(self, voter_preferences, outcome):
        return -np.sum(np.abs(voter_preferences - outcome))

    def evaluate_group_strategyproofness(self, coalition_size=3):
        group_strategyproofness_results = []
        for method in self.model.voting_rules.keys():
            truthfully_voted_outcome = self.model.allocate_funds(method)
            group_strategyproof = True
            for _ in range(100):
                coalition = np.random.choice(self.model.num_voters, coalition_size, replace=False)
                original_utilities = [self.utility(self.model.voting_matrix[i], truthfully_voted_outcome) for i in coalition]
                strategic_voting_matrix = self.model.voting_matrix.copy()
                for i in coalition:
                    strategic_voting_matrix[i] = np.random.rand(self.model.num_projects)
                self.model.voting_matrix = strategic_voting_matrix
                strategically_voted_outcome = self.model.allocate_funds(method)
                new_utilities = [self.utility(self.model.voting_matrix[i], strategically_voted_outcome) for i in coalition]
                if all(new_utilities[i] > original_utilities[i] for i in range(coalition_size)):
                    group_strategyproof = False
                    break
                self.model.voting_matrix = strategic_voting_matrix
            group_strategyproofness_results.append({
                "voting_rule": method,
                "group_strategyproof": group_strategyproof
            })
        return pd.DataFrame(group_strategyproofness_results)

    # Resistance to Control
    def add_remove_projects(self, project_to_manipulate, voting_rule, add):
        original_allocation = self.model.allocate_funds(voting_rule)
        current_funds = original_allocation[project_to_manipulate]
        if add:
            new_project = np.random.rand(self.model.num_voters, 1)
            new_project /= new_project.sum(axis=0, keepdims=True)
            self.model.voting_matrix = np.hstack((self.model.voting_matrix, new_project))
            new_allocation = self.model.allocate_funds(voting_rule)
            new_funds = new_allocation[project_to_manipulate]
        else:
            remove_idx = np.random.choice(self.model.num_projects, 1)
            self.model.voting_matrix = np.delete(self.model.voting_matrix, remove_idx, axis=1)
            new_allocation = self.model.allocate_funds(voting_rule)
            new_funds = new_allocation[project_to_manipulate]
        budget = abs(new_funds - current_funds) * self.model.num_voters
        return budget, new_allocation

    def add_remove_voters(self, project_to_manipulate, voting_rule, add):
        original_allocation = self.model.allocate_funds(voting_rule)
        current_funds = original_allocation[project_to_manipulate]
        if add:
            new_voter = np.random.rand(1, self.model.num_projects)
            new_voter /= new_voter.sum(axis=1, keepdims=True)
            self.model.voting_matrix = np.vstack((self.model.voting_matrix, new_voter))
            new_allocation = self.model.allocate_funds(voting_rule)
            new_funds = new_allocation[project_to_manipulate]
        else:
            remove_idx = np.random.choice(self.model.num_voters, 1)
            self.model.voting_matrix = np.delete(self.model.voting_matrix, remove_idx, axis=0)
            new_allocation = self.model.allocate_funds(voting_rule)
            new_funds = new_allocation[project_to_manipulate]
        budget = abs(new_funds - current_funds) * self.model.num_projects
        return budget, new_allocation

    def evaluate_control(self, num_rounds):
        # Define control functions
        control_functions = {
            "add_projects": lambda m, p, vr: self.add_remove_projects(p, vr, True),
            "remove_projects": lambda m, p, vr: self.add_remove_projects(p, vr, False),
            "add_voters": lambda m, p, vr: self.add_remove_voters(p, vr, True),
            "remove_voters": lambda m, p, vr: self.add_remove_voters(p, vr, False)
        }

        # Select a random project for manipulation
        project_to_manipulate = np.random.randint(self.model.num_projects)

        results = []
        for _ in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                round_results = {'voting_rule': voting_rule}
                for control_name, control_function in control_functions.items():
                    budget, new_allocation = control_function(self.model, project_to_manipulate, voting_rule)
                    round_results[control_name] = budget
                results.append(round_results)
        return pd.DataFrame(results)

    # Robustness
    def random_change_vote(self, vote, change_amount=0.01,num_changes=None):
        """Modify the entire vote across all projects by a small, controlled amount."""
        new_vote = vote.copy()
        min_change=0.00001
        max_change=0.1
        num_changes = 2
    
        # Randomly decide how many projects to change if num_changes is not specified
        if num_changes is None:
            num_changes = np.random.randint(1, self.model.num_projects + 1)
        
        # Randomly select which projects to change
        change_indices = np.random.choice(self.model.num_projects, num_changes, replace=False)
    
        # Iterate over all projects and randomly adjust each vote by a small amount
        for i in change_indices:
            change_value = np.random.randint(min_change*self.model.total_op_tokens, max_change*self.model.total_op_tokens)
            new_vote[i] = max(0, new_vote[i] + change_value)  # Ensure vote doesn't go below 0
        
        return new_vote

    def calculate_l1_distance(self, x, x_prime):
        """Calculate the L1 distance (Manhattan distance) between two vectors."""
        return np.sum(np.abs(x - x_prime))

    def evaluate_robustness(self, num_rounds):
        robustness_results = {f"{method}_distances": [] for method in self.model.voting_rules.keys()}
        robustness_results["changed_vote_l1_distances"] = []

        for _ in range(num_rounds):
            # Randomly select a voter and change their vote
            voter_idx = np.random.randint(0, self.model.num_voters)
            original_vote = self.model.voting_matrix[voter_idx].copy()
            new_vote = self.random_change_vote(original_vote)

            # Calculate the magnitude of the vote change
            change_in_vote = np.sum(np.abs(new_vote - original_vote))
            robustness_results["changed_vote_l1_distances"].append(change_in_vote)

            # Apply the same vote change across all voting rules
            for method in self.model.voting_rules.keys():
                original_outcome = self.model.allocate_funds(method)
                
                # Create a new voting matrix with the modified vote
                old_voting_matrix=self.model.voting_matrix.copy()
                new_voting_matrix = self.model.voting_matrix.copy()
                #print(f"before:{self.model.voting_matrix}")
                new_voting_matrix[voter_idx] = new_vote
                #print(f"before:{self.model.voting_matrix-new_voting_matrix}")
                self.model.voting_matrix = new_voting_matrix
                #print(f"during:{self.model.voting_matrix}")

                # Calculate the outcome with the modified vote
                new_outcome = self.model.allocate_funds(method)
                distance = self.calculate_l1_distance(original_outcome, new_outcome)
                robustness_results[f"{method}_distances"].append(distance)

                # Restore the original vote for the next round
                self.model.voting_matrix[voter_idx] = original_vote
                #print(f"after:{self.model.voting_matrix-old_voting_matrix}")

        # Convert the results to a DataFrame
        robustness_df = pd.DataFrame(robustness_results)
        robustness_df["round"] = robustness_df.index + 1
        return robustness_df

    # Social Welfare
    def calculate_social_welfare(self, allocation, voting_matrix):
        n = voting_matrix.shape[0]
        total_distance = np.sum(np.linalg.norm(voting_matrix - allocation, ord=1, axis=1))
        return total_distance / n

    def evaluate_social_welfare(self, num_rounds):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_social_welfare'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                allocation = self.model.allocate_funds(voting_rule)
                social_welfare = self.calculate_social_welfare(allocation, self.model.voting_matrix)
                results[f'{voting_rule}_social_welfare'].append(social_welfare)
        return pd.DataFrame(results)

    def l1_distance(x, xi):
        return np.sum(np.abs(x - xi))

    def evaluate_social_welfare_1(self,num_rounds):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_social_welfare'] = []

        for round_num in range(num_rounds):
            self.model.step()   
            for voting_rule in self.model.voting_rules.keys():
                outcome = self.model.allocate_funds(voting_rule)
                total_distance = 0
                for i in range(self.model.num_voters):
                    total_distance += self.l1_distance(outcome, self.model.voting_matrix[i])
                average_distance = total_distance / model.num_voters
                results[f'{voting_rule}_social_welfare_avg_l1_distance'].append(average_distance)

        return pd.DataFrame(results)


    def calculate_egalitarian_score(self, allocation, voting_matrix):
        distances = np.linalg.norm(voting_matrix - allocation, ord=1, axis=1)
        return np.max(distances)

    def evaluate_egalitarian_score(self, num_rounds):
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