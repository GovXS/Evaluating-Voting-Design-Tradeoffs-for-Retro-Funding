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
        elif method == "majoritarian_moving_phantoms":
            return self.simulate_bribery_majoritarian_moving_phantoms(target_project, desired_increase)
        elif method == "r4_capped_median":
            return self.simulate_bribery_capped_median(target_project, desired_increase)
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


    # New bribery simulation for 'majoritarian_moving_phantoms'
    def simulate_bribery_majoritarian_moving_phantoms(self, target_project, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        original_allocation = self.model.allocate_funds("majoritarian_moving_phantoms")
        original_funds = original_allocation[target_project]
        target_funds = original_funds + desired_increase

        for i in range(num_voters):
            new_voting_matrix = self.model.voting_matrix.copy()
            new_voting_matrix[:, target_project] += 1  # Increment votes for the target project
            new_allocation = self.model.allocate_funds("majoritarian_moving_phantoms")
            if new_allocation[target_project] >= target_funds:
                return i + 1  # Number of voters manipulated

        return np.inf  # Return infinity if bribery isn't possible

    # New bribery simulation for 'r4_capped_median'
    def simulate_bribery_capped_median(self, target_project, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        original_allocation = self.model.allocate_funds("r4_capped_median")
        original_funds = original_funds[target_project]
        target_funds = original_funds + desired_increase

        for i in range(num_voters):
            new_voting_matrix = self.model.voting_matrix.copy()
            new_voting_matrix[:, target_project] = np.minimum(new_voting_matrix[:, target_project] + 1, 500000)  # Cap the increment to K1
            new_allocation = self.model.allocate_funds("r4_capped_median")
            if new_allocation[target_project] >= target_funds:
                return i + 1  # Number of voters manipulated

        return np.inf  # Return infinity if bribery isn't possible


    def evaluate_bribery(self, num_rounds):
        min_desired_increase = 0.01  # 1% increase
        max_desired_increase = 0.5  # 50% increase
        desired_increase_percentage = np.linspace(max_desired_increase / num_rounds, max_desired_increase, num_rounds)
        results = {'round': list(range(1, num_rounds + 1))}
        results = {'round': list(range(1, num_rounds + 1)), 'desired_increase': []}
        
        for voting_rule in self.model.voting_rules.keys():
            results[f'{voting_rule}_bribery_cost'] = []

        for i in range(num_rounds):
            self.model.step()

            min_bribery_costs = {}
            desired_increase_percentage_current_round=desired_increase_percentage[i]
            results['desired_increase'].append(desired_increase_percentage_current_round)

            for voting_rule in self.model.voting_rules.keys():
                original_allocation = self.model.allocate_funds(voting_rule)
                
                min_bribery_cost = float('inf')
                for project in range(self.model.num_projects):
                    original_funds = original_allocation[project]
                    desired_increase = original_funds * desired_increase_percentage_current_round
                    
                    bribery_cost = self.simulate_bribery(voting_rule, project, desired_increase)
                    
                    if bribery_cost < min_bribery_cost:
                        min_bribery_cost = bribery_cost
                
                min_bribery_costs[voting_rule] = min_bribery_cost

            for voting_rule, min_cost in min_bribery_costs.items():
                results[f'{voting_rule}_bribery_cost'].append(min_cost)

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

    def evaluate_alignment(self, num_rounds):
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

    def evaluate_social_welfare(self,num_rounds):
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

    def simulate_voter_removal(self, project, voting_rule, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        original_allocation = self.model.allocate_funds(voting_rule)
        original_funds = original_allocation[project]

        target_funds = original_funds * (1 + desired_increase / 100)
        current_votes = self.model.voting_matrix[:, project]
        voters_sorted_by_influence = np.argsort(current_votes)[::-1]

        for i in range(num_voters):
            potential_voting_matrix = self.model.voting_matrix.copy()
            potential_voting_matrix[voters_sorted_by_influence[:i+1], :] = 0

            # Temporarily replace the voting matrix
            original_matrix = self.model.voting_matrix
            self.model.voting_matrix = potential_voting_matrix

            try:
                new_allocation = self.model.allocate_funds(voting_rule)
            finally:
                # Restore the original voting matrix
                self.model.voting_matrix = original_matrix
            
            new_funds = new_allocation[project]

            if new_funds >= target_funds:
                return i + 1  # Number of voters removed

        return np.inf  # Not possible to achieve the desired increase by removing voters

    def simulate_voter_addition(self, project, voting_rule, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        original_allocation = self.model.allocate_funds(voting_rule)
        original_funds = original_allocation[project]

        target_funds = original_funds * (1 + desired_increase / 100)
        added_voters = 0

        while True:
            new_voter = np.zeros(num_projects)
            new_voter[project] = 1  # New voters give all their votes to the target project

            potential_voting_matrix = np.vstack([self.model.voting_matrix, new_voter])

            # Temporarily replace the voting matrix
            original_matrix = self.model.voting_matrix
            self.model.voting_matrix = potential_voting_matrix

            try:
                new_allocation = self.model.allocate_funds(voting_rule)
            finally:
                # Restore the original voting matrix
                self.model.voting_matrix = original_matrix
            
            new_funds = new_allocation[project]

            if new_funds >= target_funds:
                return added_voters + 1  # Number of voters added

            added_voters += 1
            self.model.voting_matrix = potential_voting_matrix

    def evaluate_control(self, num_rounds, desired_increase):
        """
        Evaluate the resistance to control by adding or removing voters for a desired percentage increase in funding.
        """
        results = []

        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in self.model.voting_rules.keys():
                min_removal_cost = np.inf
                min_addition_cost = np.inf
                removal_possible = False

                for project in range(self.model.num_projects):
                    # Calculate the cost to remove voters
                    removal_cost = self.simulate_voter_removal(project, voting_rule, desired_increase)
                    if removal_cost < np.inf:
                        removal_possible = True
                        min_removal_cost = min(min_removal_cost, removal_cost)

                    # Calculate the cost to add voters
                    addition_cost = self.simulate_voter_addition(project, voting_rule, desired_increase)
                    min_addition_cost = min(min_addition_cost, addition_cost)

                results.append({
                    'round': round_num + 1,
                    'voting_rule': voting_rule,
                    'min_removal_cost': min_removal_cost if removal_possible else "Not Possible",
                    'min_addition_cost': min_addition_cost,
                    'desired_increase': desired_increase
                })

        return pd.DataFrame(results)
    
    