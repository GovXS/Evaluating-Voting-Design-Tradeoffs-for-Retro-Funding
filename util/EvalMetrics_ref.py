# model/EvalMetrics.py

import numpy as np
import pandas as pd
from model.VotingRules import mean_aggregation, median_aggregation, quadratic_aggregation

class EvalMetrics:
    def __init__(self, model):
        self.model = model

    # Bribery Cost
    def simulate_bribery_mean(self, target_project, desired_increase):
        num_voters, num_projects = self.model.voting_matrix.shape
        new_voting_matrix = self.model.voting_matrix.copy()

        original_allocation = self.model.mean_aggregation()
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

        original_allocation = self.model.quadratic_aggregation()
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
        original_allocation = self.model.median_aggregation()
        original_funds = original_allocation[target_project]
        target_funds = original_funds + desired_increase
        votes = new_voting_matrix[:, target_project]
        current_median_vote = np.median(votes)
        total_required_votes = target_funds * self.model.num_voters / self.model.total_op_tokens
        votes_needed = total_required_votes - current_median_vote
        bribery_cost = desired_increase
        return bribery_cost

    def evaluate_bribery_impact(self, target_project, desired_increase):
        bribery_costs = {}
        for method in ["r1_quadratic", "r2_mean", "r3_median"]:
            if method == "r1_quadratic":
                bribery_cost = self.simulate_bribery_mean(target_project, desired_increase)
            elif method == "r2_mean":
                bribery_cost = self.simulate_bribery_median(target_project, desired_increase)
            elif method == "r3_median":
                bribery_cost = self.simulate_bribery_quadratic(target_project, desired_increase)
            bribery_costs[method + "_bribery_cost"] = bribery_cost
        return bribery_costs

    # Gini Index
    def calculate_gini_index(self, allocation):
        m = len(allocation)
        if m == 0:
            return 0
        allocation_sorted = np.sort(allocation)
        cumulative_allocation = np.cumsum(allocation_sorted)
        numerator = 2 * np.sum((np.arange(1, m + 1) - 1) * allocation_sorted) - m * cumulative_allocation[-1]
        denominator = m * cumulative_allocation[-1]
        gini_index = numerator / denominator
        return gini_index

    def evaluate_gini_index(self, num_rounds, voting_rules):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in voting_rules:
            results[f'{voting_rule}_gini_index'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in voting_rules:
                allocation = self.model.allocate_funds(voting_rule)
                gini_index = self.calculate_gini_index(allocation)
                results[f'{voting_rule}_gini_index'].append(gini_index)
        return pd.DataFrame(results)

    # Ground Truth Alignment
    def generate_ground_truth(self, num_projects):
        ground_truth = np.random.rand(num_projects)
        ground_truth /= np.sum(ground_truth)
        return ground_truth

    def calculate_hamming_distance(self, x, x_star, top_k):
        x_top_k = np.argsort(-x)[:top_k]
        x_star_top_k = np.argsort(-x_star)[:top_k]
        return np.sum(np.isin(x_top_k, x_star_top_k, invert=True))

    def evaluate_alignment(self, num_rounds, voting_rules, ground_truth, top_k):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in voting_rules:
            results[f'{voting_rule}_hamming_distance'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in voting_rules:
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
        for method in ["mean", "median", "quadratic"]:
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

    def simulate_control(self, num_rounds, control_functions, project_to_manipulate, voting_rules, *args):
        results = []
        for _ in range(num_rounds):
            self.model.step()
            for voting_rule in voting_rules:
                round_results = {'voting_rule': voting_rule}
                for control_name, control_function in control_functions.items():
                    budget, new_allocation = control_function(self.model, project_to_manipulate, voting_rule)
                    round_results[control_name] = budget
                results.append(round_results)
        return pd.DataFrame(results)

    # Robustness
    def random_change_vote(self, vote, num_projects):
        new_vote = vote.copy()
        change_idx = np.random.randint(0, num_projects)
        new_vote[change_idx] = np.random.uniform(0, 1)
        return new_vote

    def calculate_distance(self, x, x_prime):
        return np.linalg.norm(x - x_prime)

    def evaluate_robustness(self, rounds):
        robustness_results = {
            "mean_distances": [],
            "median_distances": [],
            "quadratic_distances": []
        }
        for method in ["mean", "median", "quadratic"]:
            for _ in range(rounds):
                original_outcome = self.model.allocate_funds(method)
                voter_idx = np.random.randint(0, self.model.num_voters)
                original_vote = self.model.voting_matrix[voter_idx].copy()
                new_vote = self.random_change_vote(original_vote, self.model.num_projects)
                new_voting_matrix = self.model.voting_matrix.copy()
                new_voting_matrix[voter_idx] = new_vote
                self.model.voting_matrix = new_voting_matrix
                new_outcome = self.model.allocate_funds(method)
                distance = self.calculate_distance(original_outcome, new_outcome)
                if method == "mean":
                    robustness_results["mean_distances"].append(distance)
                elif method == "median":
                    robustness_results["median_distances"].append(distance)
                elif method == "quadratic":
                    robustness_results["quadratic_distances"].append(distance)
                self.model.voting_matrix[voter_idx] = original_vote
        robustness_df = pd.DataFrame(robustness_results)
        robustness_df["round"] = robustness_df.index + 1
        return robustness_df

    # Social Welfare
    def calculate_social_welfare(self, allocation, voting_matrix):
        n = voting_matrix.shape[0]
        total_distance = np.sum(np.linalg.norm(voting_matrix - allocation, ord=1, axis=1))
        return total_distance / n

    def evaluate_social_welfare(self, num_rounds, voting_rules):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in voting_rules:
            results[f'{voting_rule}_social_welfare'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in voting_rules:
                allocation = self.model.allocate_funds(voting_rule)
                social_welfare = self.calculate_social_welfare(allocation, self.model.voting_matrix)
                results[f'{voting_rule}_social_welfare'].append(social_welfare)
        return pd.DataFrame(results)

    def calculate_egalitarian_score(self, allocation, voting_matrix):
        distances = np.linalg.norm(voting_matrix - allocation, ord=1, axis=1)
        return np.max(distances)

    def evaluate_egalitarian_score(self, num_rounds, voting_rules):
        results = {'round': list(range(1, num_rounds + 1))}
        for voting_rule in voting_rules:
            results[f'{voting_rule}_egalitarian_score'] = []
        for round_num in range(num_rounds):
            self.model.step()
            for voting_rule in voting_rules:
                allocation = self.model.allocate_funds(voting_rule)
                egalitarian_score = self.calculate_egalitarian_score(allocation, self.model.voting_matrix)
                results[f'{voting_rule}_egalitarian_score'].append(egalitarian_score)
        return pd.DataFrame(results)