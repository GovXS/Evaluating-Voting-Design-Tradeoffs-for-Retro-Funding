import numpy as np
import numpy as np
import pandas as pd
import os
import sys

# Add the directory containing the VotingModel to the Python path
sys.path.append(os.path.abspath('/Users/idrees/Code/govxs/'))
from util.voting_rules import mean_aggregation, median_aggregation, quadratic_aggregation
from model.VotingModel import VotingModel

def utility(voter_preferences, outcome):
    """Calculate the utility of a voter given their preferences and the outcome."""
    return -np.sum(np.abs(voter_preferences - outcome))

def l1_distance(x, xi):
    """Calculate the L1 distance between two vectors."""
    return np.sum(np.abs(x - xi))

def calculate_robustness(model):
    distances = []
    for method in ["mean", "median", "quadratic"]:
        original_outcome = model.allocate_funds(method)
        voter_idx = np.random.randint(0, model.num_voters)
        original_vote = model.voting_matrix[voter_idx].copy()
        new_vote = np.random.rand(model.num_projects)
        new_vote /= new_vote.sum()
        new_voting_matrix = model.voting_matrix.copy()
        new_voting_matrix[voter_idx] = new_vote
        model.voting_matrix = new_voting_matrix
        new_outcome = model.allocate_funds(method)
        distance = np.linalg.norm(original_outcome - new_outcome, ord=1)
        distances.append(distance)
        model.voting_matrix[voter_idx] = original_vote
    return np.mean(distances)

def calculate_social_welfare(model):
    social_welfare_results = []
    for method in ["mean", "median", "quadratic"]:
        outcome = model.allocate_funds(method)
        total_distance = 0
        for i in range(model.num_voters):
            total_distance += np.sum(np.abs(outcome - model.voting_matrix[i]))
        average_distance = total_distance / model.num_voters
        social_welfare_results.append(average_distance)
    return np.mean(social_welfare_results)

def calculate_gini_index(model):
    gini_results = []
    for method in ["mean", "median", "quadratic"]:
        allocation = model.allocate_funds(method)
        m = len(allocation)
        if m == 0:
            return 0
        allocation_sorted = np.sort(allocation)
        cumulative_allocation = np.cumsum(allocation_sorted)
        numerator = 2 * np.sum((np.arange(1, m + 1) - 1) * allocation_sorted) - m * cumulative_allocation[-1]
        denominator = m * cumulative_allocation[-1]
        gini_results.append(numerator / denominator)
    return np.mean(gini_results)

def calculate_group_strategyproofness(model, coalition_size=3):
    group_strategyproof = True
    for method in ["mean", "median", "quadratic"]:
        truthfully_voted_outcome = model.allocate_funds(method)
        for _ in range(100):  # Number of random coalitions to test
            coalition = np.random.choice(model.num_voters, coalition_size, replace=False)
            original_utilities = [utility(model.voting_matrix[i], truthfully_voted_outcome) for i in coalition]
            strategic_voting_matrix = model.voting_matrix.copy()
            for i in coalition:
                strategic_voting_matrix[i] = np.random.rand(model.num_projects)
            model.voting_matrix = strategic_voting_matrix
            strategically_voted_outcome = model.allocate_funds(method)
            new_utilities = [utility(model.voting_matrix[i], strategically_voted_outcome) for i in coalition]
            if all(new_utilities[i] > original_utilities[i] for i in range(coalition_size)):
                group_strategyproof = False
                break
            model.voting_matrix = strategic_voting_matrix
    return group_strategyproof

def calculate_alignment_with_ground_truth(model, true_values):
    alignment_results = []
    for method in ["mean", "median", "quadratic"]:
        outcome = model.allocate_funds(method)
        hamming_dist = np.sum(true_values != outcome)
        alignment_results.append(hamming_dist)
    return np.mean(alignment_results)

