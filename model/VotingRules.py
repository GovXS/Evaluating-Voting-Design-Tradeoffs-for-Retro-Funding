import numpy as np

QUORUM = 17
MIN_AMOUNT = 1500

class VotingRules:
    def mean_aggregation(self, voting_matrix, total_op_tokens, num_voters):
        total_votes = np.sum(voting_matrix, axis=0)
        mean_votes = total_votes / num_voters
        return mean_votes / np.sum(mean_votes) * total_op_tokens

    def median_aggregation(self, voting_matrix, total_op_tokens, num_voters):
        median_votes = np.median(voting_matrix, axis=0)
        return median_votes / np.sum(median_votes) * total_op_tokens

    def quadratic_aggregation(self, voting_matrix, total_funds, num_projects):
        true_vote = np.sqrt(voting_matrix)
        sum_sqrt_tokens_per_project = np.sum(true_vote, axis=0)
        funds_allocated = (sum_sqrt_tokens_per_project / np.sum(sum_sqrt_tokens_per_project)) * total_funds
        return funds_allocated

    def quadratic_aggregation_round1(self, voting_matrix, total_op_tokens, num_voters):
        x = np.sqrt(voting_matrix)
        quadratic_votes = np.sum(x, axis=0)
        return quadratic_votes

    def round3_median(self, voting_matrix, total_op_tokens, num_voters):
        # Step 1: Calculate the median, ignoring zeros
        def non_zero_median(column):
            non_zero_values = column[column > 0]
            if len(non_zero_values) == 0:
                return 0
            return np.median(non_zero_values)
        
        median_votes = np.apply_along_axis(non_zero_median, 0, voting_matrix)
        
        # Step 2: Apply eligibility criteria (median >= MIN_AMOUNT)
        
        eligible_projects = median_votes >= MIN_AMOUNT
        eligible_median_votes = median_votes * eligible_projects
        
        # Step 3: Scale the eligible median votes to match the total_op_tokens
        if np.sum(eligible_median_votes) == 0:
            return np.zeros_like(eligible_median_votes)  # Avoid division by zero
        
        scaled_allocations = (eligible_median_votes / np.sum(eligible_median_votes)) * total_op_tokens
        
        return scaled_allocations