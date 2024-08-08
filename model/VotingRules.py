import numpy as np

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

   