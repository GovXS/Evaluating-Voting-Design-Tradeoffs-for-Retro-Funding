import numpy as np

QUORUM = 17
MIN_AMOUNT = 0

class VotingRules:

    def r1_quadratic(self, voting_matrix, total_funds, num_projects):
        true_vote = np.sqrt(voting_matrix)
        sum_sqrt_tokens_per_project = np.sum(true_vote, axis=0)
        funds_allocated = (sum_sqrt_tokens_per_project / np.sum(sum_sqrt_tokens_per_project)) * total_funds
        return funds_allocated

    def r2_mean(self, voting_matrix, total_op_tokens, num_voters):
        total_votes = np.sum(voting_matrix, axis=0)
        mean_votes = total_votes / num_voters
        return mean_votes / np.sum(mean_votes) * total_op_tokens

    def r3_median(self, voting_matrix, total_op_tokens, num_voters):
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

    def majoritarian_moving_phantoms(self, voting_matrix, total_op_tokens, num_voters):
        def f1_k(t, k, n):
            if 0 <= t <= k / (n + 1):
                return 0
            elif k / (n + 1) < t < (k + 1) / (n + 1):
                return t * (n + 1) - k
            elif (k + 1) / (n + 1) <= t <= 1:
                return 1

        def median_with_phantoms(t_star):
            F = [lambda t, k=k: f1_k(t, k, n) for k in range(n + 1)]
            median_values = [
                np.median([F[k](t_star) for k in range(n + 1)] + [votes[i][j] for i in range(n)])
                for j in range(m)
            ]
            return np.median(median_values)

        def find_t_star():
            low, high = 0.0, 1.0
            epsilon = 1e-9  
            while high - low > epsilon:
                mid = (low + high) / 2
                if np.sum([median_with_phantoms(mid) for j in range(m)]) > 1:
                    high = mid
                else:
                    low = mid
            return low

        n, m = voting_matrix.shape
        row_sums = voting_matrix.sum(axis=1, keepdims=True)
        votes = voting_matrix / row_sums
        t_star = find_t_star()
        
        distribution = np.array([median_with_phantoms(t_star) for j in range(m)])
        best_distribution = distribution * (total_op_tokens / np.sum(distribution))
        
        return best_distribution
        