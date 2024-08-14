import numpy as np

QUORUM = 17


class VotingRules:

    def r1_quadratic(self, voting_matrix, total_funds, num_voters):
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
        MIN_AMOUNT = 0
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
            def f_k(t, k, num_voters):
                if t <= k / (num_voters + 1):
                    return 0
                elif t < (k + 1) / (num_voters + 1):
                    return (num_voters + 1) * t - k
                else:
                    return 1
    
            def median_with_phantoms(t_star, j):
                phantom_votes = [f_k(t_star, k, num_voters) for k in range(num_voters + 1)]
                real_votes = voting_matrix[:, j] / sum(voting_matrix[0]) 
                return np.median(phantom_votes + list(real_votes))
    
            def find_t_star():
                low, high = 0.0, 1.0
                epsilon = 1e-9
                while high - low > epsilon:
                    mid = (low + high) / 2
                    if sum(median_with_phantoms(mid, j) for j in range(m)) > 1:
                        high = mid
                    else:
                        low = mid
                return low
                
            num_voters, m = voting_matrix.shape
            t_star = find_t_star()
            distribution = np.array([median_with_phantoms(t_star, j) for j in range(m)])
            best_distribution = distribution * (total_op_tokens / np.sum(distribution))
            return best_distribution
        
        