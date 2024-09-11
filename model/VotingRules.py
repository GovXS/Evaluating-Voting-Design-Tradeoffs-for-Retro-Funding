import numpy as np

QUORUM = 17


class VotingRules:

    def r1_quadratic(self, voting_matrix, total_funds, num_voters):
        num_voters, num_projects = voting_matrix.shape
        true_vote = np.sqrt(voting_matrix)
        sum_sqrt_tokens_per_project = np.sum(true_vote, axis=0)
        funds_allocated = (sum_sqrt_tokens_per_project / np.sum(sum_sqrt_tokens_per_project)) * total_funds
        return funds_allocated

    def r2_mean(self, voting_matrix, total_op_tokens, num_voters):
        num_voters, num_projects = voting_matrix.shape
        total_votes = np.sum(voting_matrix, axis=0)
        mean_votes = total_votes / num_voters
        return mean_votes / np.sum(mean_votes) * total_op_tokens

    def r3_median(self, voting_matrix, total_op_tokens, num_voters):
        num_voters, num_projects = voting_matrix.shape
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
            num_voters, num_projects = voting_matrix.shape
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
    
    def r4_capped_median(self,voting_matrix, total_op_tokens, num_voters):
        num_voters, num_projects = voting_matrix.shape

        # K1 is the maximum number of tokens a single voter can allocate to a single project before redistribution is triggered.
        K1 = 0.05*total_op_tokens
        # K2 is the maximum median allocation a project can receive before redistribution is triggered.
        K2 =  0.05*total_op_tokens
        # K3 is the minimum allocation required for a project to receive funding; projects below this threshold are eliminated, and their funds are redistributed.
        K3 = 0.0001*total_op_tokens

        # Step 1: Cap at K1 for each voter
        capped_scores = np.minimum(voting_matrix, K1)
        excess_scores = np.maximum(0, voting_matrix - K1)

        redistributed_scores = capped_scores.copy()
        
        for i in range(num_voters):
            uncapped_projects = capped_scores[i] < K1
            
            if np.any(uncapped_projects):
                uncapped_project_indices = np.where(uncapped_projects)[0]
                relevant_capped_scores = capped_scores[i, uncapped_project_indices]
                relevant_excess_scores = excess_scores[i, uncapped_project_indices]
                
                if np.sum(relevant_capped_scores) > 0:
                    proportionate_excess = (relevant_excess_scores * relevant_capped_scores) / np.sum(relevant_capped_scores)
                    redistributed_scores[i, uncapped_project_indices] += proportionate_excess

        # Step 2: Calculate medians
        median_scores = np.median(redistributed_scores, axis=0)

        # Step 3: Cap at K2 and redistribute
        capped_median_scores = np.minimum(median_scores, K2)
        excess_median = np.maximum(0, median_scores - K2)

        # Total excess after capping at K2
        total_excess_median = np.sum(excess_median)

        # Step 4: Redistribution of excess from K2 to only projects below K2
        eligible_projects = capped_median_scores < K2
        redistributed_median_scores = capped_median_scores.copy()

        while total_excess_median > 0:
            eligible_for_redistribution = capped_median_scores < K2  # Only redistribute to projects under K2
            
            if np.any(eligible_for_redistribution) and np.sum(capped_median_scores[eligible_for_redistribution]) > 0:
                # Redistribute excess only to eligible projects
                redistributed_median_scores[eligible_for_redistribution] += (
                    (total_excess_median * capped_median_scores[eligible_for_redistribution]) 
                    / np.sum(capped_median_scores[eligible_for_redistribution])
                )

            # Recalculate excess after redistribution
            total_excess_median = np.sum(np.maximum(0, redistributed_median_scores - K2))
            redistributed_median_scores = np.minimum(redistributed_median_scores, K2)  # Cap at K2 again

        # Step 5: Normalize to the total budget
        final_scores = redistributed_median_scores / np.sum(redistributed_median_scores) * total_op_tokens

        # Step 6: Eliminate projects with allocation below K3 and redistribute
        low_allocation_projects = final_scores < K3
        if np.any(low_allocation_projects):
            redistributed_amount = np.sum(final_scores[low_allocation_projects])
            eligible_for_redistribution = (final_scores >= K3) & (final_scores < K2)  # Only redistribute to projects below K2
            if np.any(eligible_for_redistribution) and np.sum(final_scores[eligible_for_redistribution]) > 0:
                final_scores[eligible_for_redistribution] += (redistributed_amount * final_scores[eligible_for_redistribution]) / np.sum(final_scores[eligible_for_redistribution])
            final_scores[low_allocation_projects] = 0

        # Step 7: Final normalization and capping
        total_allocated = np.sum(final_scores)
        if total_allocated > 0:
            final_allocation = final_scores / total_allocated * total_op_tokens
        else:
            final_allocation = final_scores

        return np.minimum(final_allocation, K2)  # Ensure final capping


