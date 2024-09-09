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
    
    def r4_capped_median(self, voting_matrix, total_op_tokens, num_voters):
        num_voters, num_projects = voting_matrix.shape

        # K1 is the maximum number of tokens a single voter can allocate to a single project before redistribution is triggered.
        K1 = 0.0167 * total_op_tokens # 10% of total tokens
        # K2 is the maximum median allocation a project can receive before redistribution is triggered.
        K2 = 0.0167 * total_op_tokens
        # K3 is the minimum allocation required for a project to receive funding; projects below this threshold are eliminated, and their funds are redistributed.
        K3 = 0.000001 * total_op_tokens

        # Step 1: Cap at K1 and redistribute excess
        capped_scores = np.minimum(voting_matrix, K1)
        excess_scores = np.maximum(0, voting_matrix - K1)
        
        redistributed_scores = capped_scores.copy()
        for i in range(num_voters):
            uncapped_projects = capped_scores[i] < K1
            
            # Ensure the shapes match before proceeding
            if np.any(uncapped_projects):
                uncapped_project_indices = np.where(uncapped_projects)[0]
                relevant_capped_scores = capped_scores[i, uncapped_project_indices]
                relevant_excess_scores = excess_scores[i, uncapped_project_indices]
                
                if np.sum(relevant_capped_scores) > 0:
                    proportionate_excess = (relevant_excess_scores * relevant_capped_scores) / np.sum(relevant_capped_scores)
                    redistributed_scores[i, uncapped_project_indices] += proportionate_excess

        # Debugging: Print redistributed scores before calculating medians
        #print("Redistributed Scores:", redistributed_scores)

        # Step 2: Calculate median scores
        median_scores = np.median(redistributed_scores, axis=0)
        #print("Median Scores:", median_scores)

        # Step 3: Cap at K2 and redistribute excess
        capped_median_scores = np.minimum(median_scores, K2)
        excess_median = np.maximum(0, median_scores - K2)
        total_excess_median = np.sum(excess_median)
        
        eligible_projects = capped_median_scores < K2
        redistributed_median_scores = capped_median_scores.copy()
        if np.any(eligible_projects) and np.sum(capped_median_scores[eligible_projects]) > 0:
            redistributed_median_scores[eligible_projects] += (total_excess_median * capped_median_scores[eligible_projects]) / np.sum(capped_median_scores[eligible_projects])

        # Debugging: Print redistributed median scores before normalization
        #print("Redistributed Median Scores:", redistributed_median_scores)

        # Step 4: Normalize to the budget 
        normalized_scores = redistributed_median_scores / np.sum(redistributed_median_scores) * total_op_tokens

        # Step 5: Eliminate projects with allocation less than K3 and redistribute
        final_scores = normalized_scores.copy()
        low_allocation_projects = normalized_scores < K3
        if np.any(low_allocation_projects):
            redistributed_amount = np.sum(normalized_scores[low_allocation_projects])
            eligible_for_redistribution = normalized_scores >= K3
            if np.any(eligible_for_redistribution) and np.sum(final_scores[eligible_for_redistribution]) > 0:
                final_scores[eligible_for_redistribution] += (redistributed_amount * final_scores[eligible_for_redistribution]) / np.sum(final_scores[eligible_for_redistribution])
            final_scores[low_allocation_projects] = 0
        
        # Final normalization
        final_allocation = final_scores / np.sum(final_scores) * total_op_tokens
        # Debugging: Print final allocations
        #print("Final Allocation:", final_allocation)

        return final_allocation
