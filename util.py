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
    