import numpy as np

def mean_aggregation(voting_matrix, num_voters, total_op_tokens):
    total_votes = np.sum(voting_matrix, axis=0)
    mean_votes = total_votes / num_voters
    return mean_votes / np.sum(mean_votes) * total_op_tokens

def median_aggregation(voting_matrix, total_op_tokens):
    median_votes = np.median(voting_matrix, axis=0)
    return median_votes / np.sum(median_votes) * total_op_tokens

def quadratic_aggregation(voting_matrix, total_op_tokens):
    total_votes = np.sum(voting_matrix, axis=0)
    quadratic_votes = total_votes ** 2
    return quadratic_votes / np.sum(quadratic_votes) * total_op_tokens

def quadratic_median_with_moving_phantoms(votes, k):
        quadratic_votes = np.power(votes, k)
        t_star = find_t_star(quadratic_votes)
        votes_with_phantoms = quadratic_votes + [t_star] * len(quadratic_votes)
        sorted_votes = np.sort(votes_with_phantoms)
        median_index = len(sorted_votes) // 2
        return np.power(sorted_votes[median_index], 1 / k)

def find_t_star(votes):
        # Implement a method to find t* such that the sum of the votes with phantoms equals 1
        total_votes = sum(votes)
        n = len(votes)
        t_star = (1 - total_votes) / n
        return t_star
#############################################################

# Eyal's Input:
def f1_k(t, k, n):
    if 0 <= t <= k / (n + 1):
        return 0
    elif k / (n + 1) < t < (k + 1) / (n + 1):
        return t * (n + 1) - k
    elif (k + 1) / (n + 1) <= t <= 1:
        return 1

def f2_k(t, k, n):
        return min(t*(n - k), 1)
        
def compute_median_with_moving_phantoms(votes,type):
    n, m = votes.shape  
    def median_with_phantoms(t_star):
        if type==1: 
            F = [lambda t, k=k: f1_k(t, k, n) for k in range(n + 1)]
        else:
            F = [lambda t, k=k: f2_k(t, k, n) for k in range(n + 1)]
        median_values = [
                        np.median([F[k](t_star) for k in range(n + 1)] + [votes[i][j] for i in range(n)])
                            for j in range(m)
        ]
        return np.median(median_values)

    def find_t_star():
        low, high = 0.0, 1.0
        epsilon = 0
        while high - low > epsilon:
            mid = (low + high) / 2
            if np.sum([median_with_phantoms(mid) for _ in range(m)]) > 1:
                high = mid
            else:
                low = mid
        return low

    t_star = find_t_star()
    best_distribution = [median_with_phantoms(t_star) for _ in range(m)]
    return best_distribution

def compute_mean_voting(votes):
    return np.mean(profiles, axis=0)
        
def midpoint_rule(votes):
    n, m = votes.shape
    social_welfare = np.zeros(n)
    for i in range(n):
        social_welfare[i] = np.sum(np.sum(np.abs(votes - votes[i]), axis=1))
    best_vote_index = np.argmin(social_welfare)
    return votes[best_vote_index]


