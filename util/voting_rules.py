import numpy as np

def mean_aggregation(voting_matrix,total_op_tokens,num_voters):
        total_votes = np.sum(voting_matrix, axis=0)
        mean_votes = total_votes / num_voters
        return mean_votes / np.sum(mean_votes) * total_op_tokens

def median_aggregation(voting_matrix,total_op_tokens,num_voters):
        median_votes = np.median(voting_matrix, axis=0)
        return median_votes / np.sum(median_votes) * total_op_tokens


def quadratic_aggregation(voting_matrix, total_funds, num_projects):
    # Calculate the true vote (square root of the tokens)
    true_vote = np.sqrt(voting_matrix)
    
    # Calculate the vote distribution for each project
    sum_sqrt_tokens_per_project = np.sum(true_vote, axis=0)
    
    # Allocate funds proportionally to the square root of the votes
    funds_allocated = (sum_sqrt_tokens_per_project / np.sum(sum_sqrt_tokens_per_project)) * total_funds
    return funds_allocated

def quadratic_aggregation_round1(voting_matrix,total_op_tokens,num_voters):
        x=np.sqrt(voting_matrix)
        quadratic_votes = np.sum(x, axis=0)
        return quadratic_votes
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


