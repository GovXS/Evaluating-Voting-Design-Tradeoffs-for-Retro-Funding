import numpy as np

def median_with_moving_phantoms(votes, t):
        votes_with_phantoms = votes + [t] * len(votes)
        sorted_votes = np.sort(votes_with_phantoms)
        median_index = len(sorted_votes) // 2
        return sorted_votes[median_index]

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
#To my understanding the quadratic voting is not supposed to effect the aggrigation only the cost it self (bigger votes will be more expensive than smaller ones)
# This is why Quadratic should just be a parameter in the simulation
# I write some implementation here

from scipy.optimize import brentq
def f_k(t, k, n):
    if 0 <= t <= k / (n + 1):
        return 0
    elif k / (n + 1) < t < (k + 1) / (n + 1):
        return t * (n + 1) - k
    elif (k + 1) / (n + 1) <= t <= 1:
        return 1

def compute_median_with_moving_phantoms(F, profiles):
    n = len(F) - 1  # number of functions in F, assuming indexing from 0 to n
    m = len(profiles)  # number of profiles
    def median_with_phantoms(t_star):
        median_values = [np.median([F[k](t_star) for k in range(n + 1)] + [profiles[i][j] for i in range(m)]) for j in range(m)]
        return np.median(median_values)

    def find_t_star(): #this is simple binary search, we can use somthing better i guess-This can be changed to find somthing quicker That I add at the end of the script 
        low, high = 0.0, 1.0
        while high - low > 1e-6:
            mid = (low + high) / 2
            if np.sum([median_with_phantoms(mid) for _ in range(m)]) > 1:
                high = mid
            else:
                low = mid
        return low
    
    # Use Brent's method to find the root (t_star)
    t_star = find_t_star()
    
    return median_with_phantoms(t_star)

def create_phantom_system(n):
    return [lambda t: f_k(t, k, n) for k in range(n + 1)]

# Example usage:
# Define the phantom system F with piecewise functions as described
votes = np.array([
    [0.2, 0.3, 0.5],
    [0.3, 0.1, 0.6],
    [0.1, 0.5, 0.4],
    [0.0, 0.15, 0.85]
])
F = create_phantom_system(3)
medians = []
for j in range(votes.shape[1]):
    profiles = votes[:, j]
    median_value = compute_median_with_moving_phantoms(F, profiles)
    medians.append(median_value)

print("Computed Medians for each Project:", medians)

#this is trivial but I guess should appear here also 
def compute_mean_voting(profiles):
    return np.mean(profiles, axis=0)

### This is probably better for finding t*
def median_with_phantoms(t_star):
        median_values = [
            np.median([F[k](t_star) for k in range(n + 1)] + [profiles[i][j] for i in range(m)]) 
            for j in range(m)
        ]
        return np.median(median_values)
def objective(t_star):# ti
        return np.sum([median_with_phantoms(t_star) for _ in range(m)]) - 1
t_star = brentq(objective, 0.0, 1.0)
