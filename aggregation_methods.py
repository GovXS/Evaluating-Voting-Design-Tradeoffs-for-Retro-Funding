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