import numpy as np
import random
import itertools

def random_uniform_model(num_projects, total_op_tokens):
    vote = np.random.dirichlet(np.ones(num_projects)) * total_op_tokens
    return vote

def rn_model(num_projects, total_op_tokens, alpha):
    urn = []
    for vote in itertools.product(range(total_op_tokens + 1), repeat=num_projects):
        if sum(vote) == total_op_tokens:
            urn.append(vote)
    
    chosen_vote = random.choice(urn)
    urn.extend([chosen_vote] * alpha)
    return chosen_vote

def optimized_rn_model(num_projects, total_op_tokens, alpha):
    urn = [np.random.multinomial(total_op_tokens, [1.0/num_projects] * num_projects) for _ in range(100)]
    chosen_vote = random.choice(urn)
    urn.extend([chosen_vote] * alpha)
    return chosen_vote

def mallows_model(num_projects, total_op_tokens):
    base_vote = np.random.multinomial(total_op_tokens, [1.0/num_projects] * num_projects)
    noise = np.random.randint(0, total_op_tokens // 2)
    new_vote = base_vote.copy()
    for _ in range(noise):
        from_proj = np.random.choice(range(num_projects))
        to_proj = np.random.choice(range(num_projects))
        if new_vote[from_proj] > 0:
            new_vote[from_proj] -= 1
            new_vote[to_proj] += 1
    return new_vote

def euclidean_model(num_projects, total_op_tokens):
    projects = np.random.rand(num_projects, 2)
    voter = np.random.rand(1, 2)
    distances = np.linalg.norm(projects - voter, axis=1)
    inverses = 1 / distances
    total_inverse = np.sum(inverses)
    proportions = inverses / total_inverse
    vote = np.random.multinomial(total_op_tokens, proportions)
    return vote

def multinomial_model(num_projects, total_op_tokens):
    probabilities = np.random.dirichlet(np.ones(num_projects))
    vote = np.random.multinomial(total_op_tokens, probabilities)
    return vote
