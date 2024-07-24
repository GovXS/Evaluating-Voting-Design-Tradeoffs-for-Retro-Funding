import numpy as np

def random_uniform(num_projects,total_op_tokens):

    votes = np.random.dirichlet(np.ones(num_projects)) * total_op_tokens
    return votes
