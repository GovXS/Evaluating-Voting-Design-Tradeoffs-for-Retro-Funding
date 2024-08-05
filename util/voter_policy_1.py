import numpy as np
import random
import itertools

import numpy as np

def random_uniform_model(n, num_projects, total_op_tokens):
    votes = []
    for _ in range(n):
        vote = np.random.dirichlet(np.ones(num_projects)) * total_op_tokens
        votes.append(vote)
    return votes

#These are all methods of Genarating artificial voter data and then we have a code for parsing real voter data from pabulib 
#n- number of voters
#m- number of projects
#K- number of tokens each voter distributes
#alpha- number of copies returned to the urn. The higher the value of alpha, the stronger the correlation between votes.
def rn_model(n, m, K, alpha):
    urn = []
    for vote in itertools.product(range(K + 1), repeat=m):
        if sum(vote) == K:
            urn.append(vote)
    
    votes = []
    for i in range(n):
        chosen_vote = random.choice(urn)
        votes.append(chosen_vote)
        urn.extend([chosen_vote] * alpha)
    
    return votes

def optimized_rn_model(n, m, K, alpha):
    # Generate initial random votes
    urn = [np.random.multinomial(K, [1.0/m] * m) for _ in range(100)]
    
    votes = []
    for i in range(n):
        chosen_vote = random.choice(urn)
        votes.append(chosen_vote)
        urn.extend([chosen_vote] * alpha)
    
    return votes


def mallows_model(n, m, K, base_vote=None):
    if base_vote is None:
        base_vote = np.random.multinomial(K, [1.0/m] * m)
    
    votes = [base_vote]
    for i in range(1, n):
        noise = np.random.randint(0, K // 2)
        new_vote = base_vote.copy()
        for _ in range(noise):
            from_proj = np.random.choice(range(m))
            to_proj = np.random.choice(range(m))
            if new_vote[from_proj] > 0:
                new_vote[from_proj] -= 1
                new_vote[to_proj] += 1
        votes.append(new_vote)
    
    return votes

def euclidean_model(n, m, K):
    projects = np.random.rand(m, 2)
    voters = np.random.rand(n, 2)
    
    votes = []
    for voter in voters:
        distances = np.linalg.norm(projects - voter, axis=1)
        inverses = 1 / distances
        total_inverse = np.sum(inverses)
        proportions = inverses / total_inverse
        vote = np.random.multinomial(K, proportions)
        votes.append(vote)
    
    return votes

def multinomial_model(n, m, K):
    votes = []
    for i in range(n):
        probabilities = np.random.dirichlet(np.ones(m))
        vote = np.random.multinomial(K, probabilities)
        votes.append(vote)
    
    return votes


### Parsing from Pabulib- https://pabulib.org/ (choose cumulative in vote type)- for real data from Participatory Budgeting


def parse_meta_section(lines):
    meta = {}
    for line in lines:
        if line.strip() == 'META':
            continue
        if line.strip() == 'PROJECTS':
            break
        key, value = line.split(';')
        meta[key.strip()] = value.strip()
    return meta

def parse_projects_section(lines):
    projects = []
    in_project_section = False
    for line in lines:
        if line.strip() == 'PROJECTS':
            in_project_section = True
            continue
        if in_project_section and line.strip() == 'VOTES':
            break
        if in_project_section:
            parts = line.split(';')
            projects.append(parts[0].strip())
    return projects

def parse_votes_section(lines):
    votes_data = []
    in_votes_section = False
    for line in lines:
        line = line.strip()
        if line == 'VOTES':
            in_votes_section = True
            continue
        if in_votes_section and line.startswith('META'):
            break
        if in_votes_section and line and not line.startswith('voter_id'):
            parts = line.split(';')
            if len(parts) >= 3:
                voter_id = parts[0].strip()
                projects = list(map(int, parts[1].strip().split(',')))
                points = list(map(int, parts[2].strip().split(',')))
                votes_data.append((voter_id, projects, points))
    return votes_data

def create_votes_matrix(votes_data, num_projects):
    num_voters = len(votes_data)
    votes_matrix = np.zeros((num_voters, num_projects), dtype=int)
    for voter_index, (voter_id, projects, points) in enumerate(votes_data):
        for project, point in zip(projects, points):
            votes_matrix[voter_index, project-1] = point 
    return votes_matrix

def read_from_pabulib_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    meta = parse_meta_section(lines)
    projects = parse_projects_section(lines)
    votes_data = parse_votes_section(lines)
    
    num_projects = int(meta['num_projects'])
    return create_votes_matrix(votes_data, num_projects)



