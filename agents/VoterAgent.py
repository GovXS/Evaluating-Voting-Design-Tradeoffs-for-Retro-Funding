from mesa import Agent
import numpy as np
import pandas as pd
import os
import pandas as pd


class VoterAgent(Agent):
    def __init__(self, unique_id, model, voter_type, num_projects, total_op_tokens):
        super().__init__(unique_id, model)
        self.num_projects = num_projects
        self.total_op_tokens = total_op_tokens
        self.votes = np.zeros(num_projects)
        self.voter_type = voter_type

    
    def vote(self, num_voters):
        if self.voter_type == 'r4_voting_matrix':
            return self.r4_voting_matrix(num_voters, self.num_projects, self.total_op_tokens,)
        elif self.voter_type == 'r1_voting_matrix':
            return self.r1_voting_matrix(num_voters, self.num_projects, self.total_op_tokens,)
        elif self.voter_type == 'rn_model':
            return self.optimized_rn_model(num_voters, self.num_projects, self.total_op_tokens, alpha=2)  # Specify alpha as needed
        elif self.voter_type == 'mallows_model':
            return self.mallows_model_quick(num_voters, self.num_projects, self.total_op_tokens)
        elif self.voter_type == 'euclidean_model':
            return self.euclidean_model(num_voters, self.num_projects, self.total_op_tokens)
        elif self.voter_type == 'multinomial_model':
            return self.multinomial_model(num_voters, self.num_projects, self.total_op_tokens)
        elif self.voter_type == 'random_uniform_model':
            return self.random_uniform_model(num_voters, self.num_projects, self.total_op_tokens)
        else:
            raise ValueError(f"Unknown voter type: {self.voter_type}")

    #n- number of voters
    #m- number of projects
    #K- number of tokens each voter distributes
    #alpha- number of copies returned to the urn. The higher the value of alpha, the stronger the correlation between votes.

    def random_uniform_model(self, n, m, K):
        votes = []
        for _ in range(n):
            vote = np.random.dirichlet(np.ones(m)) * K
            votes.append(vote)
        return votes
    
    def optimized_rn_model(self, n, m, K, alpha):
        urn = [np.random.multinomial(K, [1.0/m] * m) for _ in range(100)]
        votes = []
        for i in range(n):
            chosen_vote = np.random.choice(urn)
            votes.append(chosen_vote)
            urn.extend([chosen_vote] * alpha)
        return votes

    def mallows_model(self, n, m, K, base_vote=None):
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
    
    def mallows_model_quick(self,n, m, K, alpha=0.5):

        base_vote = np.random.dirichlet(np.ones(m), size=1) * K
        votes = []
        for i in range(n):
            noisy_vote = (1 - alpha) * base_vote + alpha * np.random.dirichlet(np.ones(m), size=1) * K
            votes.append(noisy_vote.flatten())
        votes_matrix = np.array(votes)
        return votes_matrix 


    def euclidean_model(self, n, m, K):
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

    def multinomial_model(self, n, m, K):
        votes = []
        for i in range(n):
            probabilities = np.random.dirichlet(np.ones(m))
            vote = np.random.multinomial(K, probabilities)
            votes.append(vote)
        return votes
    
   

    def r4_voting_matrix(self,n,m,K):
        # Get the current file's directory (this script's location)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path to the CSV file
        relative_path = os.path.join(current_dir, '..', 'data','op_voting_matrix', 'r4_voting_matrix.csv')

        # Load the voting matrix from the CSV file
        voting_matrix_df = pd.read_csv(relative_path, index_col=0)

        # Convert the DataFrame to a NumPy array
        voting_matrix = voting_matrix_df.to_numpy()
        voting_matrix=voting_matrix*K

        # Ensure the shape of the matrix matches the expected shape
        # if voting_matrix.shape != (self.num_projects, len(voting_matrix_df.index)):
        #     raise ValueError("Voting matrix shape does not match expected dimensions")

        return voting_matrix
    
    def r1_voting_matrix(self,n,m,K):
        # Get the current file's directory (this script's location)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path to the CSV file
        relative_path = os.path.join(current_dir, '..', 'data', 'op_voting_matrix','r1_voting_matrix.csv')

        # Load the voting matrix from the CSV file
        voting_matrix_df = pd.read_csv(relative_path, index_col=0)

        # Convert the DataFrame to a NumPy array
        voting_matrix = voting_matrix_df.to_numpy()
        voting_matrix=voting_matrix*K

        # Ensure the shape of the matrix matches the expected shape
        # if voting_matrix.shape != (self.num_projects, len(voting_matrix_df.index)):
        #     raise ValueError("Voting matrix shape does not match expected dimensions")

        return voting_matrix
