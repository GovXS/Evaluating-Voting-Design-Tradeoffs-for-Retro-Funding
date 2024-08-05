from mesa import Agent
import numpy as np
from util.voter_policy import random_uniform_model, rn_model, mallows_model, euclidean_model, multinomial_model, optimized_rn_model

class VoterAgent(Agent):
    def __init__(self, unique_id, model, voter_type, num_projects, total_op_tokens):
        super().__init__(unique_id, model)
        self.num_projects = num_projects
        self.total_op_tokens = total_op_tokens
        self.votes = np.zeros(num_projects)
        self.voter_type = voter_type

    def vote(self):
        if self.voter_type == 'rn_model':
            self.votes = optimized_rn_model(self.num_projects, self.total_op_tokens, alpha=2)  # Specify alpha as needed
        elif self.voter_type == 'mallows_model':
            self.votes = mallows_model(self.num_projects, self.total_op_tokens)
        elif self.voter_type == 'euclidean_model':
            self.votes = euclidean_model(self.num_projects, self.total_op_tokens)
        elif self.voter_type == 'multinomial_model':
            self.votes = multinomial_model(self.num_projects, self.total_op_tokens)
        elif self.voter_type == 'random_uniform_model':
            self.votes = random_uniform_model(self.num_projects, self.total_op_tokens)
        else:
            raise ValueError(f"Unknown voter type: {self.voter_type}")
