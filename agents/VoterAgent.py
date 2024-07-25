from mesa import Agent, Model
import numpy as np



class VoterAgent(Agent):
    def __init__(self, unique_id, model, num_projects, total_op_tokens):
        super().__init__(unique_id, model)
        self.num_projects = num_projects
        self.total_op_tokens = total_op_tokens
        self.votes = np.zeros(num_projects)

    #Voter behaviour policy function, just use disichlet distribution for now but latter will add othe methods of modleling voter behavipur here
    
    def vote(self):
        self.votes = np.random.dirichlet(np.ones(self.num_projects)) * self.total_op_tokens
