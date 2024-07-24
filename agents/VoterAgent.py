from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.voter_behaviour import random_uniform
#from voting_rules import mean_aggregation, median_aggregation,quadratic_aggregation


class VoterAgent(Agent):
    def __init__(self, unique_id, model, num_projects, total_op_tokens):
        super().__init__(unique_id, model)
        self.num_projects = num_projects
        self.total_op_tokens = total_op_tokens
        self.votes = np.zeros(num_projects)

    def vote(self):
        self.votes = random_uniform(self.num_projects,self.total_op_tokens)
