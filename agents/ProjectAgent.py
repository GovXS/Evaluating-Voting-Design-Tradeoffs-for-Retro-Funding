from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from voter_behaviour import random_uniform
#from voting_rules import mean_aggregation, median_aggregation,quadratic_aggregation


class ProjectAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.total_votes = 0
        self.funds_allocated = 0

    def add_votes(self, votes):
        self.total_votes += votes
