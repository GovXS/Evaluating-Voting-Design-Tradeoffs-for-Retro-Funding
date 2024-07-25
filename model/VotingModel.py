from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np

from agents.ProjectAgent import ProjectAgent
from agents.VoterAgent import VoterAgent
from util.voting_rules import mean_aggregation,median_aggregation,quadratic_aggregation


class VotingModel(Model):
    def __init__(self, num_voters, num_projects, total_op_tokens):
        self.num_voters = num_voters
        self.num_projects = num_projects
        self.total_op_tokens = total_op_tokens
        self.schedule = RandomActivation(self)

        self.voters = [VoterAgent(i, self, num_projects, total_op_tokens) for i in range(num_voters)]
        self.projects = [ProjectAgent(i, self) for i in range(num_projects)]

        for voter in self.voters:
            self.schedule.add(voter)
        for project in self.projects:
            self.schedule.add(project)

        self.voting_matrix = np.zeros((num_voters, num_projects))

    def step(self):
        for i, voter in enumerate(self.voters):
            voter.vote()
            self.voting_matrix[i, :] = voter.votes

    
    def allocate_funds(self, method):
        if method == "mean":
            return self.mean_aggregation()
        elif method == "median":
            return self.median_aggregation()
        elif method == "quadratic":
            return self.quadratic_aggregation()
        else:
            raise ValueError("Unknown aggregation method")

    def mean_aggregation(self):
        funds_allocated=mean_aggregation(self.voting_matrix,self.total_op_tokens,self.num_projects)
        return funds_allocated

    def median_aggregation(self):
        funds_allocated=median_aggregation(self.voting_matrix,self.total_op_tokens,self.num_projects)
        return funds_allocated

    def quadratic_aggregation(self):
        funds_allocated=quadratic_aggregation(self.voting_matrix,self.total_op_tokens,self.num_projects)
        return funds_allocated

    