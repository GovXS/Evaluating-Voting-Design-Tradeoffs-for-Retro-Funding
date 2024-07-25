from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np

from agents.ProjectAgent import ProjectAgent
from agents.VoterAgent import VoterAgent


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
        total_votes = np.sum(self.voting_matrix, axis=0)
        mean_votes = total_votes / self.num_voters
        return mean_votes / np.sum(mean_votes) * self.total_op_tokens

    def median_aggregation(self):
        median_votes = np.median(self.voting_matrix, axis=0)
        return median_votes / np.sum(median_votes) * self.total_op_tokens

    def quadratic_aggregation(self):
        total_votes = np.sum(self.voting_matrix, axis=0)
        quadratic_votes = total_votes ** 2
        return quadratic_votes / np.sum(quadratic_votes) * self.total_op_tokens

    