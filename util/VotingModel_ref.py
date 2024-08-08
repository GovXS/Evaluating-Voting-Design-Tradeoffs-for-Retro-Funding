from mesa import Model
from mesa.time import RandomActivation
import numpy as np
import pandas as pd
from agents.VoterAgent import VoterAgent
from agents.ProjectAgent import ProjectAgent
from model.VotingRules import mean_aggregation, median_aggregation, quadratic_aggregation

class VotingModel(Model):
    def __init__(self, voter_type, num_voters, num_projects, total_op_tokens):
        self.num_voters = num_voters
        self.num_projects = num_projects
        self.total_op_tokens = total_op_tokens
        self.schedule = RandomActivation(self)
        self.voter_type = voter_type

        self.voters = [VoterAgent(i, self, voter_type, num_projects, total_op_tokens) for i in range(num_voters)]
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
        return self.voting_matrix

    def run_simulation(self):
        self.step()
        results_df = self.compile_fund_allocations()
        return results_df

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
        funds_allocated = mean_aggregation(self.voting_matrix, self.total_op_tokens, self.num_projects)
        return funds_allocated

    def median_aggregation(self):
        funds_allocated = median_aggregation(self.voting_matrix, self.total_op_tokens, self.num_projects)
        return funds_allocated

    def quadratic_aggregation(self):
        funds_allocated = quadratic_aggregation(self.voting_matrix, self.total_op_tokens, self.num_projects)
        return funds_allocated
    
    def compile_fund_allocations(self):
        mean_allocations = self.allocate_funds("mean")
        median_allocations = self.allocate_funds("median")
        quadratic_allocations = self.allocate_funds("quadratic")

        results_df = pd.DataFrame({
            "Project": [f"Project {i+1}" for i in range(self.num_projects)],
            "Mean Aggregation": mean_allocations,
            "Median Aggregation": median_allocations,
            "Quadratic Aggregation": quadratic_allocations
        })

        return results_df
