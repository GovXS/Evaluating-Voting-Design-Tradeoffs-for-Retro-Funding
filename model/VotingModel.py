from mesa import Model
from mesa.time import RandomActivation
import numpy as np
import pandas as pd
from agents.VoterAgent import VoterAgent
from agents.ProjectAgent import ProjectAgent
from model.VotingRules import VotingRules

class VotingModel(Model):
    def __init__(self, voter_type, num_voters, num_projects, total_op_tokens):
        self.num_voters = num_voters
        self.num_projects = num_projects
        self.total_op_tokens = total_op_tokens
        self.schedule = RandomActivation(self)
        self.voter_type = voter_type

        self.voter = VoterAgent(0, self, voter_type, num_projects, total_op_tokens)

        #self.voters = [VoterAgent(i, self, voter_type, num_projects, total_op_tokens) for i in range(num_voters)]
        self.projects = [ProjectAgent(i, self) for i in range(num_projects)]

        #for voter in self.voters:
        #    self.schedule.add(voter)
        for project in self.projects:
            self.schedule.add(project)

        self.voting_matrix = np.zeros((num_voters, num_projects))

        # Initialize the voting rules
        self.voting_rules = self._discover_voting_rules()

    def _discover_voting_rules(self):
        voting_rules = {}
        voting_rules_instance = VotingRules()
        for attr_name in dir(voting_rules_instance):
            attr = getattr(voting_rules_instance, attr_name)
            if callable(attr) and not attr_name.startswith("__"):
                voting_rules[attr_name] = attr
        return voting_rules
    
    def step(self):
        voting_matrix = self.voter.vote(self.num_voters)
        self.voting_matrix = np.array(voting_matrix)
        return self.voting_matrix

    '''
    def step(self):
        for i, voter in enumerate(self.voters):
            voter.vote()
            self.voting_matrix[i, :] = voter.votes
        return self.voting_matrix
    '''
    # The voter generates the entire voting matrix
        
    def run_simulation(self):
        self.step()
        results_df = self.compile_fund_allocations()
        return results_df

    def allocate_funds_1(self, method):
        if method not in self.voting_rules:
            raise ValueError(f"Unknown aggregation method: {method}")
        return self.voting_rules[method](self.voting_matrix, self.total_op_tokens, self.num_voters)
    
    def allocate_funds(self, method, voting_matrix=None):
        """
        Allocate funds using the specified voting rule.

        Parameters:
        - method: The voting rule method to be used (e.g., 'r1_quadratic', 'r2_mean', etc.).
        - voting_matrix: (Optional) A custom voting matrix to use for fund allocation. 
                        If None, the default self.voting_matrix will be used.

        Returns:
        - allocation: The fund allocation according to the voting rule.
        """
        # Use the provided voting_matrix if available, otherwise use the default self.voting_matrix
        if voting_matrix is None:
            voting_matrix = self.voting_matrix
        
        # Check if the method exists in the voting rules
        if method not in self.voting_rules:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        #if voting_matrix.shape != (self.num_voters, self.num_projects):
        #    raise ValueError(f"Invalid voting_matrix shape. Expected {(self.num_voters, self.num_projects)} got {voting_matrix.shape}")


        # Call the appropriate voting rule function with the selected matrix
        return self.voting_rules[method](voting_matrix, self.total_op_tokens, self.num_voters)


    def add_voting_rule(self, name, func):
        self.voting_rules[name] = func

    def remove_voting_rule(self, name):
        if name in self.voting_rules:
            del self.voting_rules[name]

    def compile_fund_allocations(self):
        num_voters, num_projects = self.voting_matrix.shape
        self.num_projects=num_projects
        self.num_voters=num_voters
        allocations = {name: self.allocate_funds(name) for name in self.voting_rules.keys()}
        allocations["Project"] = [f"Project {i+1}" for i in range(self.num_projects)]
        
        results_df = pd.DataFrame(allocations)
        return results_df
