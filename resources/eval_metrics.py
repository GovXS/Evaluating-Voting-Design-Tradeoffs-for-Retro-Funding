import numpy as np
from resources.voting_model import Voter

def evaluate_resistance_to_malicious_behavior(simulation, method, num_tests=100):
    results = []
    for _ in range(num_tests):
        simulation.simulate_voting()
        original_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        # Introduce a malicious voter who votes strategically
        malicious_voter = Voter(voter_id=-1, op_available=simulation.round.max_funding, laziness=0, expertise=1)
        malicious_voter.cast_vote(simulation.round.projects[0], malicious_voter.total_op)
        simulation.round.voters.append(malicious_voter)
        
        malicious_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        difference = np.abs(np.array(original_allocations) - np.array(malicious_allocations)).sum()
        results.append(difference)
        
        simulation.reset_round()
    
    avg_difference = np.mean(results)
    return avg_difference


def evaluate_resistance_to_collusion(simulation, method, num_tests=100, collusion_size=10):
    results = []
    for _ in range(num_tests):
        simulation.simulate_voting()
        original_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        # Introduce a group of malicious voters who vote strategically
        malicious_voters = [
            Voter(voter_id=-(i+1), op_available=simulation.round.max_funding / collusion_size, laziness=0, expertise=1)
            for i in range(collusion_size)
        ]
        for mv in malicious_voters:
            mv.cast_vote(simulation.round.projects[0], mv.total_op)
            simulation.round.voters.append(mv)
        
        collusion_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        difference = np.abs(np.array(original_allocations) - np.array(collusion_allocations)).sum()
        results.append(difference)
        
        simulation.reset_round()
    
    avg_difference = np.mean(results)
    return avg_difference


def evaluate_incentive_compatibility(simulation, method, num_tests=100):
    results = []
    for _ in range(num_tests):
        simulation.simulate_voting()
        original_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        # Simulate honest voting
        for voter in simulation.round.voters:
            voter.reset_voter()
            for project in simulation.round.projects:
                amount = np.random.uniform(0, voter.balance_op)
                voter.cast_vote(project, amount)
        
        honest_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        difference = np.abs(np.array(original_allocations) - np.array(honest_allocations)).sum()
        results.append(difference)
        
        simulation.reset_round()
    
    avg_difference = np.mean(results)
    return avg_difference


def evaluate_simplicity(simulation, method):
    # Implement logic to evaluate simplicity
    pass

def evaluate_robustness(simulation, method, num_tests=100):
    results = []
    for _ in range(num_tests):
        simulation.simulate_voting()
        original_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        for voter in np.random.choice(simulation.round.voters, size=int(0.1 * simulation.round.num_voters), replace=False):
            voter.reset_voter()
            for project in np.random.choice(simulation.round.projects, size=int(0.5 * simulation.round.num_projects), replace=False):
                amount = np.random.uniform(0, voter.balance_op)
                voter.cast_vote(project, amount)
        
        modified_allocations = simulation.round.calculate_allocations(method, quorum=1, min_amount=1)
        
        difference = np.abs(np.array(original_allocations) - np.array(modified_allocations)).sum()
        results.append(difference)
        
        simulation.reset_round()
    
    avg_difference = np.mean(results)
    return avg_difference

def evaluate_representativeness(simulation, method):
    # Implement logic to evaluate representativeness
    pass

def evaluate_diversity(simulation, method):
    # Implement logic to evaluate diversity of results
    pass
