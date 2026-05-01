# converted from NetLogo image score model
from mesa.datacollection import DataCollector
import mesa
import random

class ClassicAgent(mesa.Agent):
    def __init__(self, model, strategy):
        super().__init__(model)
        self.strategy = strategy
        self.fitness = 0
        self.score = model.initial_reputations
        self.memory = []
        
        # Initial liquidity distribution
        if model.initial_liquidity < 1.0:
            self.balance = 1 if random.random() < model.initial_liquidity else 0
        else:
            self.balance = int(model.initial_liquidity)

    def cooperate(self, partner):
        self.fitness -= self.model.cost
        partner.fitness += self.model.benefit
        self.score += 1
        self.model.cooperations_this_turn += 1

    def defect(self, partner):
        self.score -= 1
        if self.unique_id not in partner.memory:
            partner.memory.append(self.unique_id)

    def step(self):
        # Pick a random partner
        other_agents = [a for a in self.model.agents if a != self]
        if not other_agents: return
        partner = random.choice(other_agents)

        # Strategy Execution
        if self.strategy == "cooperator":
            self.cooperate(partner)
            
        elif self.strategy == "defector":
            self.defect(partner)
            
        elif self.strategy == "direct":
            if partner.unique_id not in self.memory:
                self.cooperate(partner)
            else:
                self.defect(partner)
                self.memory.remove(partner.unique_id) # Forgive after retaliating
                
        elif self.strategy == "indirect":
            if partner.score > 0:
                self.cooperate(partner)
            else:
                self.defect(partner)
                
        elif self.strategy == "money":
            if partner.balance > 0:
                self.cooperate(partner)
                self.balance += 1
                partner.balance -= 1
            else:
                self.defect(partner)

class ClassicModel(mesa.Model):
    def __init__(self, n_coops, n_defs, n_dirs, n_inds, n_mons, bcr, liq, init_rep):
        super().__init__()
        self.cost = 1
        self.benefit = self.cost * bcr
        self.initial_liquidity = liq
        self.initial_reputations = init_rep
        self.cooperations_this_turn = 0
        
        # Create populations
        strategies = (["cooperator"] * n_coops + ["defector"] * n_defs + 
                      ["direct"] * n_dirs + ["indirect"] * n_inds + ["money"] * n_mons)
        for strat in strategies:
            ClassicAgent(self, strat)

        # ... (after agent creation loop) ...
        self.datacollector = DataCollector(
            model_reporters={
                "Cooperators": lambda m: sum(1 for a in m.agents if a.strategy == "cooperator"),
                "Defectors": lambda m: sum(1 for a in m.agents if a.strategy == "defector"),
                "Direct Reciprocity": lambda m: sum(1 for a in m.agents if a.strategy == "direct"),
                "Indirect Reciprocity": lambda m: sum(1 for a in m.agents if a.strategy == "indirect"),
                "Money Users": lambda m: sum(1 for a in m.agents if a.strategy == "money"),
            }
        )

    def step(self):
        self.cooperations_this_turn = 0
        for agent in self.agents:
            agent.fitness = 0 # Reset at start of round
            
        self.agents.shuffle_do("step")
        
        # End of round baseline update
        for agent in self.agents:
            agent.fitness += 1 
            
        # Evolutionary Updating (Roulette Wheel on 1 random agent)
        agent_to_change = random.choice(self.agents)
        weights = [max(0, a.fitness) for a in self.agents]
        if sum(weights) > 0:
            role_model = random.choices(self.agents, weights=weights, k=1)[0]
            agent_to_change.strategy = role_model.strategy

        self.datacollector.collect(self)