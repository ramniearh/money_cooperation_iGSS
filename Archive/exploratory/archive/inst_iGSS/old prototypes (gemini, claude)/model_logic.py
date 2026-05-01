import mesa
import random

class EconomicAgent(mesa.Agent):
    def __init__(self, model, specialty_good):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.inventory = {0: 0, 1: 0, 2: 0} 
        self.inventory[self.specialty_good] = 5 
        self.ledger = {} 

    def get_balance(self, partner):
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        # We pass the same 3 variables as before
        args = (self.inventory[self.specialty_good], 
                partner.inventory[partner.specialty_good], 
                self.get_balance(partner))
        
        # The rule now returns a number. Positive = True.
        return self.model.decision_rule(*args) > 0

    def step(self):
        self.inventory[self.specialty_good] += 1
        other_agents = [a for a in self.model.agents if a != self]
        if other_agents:
            partner = random.choice(other_agents)
            if self.evaluate_rule(partner):
                self.give_on_credit(partner)

    def give_on_credit(self, partner):
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1

class LogicModel(mesa.Model):
    def __init__(self, N, rule):
        super().__init__()
        self.decision_rule = rule 
        for i in range(N):
            EconomicAgent(self, i % 3)

    def step(self):
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        return sum(sum(1 for amt in a.inventory.values() if amt > 0) for a in self.agents)