import mesa
import random


class EconAgent(mesa.Agent):
    "An agnostic agent that can produce, and sees things"

    def __init__(self, model, production_good):
        super().__init__(model)
        self.production_good = production_good
        self.inventory = {0: 0, 1: 0, 2:0}
        self.inventory[self.production_good] = 2
        self.publicly_visible_variable = {}
    
    def see_other(self, other):   # Helper function
        return other.unique_id
    
    def evaluate(self, partner):  # uses GP to decide whether to act
        my_supply = self.inventory[self.production_good]
        partner_visible_thing = self.see_other(partner)
        
        decision_score = self.model.decision_rule(my_supply, partner_visible_thing)
        
        return decision_score > 0
    
    def step(self):
        for good in self.inventory: # placeholder dirty "consume" function (agents reduce their inventories of everything by 1)
            if self.inventory[good] > 0:
                self.inventory[good] -= 1  # Consume one unit of each good
        self.inventory[self.production_good] += 2  # Produce two units of the good

        other_agents = [a for a in self.model.agents if a != self] # random-match to a partner? pending other topologies e.g. random fixed network
        partner = random.choice(other_agents)

        if self.evaluate(partner):
            self.give(partner)
            pass

    def give(self, partner):
        if self.inventory[self.production_good] > 0:
            # Transfer one unit of the good from self to partner:
            self.inventory[self.production_good] -= 1
            partner.inventory[self.production_good] += 1
            # Later... update publicly visible variable to reflect the transfer (?!?)

class DivisionLaborModel(mesa.Model):
    def __init__(self, N, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule
        self.current_step = 0

        for i in range(self.num_agents):
            specialty = i % 3
            EconAgent(self, specialty)
    
    def step(self):
        self.agents.shuffle_do("step") # Shuffle the order of agents and have them all step
        self.current_step += 1

    def calculate_fitness(self):
        total_diverse_goods = 0
        for agent in self.agents:
            diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
            total_diverse_goods += diverse_goods
        return total_diverse_goods / self.num_agents  # Average number of different goods consumed per agent
    
