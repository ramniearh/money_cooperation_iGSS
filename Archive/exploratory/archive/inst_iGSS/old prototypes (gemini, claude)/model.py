import mesa
import random

class EconomicAgent(mesa.Agent):
    """An agent that produces one good and wants to consume others."""
    
    def __init__(self, model, specialty_good):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.inventory = {0: 0, 1: 0, 2: 0} 
        self.inventory[self.specialty_good] = 5 
        self.decision_rule = None 

    def evaluate_rule(self, partner):
        """Uses the GP rule to decide whether to trade."""
        # Get our inventory levels
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        
        # Ask the model's 'brain' what to do
        # We pass my_supply and partner_supply as inputs (ARG0 and ARG1)
        decision_score = self.model.decision_rule(my_supply, partner_supply)
        
        # If the math formula outputs a positive number, we trade!
        return decision_score > 0

    def evaluate_rule(self, partner):
        """Placeholder: randomly decide to trade 50% of the time."""
        return random.choice([True, False])

    def step(self):
        """What the agent does every time 'tick'."""
        self.inventory[self.specialty_good] += 1
        
        other_agents = [a for a in self.model.agents if a != self]
        if not other_agents: return
        partner = random.choice(other_agents)
        
        if self.evaluate_rule(partner):
            self.spot_exchange(partner)

    def spot_exchange(self, partner):
        """The 'Relational Primitive' for trading."""
        if self.inventory[self.specialty_good] > 0 and partner.inventory[partner.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            partner.inventory[partner.specialty_good] -= 1
            self.inventory[partner.specialty_good] += 1

class ExchangeModel(mesa.Model):
    """The macro-environment."""
    
    def __init__(self, N, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule 

        for i in range(self.num_agents):
            specialty = i % 3 
            EconomicAgent(self, specialty)

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        """Returns the total number of unique goods held across all agents."""
        total_diverse_goods = 0
        for agent in self.agents:
            diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
            total_diverse_goods += diverse_goods
        return total_diverse_goods