import mesa
import random

class EconomicAgent(mesa.Agent):
    """An agent that can either be a cooperative producer or a selfish freeloader."""
    
    def __init__(self, model, specialty_good, is_freeloader=False):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.is_freeloader = is_freeloader
        
        self.inventory = {0: 0, 1: 0, 2: 0} 
        self.inventory[self.specialty_good] = 5 
        
        # Positive = They owe me. Negative = I owe them.
        self.ledger = {} 

    def get_balance(self, partner):
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        """Freeloaders ALWAYS say no. Producers use the AI's rule."""
        if self.is_freeloader:
            return False # The Freeloader's selfish brain
            
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        return decision_score > 0

    def step(self):
        # Everyone produces (so freeloaders look normal from the outside!)
        self.inventory[self.specialty_good] += 1
        
        other_agents = [a for a in self.model.agents if a != self]
        if not other_agents: return
        partner = random.choice(other_agents)
        
        if self.evaluate_rule(partner):
            self.give_on_credit(partner)

    def give_on_credit(self, partner):
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


class FreeloaderModel(mesa.Model):
    """The macro-environment with social risk."""
    
    def __init__(self, N, num_freeloaders, rule):
        super().__init__()
        self.decision_rule = rule 
        
        # Add normal agents
        for i in range(N - num_freeloaders):
            specialty = i % 3 
            EconomicAgent(self, specialty, is_freeloader=False)
            
        # Add the Freeloaders
        for i in range(num_freeloaders):
            specialty = i % 3
            EconomicAgent(self, specialty, is_freeloader=True)

    def step(self):
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        """
        Calculates welfare ONLY for the cooperative producers.
        Goods trapped with Freeloaders are considered 'wasted'.
        """
        total_diverse_goods = 0
        for agent in self.agents:
            # We only score based on the good guys
            if not agent.is_freeloader: 
                diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
                total_diverse_goods += diverse_goods
        return total_diverse_goods