import mesa
import random

class EconomicAgent(mesa.Agent):
    """An agent facing both Scarcity and Social Risk."""
    
    def __init__(self, model, specialty_good, is_freeloader=False):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.is_freeloader = is_freeloader
        self.is_dead = False
        
        self.vitality = 10 
        self.inventory = {0: 0, 1: 0, 2: 0} 
        self.inventory[self.specialty_good] = 5 
        
        self.ledger = {} 

    def get_balance(self, partner):
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        """Freeloaders say no. Producers use the AI's rule."""
        if self.is_freeloader:
            return False 
            
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        return decision_score > 0

    def step(self):
        # 1. Death Check
        if self.is_dead:
            return 
            
        # 2. Metabolism
        self.vitality -= 1
        if self.vitality <= 0:
            self.is_dead = True
            return

        self.inventory[self.specialty_good] += 1
        
        # 3. Social Interaction (Only with living agents)
        other_agents = [a for a in self.model.agents if a != self and not a.is_dead]
        if other_agents:
            partner = random.choice(other_agents)
            if self.evaluate_rule(partner):
                self.give_on_credit(partner)

        # 4. Nutrition
        for good_id, amount in self.inventory.items():
            if good_id != self.specialty_good and amount > 0:
                self.inventory[good_id] -= 1 
                self.vitality = min(15, self.vitality + 3) 
                break 

    def give_on_credit(self, partner):
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


class CombinedModel(mesa.Model):
    """The macro-environment testing true financial institutions."""
    
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
        Fitness strictly rewards the survival and wealth of COOPERATIVE agents.
        A dead agent = 0. A freeloader = 0. 
        """
        total_fitness = 0
        for agent in self.agents:
            # Only score living, cooperative agents
            if not agent.is_dead and not agent.is_freeloader:
                # 10 points for surviving, +1 per diverse good
                diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
                total_fitness += (10 + diverse_goods)
        return total_fitness