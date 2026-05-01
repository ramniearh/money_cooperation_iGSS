import mesa
import random

class EconomicAgent(mesa.Agent):
    """An agent that must consume foreign goods to survive."""
    
    def __init__(self, model, specialty_good):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.is_dead = False
        
        # New Scarcity Mechanic
        self.vitality = 10 
        
        self.inventory = {0: 0, 1: 0, 2: 0} 
        self.inventory[self.specialty_good] = 5 
        
        # The Ledger
        self.ledger = {} 

    def get_balance(self, partner):
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        """Uses the GP rule to decide whether to issue credit."""
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        return decision_score > 0

    def step(self):
        # 1. Death Check
        if self.is_dead:
            return 
            
        # 2. Metabolism (Burn energy to live and produce)
        self.vitality -= 1
        if self.vitality <= 0:
            self.is_dead = True
            return

        self.inventory[self.specialty_good] += 1
        
        # 3. Social Interaction
        other_agents = [a for a in self.model.agents if a != self and not a.is_dead]
        if other_agents:
            partner = random.choice(other_agents)
            if self.evaluate_rule(partner):
                self.give_on_credit(partner)

        # 4. Nutrition (Eat foreign goods to regain vitality)
        for good_id, amount in self.inventory.items():
            if good_id != self.specialty_good and amount > 0:
                self.inventory[good_id] -= 1  # Eat 1 foreign good
                self.vitality = min(15, self.vitality + 3) # Max health is 15
                break # Only eat 1 good per tick

    def give_on_credit(self, partner):
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


class ScarcityModel(mesa.Model):
    """The macro-environment where agents can starve."""
    
    def __init__(self, N, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule 
        
        for i in range(self.num_agents):
            specialty = i % 3 
            EconomicAgent(self, specialty)

    def step(self):
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        """
        Fitness is heavily weighted by survival!
        A dead agent provides 0 fitness.
        """
        total_fitness = 0
        for agent in self.agents:
            if not agent.is_dead:
                # 10 points just for surviving, plus points for diverse goods
                diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
                total_fitness += (10 + diverse_goods)
        return total_fitness