
import mesa
import random

class EconomicAgent(mesa.Agent):
    """Agent representing a producer/consumer with metabolic needs and a local debt ledger."""
    
    def __init__(self, model, specialty_good, is_freeloader=False):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.is_freeloader = is_freeloader
        self.is_dead = False
        
        self.vitality = 10 # Base metabolism
        self.inventory = {i: 0 for i in range(10)} # 10 goods in the economy
        self.inventory[self.specialty_good] = 5 
        
        self.ledger = {} # Tracks bilateral IOUs (Positive = they owe me)

    def get_balance(self, partner):
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        """Consults the GP-evolved rule to make a trade decision."""
        if self.is_freeloader:
            return False 
            
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        return decision_score > 0

    def step(self):
        if self.is_dead: return 
            
        # 1. Metabolism check
        self.vitality -= 1
        if self.vitality <= 0:
            self.is_dead = True
            return

        # 2. Production
        self.inventory[self.specialty_good] += 1
        
        # 3. Social Interaction (Random mixing)
        other_agents = [a for a in self.model.agents if a != self and not a.is_dead]
        if not other_agents: return
        partner = random.choice(other_agents)
        
        i_want_to_give = self.evaluate_rule(partner)
        they_want_to_give = partner.evaluate_rule(self)

        # 4. Exchange execution
        if i_want_to_give and they_want_to_give:
            self.spot_swap(partner)
        elif i_want_to_give and not they_want_to_give:
            self.give_on_credit(partner)
        elif not i_want_to_give and they_want_to_give:
            partner.give_on_credit(self)

        # 5. Nutrition (Consume one foreign good per tick to restore vitality)
        for good_id, amount in self.inventory.items():
            if good_id != self.specialty_good and amount > 0:
                self.inventory[good_id] -= 1  
                self.vitality = min(15, self.vitality + 3) 
                break 

    def spot_swap(self, partner):
        """Simultaneous exchange. No debt recorded."""
        if self.inventory[self.specialty_good] > 0 and partner.inventory[partner.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            partner.inventory[partner.specialty_good] -= 1
            self.inventory[partner.specialty_good] += 1

    def give_on_credit(self, partner):
        """Unilateral transfer. Generates an IOU."""
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


class ToyModel(mesa.Model):
    """The macro-environment managing agent lifecycle and fitness calculation."""
    
    def __init__(self, N, num_freeloaders, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule 
        
        for i in range(self.num_agents - num_freeloaders):
            EconomicAgent(self, specialty_good=(i % 10), is_freeloader=False)
            
        for i in range(num_freeloaders):
            EconomicAgent(self, specialty_good=(i % 10), is_freeloader=True)

    def step(self):
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        """Scores only living, cooperative agents (Survival + Diversity)."""
        total_fitness = 0
        for agent in self.agents:
            if not agent.is_dead and not agent.is_freeloader:
                diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
                total_fitness += (10 + diverse_goods)
        return total_fitness