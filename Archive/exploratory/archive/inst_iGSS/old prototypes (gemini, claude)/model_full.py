# =============================================================================
# FILE: model_full.py
# PURPOSE: The Ultimate Crucible - 10 Goods, Spot Exchange, Debt, Scarcity & Freeloaders.
# =============================================================================

import mesa
import random

class EconomicAgent(mesa.Agent):
    
    def __init__(self, model, specialty_good, is_freeloader=False):
        super().__init__(model) 
        self.specialty_good = specialty_good
        
        # --- NEW STATE VARIABLES ---
        # These define the fundamental "nature" and "health" of the agent.
        self.is_freeloader = is_freeloader
        self.is_dead = False
        
        # Metabolism: Everyone starts with 10 days of energy. Max health is 15.
        self.vitality = 10 
        
        self.inventory = {i: 0 for i in range(10)} 
        self.inventory[self.specialty_good] = 5 
        self.ledger = {} 

    def get_balance(self, partner):
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        # THE FREELOADER OVERRIDE:
        # If this agent is a freeloader, they don't even consult the AI brain. 
        # They just say "NO" to giving anything away, forever.
        if self.is_freeloader:
            return False 
            
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        return decision_score > 0

    def step(self):
        # 1. THE REAPER CHECK
        # If you starved to death yesterday, you do absolutely nothing today.
        if self.is_dead:
            return 
            
        # 2. METABOLISM (The ticking clock)
        self.vitality -= 1
        if self.vitality <= 0:
            self.is_dead = True
            return # You die before you can produce or trade today.

        # 3. PRODUCE
        self.inventory[self.specialty_good] += 1
        
        # 4. FIND A PARTNER
        # Notice we added 'not a.is_dead' so agents don't try to trade with corpses.
        other_agents = [a for a in self.model.agents if a != self and not a.is_dead]
        if not other_agents: return
        partner = random.choice(other_agents)
        
        # 5. THE INTERACTION (Double-Blind)
        i_want_to_give = self.evaluate_rule(partner)
        they_want_to_give = partner.evaluate_rule(self)

        # Execute the trade logic (Spot Swap or Credit)
        if i_want_to_give and they_want_to_give:
            self.spot_swap(partner)
        elif i_want_to_give and not they_want_to_give:
            self.give_on_credit(partner)
        elif not i_want_to_give and they_want_to_give:
            partner.give_on_credit(self)

        # 6. NUTRITION (Eating to survive)
        # Look through my inventory for anything that I didn't make myself.
        for good_id, amount in self.inventory.items():
            if good_id != self.specialty_good and amount > 0:
                self.inventory[good_id] -= 1  # Eat 1 unit
                self.vitality = min(15, self.vitality + 3) # Gain 3 energy, cap at 15
                break # Only eat ONE good per day!

    def spot_swap(self, partner):
        # Instant, simultaneous exchange of goods
        if self.inventory[self.specialty_good] > 0 and partner.inventory[partner.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            partner.inventory[partner.specialty_good] -= 1
            self.inventory[partner.specialty_good] += 1

    def give_on_credit(self, partner):
        # Unilateral transfer generating an IOU
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


class FullModel(mesa.Model):
    
    # We now pass 'num_freeloaders' into the creation of the world.
    def __init__(self, N, num_freeloaders, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule 
        
        # Create the Good Guys
        for i in range(self.num_agents - num_freeloaders):
            specialty = i % 10 
            EconomicAgent(self, specialty, is_freeloader=False)
            
        # Create the Sociopaths (Freeloaders)
        for i in range(num_freeloaders):
            specialty = i % 10
            EconomicAgent(self, specialty, is_freeloader=True)

    def step(self):
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        """
        The Macro Target has changed. 
        You only score points for LIVING, COOPERATIVE agents.
        """
        total_fitness = 0
        for agent in self.agents:
            # We don't care if freeloaders survive, they give zero points to the AI
            if not agent.is_dead and not agent.is_freeloader:
                # 10 points just for surviving the famine! 
                # +1 point for every diverse good in their pocket.
                diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
                total_fitness += (10 + diverse_goods)
        return total_fitness