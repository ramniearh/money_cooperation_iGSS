import mesa
import random

class EconomicAgent(mesa.Agent):
    """An agent that produces one good, consumes others, and tracks IOUs."""
    
    def __init__(self, model, specialty_good):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.inventory = {0: 0, 1: 0, 2: 0} 
        self.inventory[self.specialty_good] = 5 
        
        # The Ledger: Tracks balances with other agents by their unique_id
        # Positive balance = They owe me. Negative balance = I owe them.
        self.ledger = {} 

    def get_balance(self, partner):
        """Helper function to check how much a partner owes this agent."""
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        """Uses the GP rule to decide whether to issue credit."""
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        # The AI brain now takes 3 inputs (ARG0, ARG1, ARG2)
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        
        return decision_score > 0

    def step(self):
        """What the agent does every time 'tick'."""
        self.inventory[self.specialty_good] += 1
        
        other_agents = [a for a in self.model.agents if a != self]
        if not other_agents: return
        partner = random.choice(other_agents)
        
        if self.evaluate_rule(partner):
            self.give_on_credit(partner)

    def give_on_credit(self, partner):
        """The 'Debt Primitive'. I give a good, you give me an IOU."""
        if self.inventory[self.specialty_good] > 0:
            # 1. Transfer the good
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            # 2. Update the Ledgers
            # Partner's debt to me goes up by 1
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            # My debt to partner goes down by 1 (I owe them less)
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


class DebtModel(mesa.Model):
    """The macro-environment for the debt scenario."""
    
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