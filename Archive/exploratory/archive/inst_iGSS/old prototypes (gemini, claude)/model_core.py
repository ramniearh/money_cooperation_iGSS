# =============================================================================
# FILE 1: model_core.py
# PURPOSE: This defines the rules of the universe, the agents, and the economy.
# =============================================================================

import mesa
import random

# -----------------------------------------------------------------------------
# CLASS: EconomicAgent
# OOP LESSON: A "Class" is a blueprint. It's not a real person yet, just the 
# instructions on how to build one. 
# "mesa.Agent" means we are inheriting all the basic agent abilities from Mesa 
# (like the ability to have a unique ID and belong to a model).
# -----------------------------------------------------------------------------
class EconomicAgent(mesa.Agent):
    
    # __init__ is the "Constructor". It runs once the exact moment an agent is born.
    # OOP LESSON: "self" refers to the specific agent being built right now. 
    # If the Class is a blueprint for a house, "self" is the actual address of the house.
    def __init__(self, model, specialty_good):
        super().__init__(model) # This tells Mesa to do its own background setup
        
        # EXTENSION IDEA: Right now, agents only have 1 specialty. 
        # We could make 'specialty_good' a list, so agents produce multiple things!
        self.specialty_good = specialty_good
        
        # We now have 10 goods in the economy (numbered 0 through 9).
        # We create a dictionary (a ledger) where every good starts at 0.
        self.inventory = {i: 0 for i in range(10)} 
        
        # Give them a starting bonus of their own good so they aren't broke on Day 1.
        self.inventory[self.specialty_good] = 5 
        
        # The Debt Ledger. It starts empty.
        # If I give you a good and get nothing back, your score goes up to +1.
        self.ledger = {} 

    # A "Method" (a function that belongs to this specific agent)
    def get_balance(self, partner):
        # .get() looks in the dictionary for the partner's ID. 
        # If it doesn't find it, it defaults to returning 0 (no debt history).
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        """This is where the agent asks the AI Brain what to do."""
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        # We pass our 3 variables into the mathematical rule DEAP invented.
        # *args is just a Python shortcut for unpacking the variables.
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        
        # If the math formula outputs a number greater than 0, the agent says "YES" to trading.
        return decision_score > 0

    def step(self):
        """The 'step' is the heartbeat. It runs once per 'tick' of the clock."""
        
        # 1. PRODUCE: The agent magically creates 1 unit of their specialty good.
        self.inventory[self.specialty_good] += 1
        
        # 2. FIND A PARTNER: Look at everyone in the model, exclude myself, and pick one randomly.
        other_agents = [a for a in self.model.agents if a != self]
        if not other_agents: return
        partner = random.choice(other_agents)
        
        # 3. THE DOUBLE-BLIND EVALUATION:
        # Both agents look at each other and consult their AI rule.
        i_want_to_give = self.evaluate_rule(partner)
        they_want_to_give = partner.evaluate_rule(self)

        # 4. THE INTERACTION: Combining Spot Exchange and Debt!
        
        # Scenario A: We both said YES. Instant Spot Exchange!
        if i_want_to_give and they_want_to_give:
            self.spot_swap(partner)
            
        # Scenario B: I said YES, but they said NO (or they are broke).
        # I give them a good, and they go into debt to me.
        elif i_want_to_give and not they_want_to_give:
            self.give_on_credit(partner)
            
        # Scenario C: They said YES, I said NO. They give me a good, I go into debt.
        elif not i_want_to_give and they_want_to_give:
            partner.give_on_credit(self)
            
        # Scenario D: Both said NO. Nothing happens. They walk away.

    def spot_swap(self, partner):
        """Spot Exchange: We swap goods simultaneously. No debt is recorded."""
        # Check to make sure we actually have goods to give!
        if self.inventory[self.specialty_good] > 0 and partner.inventory[partner.specialty_good] > 0:
            # I lose mine, they gain it
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            # They lose theirs, I gain it
            partner.inventory[partner.specialty_good] -= 1
            self.inventory[partner.specialty_good] += 1

    def give_on_credit(self, partner):
        """Debt Primitive: Unilateral transfer. I give a good, you give an IOU."""
        if self.inventory[self.specialty_good] > 0:
            # Transfer the good
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            # Record the Debt! 
            # They owe me 1 more than they did yesterday.
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            
            # I owe them 1 less than I did yesterday.
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


# -----------------------------------------------------------------------------
# CLASS: CoreModel
# OOP LESSON: If EconomicAgent is the blueprint for a person, CoreModel is the 
# blueprint for the planet they live on.
# -----------------------------------------------------------------------------
class CoreModel(mesa.Model):
    
    # N is the number of agents. rule is the math equation from DEAP.
    def __init__(self, N, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule 
        
        # Let's create our agents!
        for i in range(self.num_agents):
            # The modulo operator (%) wraps the numbers. 
            # Agent 0 makes good 0. Agent 10 makes good 0 again.
            # This ensures all 10 goods are produced equally in the economy.
            specialty = i % 10 
            EconomicAgent(self, specialty)

    def step(self):
        """This tells Mesa to make every agent run their 'step' function once."""
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        """
        The Macro Target. Since there are 10 goods, and 30 agents, 
        the maximum possible score for a perfect utopian economy is 300.
        """
        total_diverse_goods = 0
        for agent in self.agents:
            diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
            total_diverse_goods += diverse_goods
        return total_diverse_goods