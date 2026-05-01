import mesa
import random

class EconomicAgent(mesa.Agent):
    """An agent wandering a 2D space, searching for trade partners."""
    
    def __init__(self, model, specialty_good):
        super().__init__(model) 
        self.specialty_good = specialty_good
        self.inventory = {0: 0, 1: 0, 2: 0} 
        self.inventory[self.specialty_good] = 5 
        
        # The Ledger
        self.ledger = {} 

    def get_balance(self, partner):
        return self.ledger.get(partner.unique_id, 0)

    def evaluate_rule(self, partner):
        my_supply = self.inventory[self.specialty_good]
        partner_supply = partner.inventory[partner.specialty_good]
        partner_debt = self.get_balance(partner) 
        
        decision_score = self.model.decision_rule(my_supply, partner_supply, partner_debt)
        return decision_score > 0

    def step(self):
        # 1. Produce
        self.inventory[self.specialty_good] += 1
        
        # 2. Move (The Random Walk Friction)
        # Find neighboring cells and pick one randomly
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

        # 3. Social Interaction (Only with agents in the EXACT SAME cell)
        cellmates = [a for a in self.model.grid.get_cell_list_contents([self.pos]) if a != self]
        
        if cellmates:
            partner = random.choice(cellmates)
            if self.evaluate_rule(partner):
                self.give_on_credit(partner)

    def give_on_credit(self, partner):
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1


class TopologyModel(mesa.Model):
    """The macro-environment with spatial constraints."""
    
    def __init__(self, N, width, height, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule 
        
        # NEW: The Spatial Grid (torus=True means they wrap around the edges like Pac-Man)
        self.grid = mesa.space.MultiGrid(width, height, torus=True)
        
        for i in range(self.num_agents):
            specialty = i % 3 
            a = EconomicAgent(self, specialty)
            
            # Place the agent on a random cell in the grid
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        total_diverse_goods = 0
        for agent in self.agents:
            diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
            total_diverse_goods += diverse_goods
        return total_diverse_goods