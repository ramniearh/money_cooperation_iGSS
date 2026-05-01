import mesa
import random
import networkx as nx # Mesa uses NetworkX for the heavy lifting

class EconomicAgent(mesa.Agent):
    def __init__(self, model, specialty_good):
        super().__init__(model)
        self.specialty_good = specialty_good
        self.inventory = {i: 0 for i in range(10)} 
        self.inventory[self.specialty_good] = 5 
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
        self.inventory[self.specialty_good] += 1
        
        # --- NEW NETWORK LOGIC ---
        # Instead of picking from 'all agents', we pick from our 'neighbors'
        # on the network grid.
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbor_agents = self.model.grid.get_cell_list_contents(neighbors_nodes)
        
        if not neighbor_agents: return
        partner = random.choice(neighbor_agents)
        # -------------------------
        
        i_want_to_give = self.evaluate_rule(partner)
        they_want_to_give = partner.evaluate_rule(self)

        if i_want_to_give and they_want_to_give:
            self.spot_swap(partner)
        elif i_want_to_give and not they_want_to_give:
            self.give_on_credit(partner)
        elif not i_want_to_give and they_want_to_give:
            partner.give_on_credit(self)

    def spot_swap(self, partner):
        if self.inventory[self.specialty_good] > 0 and partner.inventory[partner.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            partner.inventory[partner.specialty_good] -= 1
            self.inventory[partner.specialty_good] += 1

    def give_on_credit(self, partner):
        if self.inventory[self.specialty_good] > 0:
            self.inventory[self.specialty_good] -= 1
            partner.inventory[self.specialty_good] += 1
            self.ledger[partner.unique_id] = self.get_balance(partner) + 1
            partner.ledger[self.unique_id] = partner.get_balance(self) - 1

class NetworkModel(mesa.Model):
    def __init__(self, N, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule 
        
        # --- INITIALIZE NETWORK ---
        # Create a Ring Lattice (each agent connected to k neighbors)
        self.G = nx.watts_strogatz_graph(n=N, k=2, p=0)
        self.grid = mesa.space.NetworkGrid(self.G)
        # --------------------------
        
        for i, node in enumerate(self.G.nodes()):
            specialty = i % 10 
            a = EconomicAgent(self, specialty)
            self.grid.place_agent(a, node)

    def step(self):
        self.agents.shuffle_do("step")
        
    def calculate_fitness(self):
        total_diverse_goods = 0
        for agent in self.agents:
            diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
            total_diverse_goods += diverse_goods
        return total_diverse_goods