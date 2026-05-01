import mesa
import random

class PrimitiveAgent(mesa.Agent):
    def __init__(self, model, is_defector=False):
        super().__init__(model)
        self.is_defector = is_defector 
        self.fitness = 0
        
        # --- THE RAW PRIMITIVES (Action History) ---
        # Global Counters
        self.helped_others = 0
        self.helped_by_others = 0
        
        # Local/Bilateral Ledgers
        self.favors_given_to = {}
        self.favors_received_from = {}

    def evaluate_rule(self, partner):
        """The AI looks at the raw data to decide if it will help the partner."""
        if self.is_defector:
            return False
            
        # ARG0: Global Given (How many times has this partner helped ANYONE?)
        global_given = partner.helped_others
        
        # ARG1: Global Received (How many times has this partner been helped by ANYONE?)
        global_received = partner.helped_by_others
        
        # ARG2: Local Given (How many times has this partner helped ME?)
        local_given = self.favors_received_from.get(partner.unique_id, 0)
        
        # ARG3: Local Received (How many times have I helped this partner?)
        local_received = self.favors_given_to.get(partner.unique_id, 0)
        
        # Feed the primitives to the AI's evolved equation
        decision = self.model.decision_rule(global_given, global_received, local_given, local_received)
        return decision > 0

    def step(self):
        other_agents = [a for a in self.model.agents if a != self]
        if not other_agents: return
        
        # Pick a random person to potentially help
        partner = random.choice(other_agents)

        i_want_to_help = self.evaluate_rule(partner)

        if i_want_to_help:
            # The Universal Law of Energy Transfer
            self.fitness -= self.model.cost
            partner.fitness += self.model.benefit
            
            # --- LOG THE RAW ACTIONS ---
            self.helped_others += 1
            partner.helped_by_others += 1
            
            self.favors_given_to[partner.unique_id] = self.favors_given_to.get(partner.unique_id, 0) + 1
            partner.favors_received_from[self.unique_id] = partner.favors_received_from.get(self.unique_id, 0) + 1


class PrimitiveModel(mesa.Model):
    def __init__(self, N, num_defectors, rule, bcr=5.0):
        super().__init__()
        self.cost = 1
        self.benefit = self.cost * bcr
        self.decision_rule = rule 
        
        # Create AI Agents
        for _ in range(N - num_defectors):
            PrimitiveAgent(self, is_defector=False)
            
        # Create Hardcoded Parasites
        for _ in range(num_defectors):
            PrimitiveAgent(self, is_defector=True)

    def step(self):
        for agent in self.agents:
            agent.fitness = 0 
        self.agents.shuffle_do("step")
        for agent in self.agents:
            agent.fitness += 1 
            
    def calculate_fitness(self):
        # The Macro Target: Maximize the aggregate fitness of the AI-driven society
        return sum(a.fitness for a in self.agents if not a.is_defector)