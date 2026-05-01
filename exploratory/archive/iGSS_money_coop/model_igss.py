import mesa
import random

class IgssAgent(mesa.Agent):
    def __init__(self, model, is_defector=False):
        super().__init__(model)
        self.is_defector = is_defector # Hardcoded sociopaths to apply evolutionary pressure
        self.fitness = 0
        self.score = model.initial_reputations
        self.memory = []
        
        if model.initial_liquidity < 1.0:
            self.balance = 1 if random.random() < model.initial_liquidity else 0
        else:
            self.balance = int(model.initial_liquidity)

    def evaluate_rule(self, partner):
        if self.is_defector:
            return False
            
        # The 3 variables available to the AI's heuristic
        in_memory = 1 if partner.unique_id in self.memory else 0
        partner_rep = partner.score
        partner_balance = partner.balance
        
        decision = self.model.decision_rule(in_memory, partner_rep, partner_balance)
        return decision > 0

    def step(self):
        other_agents = [a for a in self.model.agents if a != self]
        partner = random.choice(other_agents)

        i_want_to_help = self.evaluate_rule(partner)

        if i_want_to_help:
            # The Universal Law of Cooperation & Exchange
            self.fitness -= self.model.cost
            partner.fitness += self.model.benefit
            self.score += 1
            
            # Money transfer is automatic IF the partner has liquidity
            if partner.balance > 0:
                self.balance += 1
                partner.balance -= 1
        else:
            # The Universal Law of Defection
            self.score -= 1
            if self.unique_id not in partner.memory:
                partner.memory.append(self.unique_id)
            # Memory wipe logic for retaliation
            if partner.unique_id in self.memory:
                self.memory.remove(partner.unique_id)

class IgssModel(mesa.Model):
    def __init__(self, N, num_defectors, rule, bcr=5.0, liq=1.0, init_rep=1):
        super().__init__()
        self.cost = 1
        self.benefit = self.cost * bcr
        self.initial_liquidity = liq
        self.initial_reputations = init_rep
        self.decision_rule = rule 
        
        for _ in range(N - num_defectors):
            IgssAgent(self, is_defector=False)
        for _ in range(num_defectors):
            IgssAgent(self, is_defector=True)

    def step(self):
        for agent in self.agents:
            agent.fitness = 0 
        self.agents.shuffle_do("step")
        for agent in self.agents:
            agent.fitness += 1 
            
    def calculate_fitness(self):
        # We target the aggregate fitness of the AI-driven agents
        return sum(a.fitness for a in self.agents if not a.is_defector)