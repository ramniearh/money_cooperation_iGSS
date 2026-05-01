import mesa
import random

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_CONFIG = {
    # 1. Active Mechanisms (The Master Switches)
    "USE_STANDING": False,  # ARG0: Indirect Reciprocity
    "USE_MEMORY": False,    # ARG1: Direct Reciprocity
    "USE_TOKENS": True,     # ARG2: Monetary Exchange
    
    # 2. Economic Parameters
    "BENEFIT_TO_COST_RATIO": 3,
    "COST": 1,
    "INITIAL_LIQUIDITY": 2.0,
    
    # 3. Population Distribution
    "NUM_IGSS": 10,  # The AI Lab Rats
    "NUM_UC": 10,     # Unconditional Cooperators (Sheep)
    "NUM_D": 10,      # Unconditional Defectors (Wolves)
    
    # 4. Simulation Duration
    "NUM_ROUNDS": 20 
}
# =============================================================================

class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        
        self.payoff = 0
        self.memory = set()
        self.standing = 1
        
        liq = self.model.config["INITIAL_LIQUIDITY"]
        if liq < 1.0:
            self.tokens = 1 if random.random() < liq else 0
        else:
            self.tokens = int(liq)

    def evaluate_partner(self, partner):
        # Hardcoded static behaviors
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # iGSS Discovery behavior
        arg0 = partner.standing if self.model.config["USE_STANDING"] else 0
        arg1 = 1 if partner.unique_id in self.memory and self.model.config["USE_MEMORY"] else 0
        arg2 = partner.tokens if self.model.config["USE_TOKENS"] else 0
        
        decision_score = self.model.igss_rule(arg0, arg1, arg2)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        # Populate the room
        for _ in range(config["NUM_IGSS"]): CoopAgent(self, "iGSS-Agent")
        for _ in range(config["NUM_UC"]): CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): CoopAgent(self, "Defector")

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for i in range(0, len(agents) - 1, 2):
            agent_A = agents[i]
            agent_B = agents[i+1]

            a_coops = agent_A.evaluate_partner(agent_B)
            b_coops = agent_B.evaluate_partner(agent_A)

            self.resolve(helper=agent_A, recipient=agent_B, cooperates=a_coops)
            self.resolve(helper=agent_B, recipient=agent_A, cooperates=b_coops)

    def resolve(self, helper, recipient, cooperates):
        if cooperates:
            helper.payoff -= self.cost
            recipient.payoff += self.benefit
            helper.standing = 1
            
            if self.config["USE_TOKENS"] and recipient.tokens > 0:
                recipient.tokens -= 1
                helper.tokens += 1
        else:
            if recipient.standing == 1:
                helper.standing = 0
            
            recipient.memory.add(helper.unique_id)
            
            if recipient.unique_id in helper.memory:
                helper.memory.remove(recipient.unique_id)

    def get_igss_fitness(self):
        """Returns the average payoff ONLY of the iGSS agents"""
        igss_agents = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        if not igss_agents: return 0
        return sum(a.payoff for a in igss_agents) / len(igss_agents)