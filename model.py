import mesa
import random

# =============================================================================
# MODEL CONFIGURATION (Base Defaults)
# =============================================================================
MODEL_CONFIG = {
    "USE_MEMORY": True,    # ARG0: Direct Reciprocity (Moved to first)
    "USE_STANDING": True,  # ARG1: Indirect Reciprocity
    "USE_TOKENS": True,     # ARG2: Monetary Exchange
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "INITIAL_LIQUIDITY": 2.0,
    "NUM_IGSS": 10,  
    "NUM_UC": 10,    
    "NUM_D": 10,     
    "NUM_ROUNDS": 20 
}

def compute_igss_wealth(model):
    if not model.config["USE_TOKENS"]: return 0
    return sum(a.tokens for a in model.agents if a.agent_type == "iGSS-Agent")

def compute_defector_wealth(model):
    if not model.config["USE_TOKENS"]: return 0
    return sum(a.tokens for a in model.agents if a.agent_type == "Defector")

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
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # REORDERED ARGUMENTS: Memory (DR) -> Standing (IR) -> Tokens (Money)
        arg0 = 1 if partner.unique_id in self.memory and self.model.config["USE_MEMORY"] else 0
        arg1 = partner.standing if self.model.config["USE_STANDING"] else 0
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
        
        for _ in range(config["NUM_IGSS"]): CoopAgent(self, "iGSS-Agent")
        for _ in range(config["NUM_UC"]): CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): CoopAgent(self, "Defector")

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "iGSS_Wealth": compute_igss_wealth,
                "Defector_Wealth": compute_defector_wealth
            }
        )

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
            
        self.datacollector.collect(self)

    def resolve(self, helper, recipient, cooperates):
        if cooperates:
            helper.payoff -= self.cost
            recipient.payoff += self.benefit
            helper.standing = 1
            if self.config["USE_TOKENS"] and recipient.tokens > 0:
                recipient.tokens -= 1
                helper.tokens += 1
        else:
            if recipient.standing == 1: helper.standing = 0
            recipient.memory.add(helper.unique_id)
            if recipient.unique_id in helper.memory:
                helper.memory.remove(recipient.unique_id)

    def get_igss_fitness(self):
        igss_agents = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        if not igss_agents: return 0
        return sum(a.payoff for a in igss_agents) / len(igss_agents)