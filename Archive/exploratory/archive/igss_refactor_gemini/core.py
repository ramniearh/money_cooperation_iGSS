import mesa
import random

# =============================================================================
# UNIFIED DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    "USE_MEMORY": True,    
    "USE_STANDING": True,  
    "USE_TOKENS": True,    
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "INITIAL_LIQUIDITY": 1.0,
    "BASELINE_FITNESS": 1,   # Aligns with NetLogo: +1 per round to avoid negative fitness
    "NUM_IGSS": 10,  
    "NUM_UC": 10,    
    "NUM_D": 10,     
    "NUM_ROUNDS": 20 
}

# =============================================================================
# AGENT DEFINITION
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.memory = set()
        self.standing = 1
        
        # Initialize tokens based on liquidity settings
        liq = self.model.config["INITIAL_LIQUIDITY"]
        if liq < 1.0:
            self.tokens = 1 if random.random() < liq else 0
        else:
            self.tokens = int(liq)

    def evaluate_partner(self, partner):
        """Determines if this agent cooperates with the partner."""
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        if self.model.action_rule is None:
            raise ValueError("iGSS-Agents require an action_rule to operate.")

        # Extract signals, zeroing them out if the config disables them
        arg0_memory = 1 if (partner.unique_id in self.memory and self.model.config["USE_MEMORY"]) else 0
        arg1_standing = partner.standing if self.model.config["USE_STANDING"] else 0
        arg2_tokens = partner.tokens if self.model.config["USE_TOKENS"] else 0
        
        # Evaluate using the injected rule (whether hardcoded or evolved)
        decision_score = self.model.action_rule(arg0_memory, arg1_standing, arg2_tokens)
        return decision_score > 0

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class CooperationModel(mesa.Model):
    def __init__(self, config=DEFAULT_CONFIG, action_rule=None, assessment_rule=None):
        super().__init__()
        self.config = config
        self.action_rule = action_rule
        self.assessment_rule = assessment_rule
        
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        self.baseline_fitness = config.get("BASELINE_FITNESS", 1)
        
        # Populate Agents
        for _ in range(config["NUM_IGSS"]): CoopAgent(self, "iGSS-Agent")
        for _ in range(config["NUM_UC"]): CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): CoopAgent(self, "Defector")

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        # ASYMMETRIC MATCHING (NetLogo alignment)
        # Every agent acts as a donor exactly once per step.
        for helper in agents:
            # Pick a random recipient that is not the helper
            possible_recipients = [a for a in agents if a != helper]
            recipient = self.random.choice(possible_recipients)
            
            # Helper decides whether to cooperate
            cooperates = helper.evaluate_partner(recipient)
            
            # Resolve the single directional interaction
            self.resolve(helper, recipient, cooperates)

        # Apply baseline fitness to all agents at the end of the round
        for agent in agents:
            agent.payoff += self.baseline_fitness

    def resolve(self, helper, recipient, cooperates):
        action_signal = 1 if cooperates else 0

        # --- 1. Base Economic Execution ---
        if cooperates:
            helper.payoff -= self.cost
            recipient.payoff += self.benefit

        # --- 2. Context-Aware Assessment ---
        if self.assessment_rule is not None:
            # If an assessment rule (or wrapper) is provided, it handles all state updates
            self.assessment_rule(helper, recipient, action_signal)
        else:
            # FALLBACK MECHANICS (For pure ABM testing without DEAP injected rules)
            
            # Standing Update (Indirect Reciprocity)
            if self.config["USE_STANDING"]:
                if cooperates:
                    helper.standing = 1
                elif recipient.standing == 1:
                    helper.standing = 0
            
            # Memory Update (Direct Reciprocity)
            if self.config["USE_MEMORY"]:
                if not cooperates:
                    recipient.memory.add(helper.unique_id)
                elif helper.unique_id in recipient.memory:
                    # Clear memory upon forgiveness/cooperation (NetLogo alignment)
                    recipient.memory.remove(helper.unique_id)
                    
            # Token Exchange (Money)
            # Decoupled from the global 'law of physics'. Only applied if the helper 
            # is an iGSS-Agent, preventing Unconditional Cooperators from stealing tokens.
            if self.config["USE_TOKENS"] and helper.agent_type == "iGSS-Agent":
                if cooperates and recipient.tokens > 0:
                    recipient.tokens -= 1
                    helper.tokens += 1

    def get_igss_fitness(self):
        igss_agents = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        if not igss_agents: return 0
        return sum(a.payoff for a in igss_agents) / len(igss_agents)