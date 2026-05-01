"""
Core Agent-Based Model for iGSS

Shared Agent and Model classes used across all modes.
Uses Mesa framework for agent-based simulation.
"""

import random
import mesa
from mechanisms import (
    get_memory_signal, get_standing_signal, get_token_signal,
    initialize_tokens, check_partner_solvent
)


# =============================================================================
# DATA COLLECTOR FUNCTIONS
# =============================================================================

def compute_igss_wealth(model):
    """Sum of tokens held by iGSS agents."""
    return sum(a.tokens for a in model.agents if a.agent_type == "iGSS-Agent")


def compute_defector_wealth(model):
    """Sum of tokens held by Defector agents."""
    return sum(a.tokens for a in model.agents if a.agent_type == "Defector")


# =============================================================================
# AGENT CLASS
# =============================================================================

class CoopAgent(mesa.Agent):
    """
    Agent in the cooperation game.
    
    Agent Types:
    - "iGSS-Agent": Evolvable agent using learned rules
    - "Unconditional Cooperator": Always cooperates
    - "Defector": Always defects
    """
    
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model)
        self.agent_type = agent_type
        self.payoff = 0
        self.memory = set()      # For Direct Reciprocity: IDs of agents who defected against me
        self.standing = 1        # For Indirect Reciprocity: 1 = Good, 0 = Bad
        self.tokens = 0          # For Monetary Exchange
        
        # Initialize tokens if money mechanism is active
        if model.config.get("USE_TOKENS", False):
            initialize_tokens(self, model.config)
    
    def evaluate_partner(self, partner, action_rule=None):
        """
        Decide whether to cooperate with partner.
        
        Args:
            partner: The agent being evaluated
            action_rule: Compiled GP function (for iGSS agents). 
                        If None, uses hardcoded logic (for Mode 2).
                        
        Returns:
            bool: True to cooperate, False to defect
        """
        # Fixed agent types
        if self.agent_type == "Unconditional Cooperator":
            return True
        if self.agent_type == "Defector":
            return False
        
        # iGSS-Agent: Use evolved or hardcoded rule
        config = self.model.config
        
        # Mode 2: Hardcoded action rule (cooperate if standing > 0)
        if action_rule is None:
            return partner.standing > 0
        
        # Mode 1 & 3: Use evolved action rule with signals
        # Gather signals based on config
        arg0 = get_memory_signal(self, partner) if config.get("USE_MEMORY", False) else 0
        arg1 = get_standing_signal(partner) if config.get("USE_STANDING", False) else 0
        arg2 = get_token_signal(partner) if config.get("USE_TOKENS", False) else 0
        
        # Check solvency constraint for money
        if config.get("REQUIRE_TOKENS_TO_COOPERATE", False):
            if not check_partner_solvent(partner, config):
                return False
        
        # Evaluate the rule
        decision_score = action_rule(arg0, arg1, arg2)
        return decision_score > 0


# =============================================================================
# MODEL CLASS
# =============================================================================

class CooperationModel(mesa.Model):
    """
    Agent-based model for cooperation game.
    
    Supports three modes:
    - Mode 1: Evolved action rules with hardcoded standing update
    - Mode 2: Hardcoded action rules with evolved assessment
    - Mode 3: Co-evolved action and assessment rules
    """
    
    def __init__(self, config, action_rule=None, assessment_rule=None):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
            action_rule: Compiled GP function for action decisions (Modes 1 & 3)
            assessment_rule: Compiled GP function for standing updates (Modes 2 & 3)
        """
        super().__init__()
        self.config = config
        self.action_rule = action_rule
        self.assessment_rule = assessment_rule
        
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        # Create agents
        for _ in range(config["NUM_IGSS"]):
            CoopAgent(self, "iGSS-Agent")
        for _ in range(config["NUM_UC"]):
            CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]):
            CoopAgent(self, "Defector")
        
        # Data collector (only if tokens are used)
        if config.get("USE_TOKENS", False):
            self.datacollector = mesa.DataCollector(
                model_reporters={
                    "iGSS_Wealth": compute_igss_wealth,
                    "Defector_Wealth": compute_defector_wealth
                }
            )
        else:
            self.datacollector = None
    
    def step(self):
        """Execute one round of the simulation."""
        from mechanisms import update_memory, update_tokens
        from mechanisms import update_standing_hardcoded, update_standing_evolved
        
        agents = list(self.agents)
        random.shuffle(agents)
        
        # Pair up agents and interact
        for i in range(0, len(agents) - 1, 2):
            agent_A = agents[i]
            agent_B = agents[i+1]
            
            # Both agents evaluate their partner
            a_coops = agent_A.evaluate_partner(agent_B, self.action_rule)
            b_coops = agent_B.evaluate_partner(agent_A, self.action_rule)
            
            # Resolve both interactions
            self._resolve(agent_A, agent_B, a_coops, update_memory, update_tokens,
                         update_standing_hardcoded, update_standing_evolved)
            self._resolve(agent_B, agent_A, b_coops, update_memory, update_tokens,
                         update_standing_hardcoded, update_standing_evolved)
        
        # Collect data if using tokens
        if self.datacollector:
            self.datacollector.collect(self)
    
    def _resolve(self, helper, recipient, cooperates, update_memory_fn, 
                 update_tokens_fn, update_standing_hardcoded_fn, update_standing_evolved_fn):
        """
        Resolve a single interaction.
        
        Args:
            helper: The agent making the decision
            recipient: The agent receiving the decision
            cooperates: Boolean - did helper cooperate?
            update_memory_fn: Function to update memory
            update_tokens_fn: Function to update tokens
            update_standing_hardcoded_fn: Function for hardcoded standing update
            update_standing_evolved_fn: Function for evolved standing update
        """
        if cooperates:
            # Payoff transfer
            helper.payoff -= self.cost
            recipient.payoff += self.benefit
        
        # Update mechanisms based on mode
        config = self.config
        
        # Memory update (Direct Reciprocity)
        if config.get("USE_MEMORY", False):
            update_memory_fn(helper, recipient, cooperates)
        
        # Token update (Monetary Exchange)
        if config.get("USE_TOKENS", False):
            update_tokens_fn(helper, recipient, cooperates, config)
        
        # Standing update (Indirect Reciprocity)
        if config.get("USE_STANDING", False):
            if self.assessment_rule is not None:
                # Modes 2 & 3: Evolved assessment
                update_standing_evolved_fn(helper, recipient, cooperates, self.assessment_rule)
            else:
                # Mode 1: Hardcoded assessment
                update_standing_hardcoded_fn(helper, recipient, cooperates)
    
    def get_igss_fitness(self):
        """
        Calculate average payoff of iGSS agents.
        
        Returns:
            float: Average payoff (fitness) of iGSS agents
        """
        igss_agents = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        if not igss_agents:
            return 0
        return sum(a.payoff for a in igss_agents) / len(igss_agents)
