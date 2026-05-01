"""
Agent definitions for the IGSS framework.

Implements cooperative agents with different behavioral strategies
including memory-based (DR), reputation-based (IR), and hybrid approaches.
"""

from typing import Optional, Callable, TYPE_CHECKING
import mesa

if TYPE_CHECKING:
    from .models import CooperationModel


class CoopAgent(mesa.Agent):
    """
    Cooperative agent for social simulation experiments.
    
    Agents can be of different types:
    - iGSS-Agent: Uses evolved GP rules for decision-making
    - Control-*: Uses hardcoded baseline strategies
    - Unconditional Cooperator: Always cooperates
    - Defector: Always defects
    
    Attributes:
        agent_type: Classification of agent behavior
        payoff: Cumulative payoff from interactions
        memory: Set of agent IDs that defected against this agent (DR)
        standing: Binary reputation status (IR)
        token: Currency holding for money experiments
    """
    
    def __init__(self, model: "CooperationModel", agent_type: str = "iGSS-Agent"):
        """
        Initialize a cooperative agent.
        
        Args:
            model: The CooperationModel containing this agent
            agent_type: Type of agent behavior
        """
        super().__init__(model)
        self.agent_type = agent_type
        self.payoff = 0.0
        
        # Mechanism-specific state
        self.memory: set = set()  # For Direct Reciprocity
        self.standing: int = 1    # For Indirect Reciprocity (1=Good, 0=Bad)
        self.token: int = 0       # For Money experiments
    
    def evaluate_partner(self, partner: "CoopAgent") -> bool:
        """
        Decide whether to cooperate with a potential recipient.
        
        Args:
            partner: The agent being evaluated as recipient
            
        Returns:
            True to cooperate, False to defect
        """
        # Unconditional strategies
        if self.agent_type == "Unconditional Cooperator":
            return True
        if self.agent_type == "Defector":
            return False
        
        # Control baselines (hardcoded strategies)
        if self._evaluate_control(partner):
            return True
        
        # iGSS evolved strategies
        return self._evaluate_igss(partner)
    
    def _evaluate_control(self, partner: "CoopAgent") -> Optional[bool]:
        """
        Evaluate using hardcoded control strategies.
        
        Returns:
            Boolean decision if control strategy applies, None otherwise
        """
        model = self.model
        mechanism = model.config.mechanism
        
        # Direct Reciprocity: Tit-for-Tat
        if self.agent_type == "Control-TFT":
            return partner.unique_id not in self.memory
        
        # Indirect Reciprocity: Standing Discriminator
        if self.agent_type == "Control-Standing":
            return partner.standing == 1
        
        # Joint: Hybrid AND logic
        if self.agent_type == "Control-Hybrid":
            in_memory = partner.unique_id in self.memory
            return (not in_memory) and (partner.standing == 1)
        
        # Joint Mode 2: AND/OR logic
        if self.agent_type == "Control-Agent" and mechanism == "joint":
            in_memory = partner.unique_id in self.memory
            good_standing = partner.standing == 1
            logic = model.config.model.action_logic
            
            if logic == "AND":
                return (not in_memory) and good_standing
            elif logic == "OR":
                return (not in_memory) or good_standing
        
        # Money: Accumulating Capitalist
        if self.agent_type == "Control-Agent" and mechanism == "money":
            return partner.token >= 1
        
        # IR Mode 2: Strict Discriminator
        if self.agent_type == "Control-Agent" and mechanism == "IR":
            return partner.standing == 1
        
        # DR Mode 2/3: Strict Tit-for-Tat
        if self.agent_type == "Control-Agent" and mechanism == "DR":
            return partner.unique_id not in self.memory
        
        return None  # Not a control agent
    
    def _evaluate_igss(self, partner: "CoopAgent") -> bool:
        """
        Evaluate using evolved iGSS rules.
        
        Returns:
            True if decision_score > 0, False otherwise
        """
        model = self.model
        mechanism = model.config.mechanism
        mode = model.config.mode
        
        # Get the appropriate evolved rule
        igss_rule = model.igss_action_rule
        
        if igss_rule is None:
            # Fallback if no rule provided
            return True
        
        # Prepare inputs based on mechanism and mode
        if mechanism == "DR":
            in_memory = 1 if partner.unique_id in self.memory else 0
            decision_score = igss_rule(in_memory)
            
        elif mechanism == "IR":
            decision_score = igss_rule(partner.standing)
            
        elif mechanism == "joint":
            in_memory = 1 if partner.unique_id in self.memory else 0
            decision_score = igss_rule(in_memory, partner.standing)
            
        elif mechanism == "money":
            decision_score = igss_rule(partner.token, self.token)
            
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        return decision_score > 0
    
    def reset_state(self):
        """Reset agent state for new simulation run."""
        self.payoff = 0.0
        self.memory = set()
        self.standing = 1
        self.token = 0
