"""
Agent-Based Model for cooperation experiments.

Implements the core simulation logic for different cooperation mechanisms
including Direct Reciprocity, Indirect Reciprocity, Joint mechanisms, and Money.
"""

import random
from typing import Optional, Callable, Dict, List, TYPE_CHECKING
import mesa

from .agents import CoopAgent
from .config import ExperimentConfig

if TYPE_CHECKING:
    from .config import ModelConfig


class CooperationModel(mesa.Model):
    """
    Agent-Based Model for cooperation experiments.
    
    Simulates interactions between agents with different cooperation strategies
    in a directed network topology where donors choose recipients.
    
    Attributes:
        config: Experiment configuration
        igss_action_rule: Evolved GP rule for action decisions (Mode 1, 3)
        igss_assessment_rule: Evolved GP rule for assessment (Mode 2, 3)
        igss_mem_rule: Evolved GP rule for memory updates (Joint Mode 2, 3)
        igss_stand_rule: Evolved GP rule for standing updates (Joint Mode 2, 3)
        benefit: Payoff benefit from receiving cooperation
        cost: Payoff cost from cooperating
        control_mode: Whether running control baseline
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        igss_action_rule: Optional[Callable] = None,
        igss_assessment_rule: Optional[Callable] = None,
        igss_mem_rule: Optional[Callable] = None,
        igss_stand_rule: Optional[Callable] = None,
        control_mode: bool = False,
    ):
        """
        Initialize the cooperation model.
        
        Args:
            config: Complete experiment configuration
            igss_action_rule: Compiled GP action rule (or None)
            igss_assessment_rule: Compiled GP assessment rule (or None)
            igss_mem_rule: Compiled GP memory update rule (or None)
            igss_stand_rule: Compiled GP standing update rule (or None)
            control_mode: Whether to use hardcoded control strategies
        """
        super().__init__()
        self.config = config
        self.model_config = config.model
        
        # Store evolved rules
        self.igss_action_rule = igss_action_rule
        self.igss_assessment_rule = igss_assessment_rule
        self.igss_mem_rule = igss_mem_rule
        self.igss_stand_rule = igss_stand_rule
        
        self.control_mode = control_mode
        
        # Calculate payoffs
        self.benefit = config.model.benefit_to_cost_ratio * config.model.cost
        self.cost = config.model.cost
        
        # Create agents
        self._create_agents()
    
    def _create_agents(self):
        """Create agent population based on configuration."""
        target_agent = (
            self.config.control_agent_type 
            if self.control_mode 
            else self.config.igss_agent_type
        )
        
        # Create agents
        for _ in range(self.model_config.num_igss):
            CoopAgent(self, target_agent)
        for _ in range(self.model_config.num_uc):
            CoopAgent(self, "Unconditional Cooperator")
        for _ in range(self.model_config.num_d):
            CoopAgent(self, "Defector")
        
        # Initialize money for money experiments
        if self.config.mechanism == "money":
            self._initialize_money()
    
    def _initialize_money(self):
        """Initialize token endowments for money experiments."""
        economic_agents = [
            a for a in self.agents 
            if a.agent_type in [self.config.igss_agent_type, self.config.control_agent_type]
        ]
        random.shuffle(economic_agents)
        
        cutoff = int(len(economic_agents) * self.model_config.endowment_fraction)
        for i, agent in enumerate(economic_agents):
            if i < cutoff:
                agent.token = self.model_config.initial_endowment
    
    def step(self):
        """Execute one round of the simulation."""
        agents = list(self.agents)
        random.shuffle(agents)
        
        for donor in agents:
            # Select random recipient (directed network topology)
            possible_recipients = [
                a for a in agents 
                if a.unique_id != donor.unique_id
            ]
            if not possible_recipients:
                continue
                
            recipient = random.choice(possible_recipients)
            
            # Donor decides whether to cooperate
            cooperates = donor.evaluate_partner(recipient)
            
            # Resolve the interaction
            self._resolve(donor, recipient, cooperates)
    
    def _resolve(self, donor: CoopAgent, recipient: CoopAgent, cooperates: bool):
        """
        Resolve an interaction between donor and recipient.
        
        Args:
            donor: The agent giving help
            recipient: The agent receiving help
            cooperates: Whether donor chose to cooperate
        """
        mechanism = self.config.mechanism
        mode = self.config.mode
        
        # Economic execution
        if cooperates:
            donor.payoff -= self.cost
            recipient.payoff += self.benefit
        
        # Mechanism-specific resolution
        if mechanism == "DR":
            self._resolve_dr(donor, recipient, cooperates)
        elif mechanism == "IR":
            self._resolve_ir(donor, recipient, cooperates)
        elif mechanism == "joint":
            self._resolve_joint(donor, recipient, cooperates)
        elif mechanism == "money":
            self._resolve_money(donor, recipient, cooperates)
    
    def _resolve_dr(self, donor: CoopAgent, recipient: CoopAgent, cooperates: bool):
        """Direct Reciprocity resolution."""
        mode = self.config.mode
        
        if self.control_mode:
            # Hardcoded: Recipient remembers defector
            if not cooperates:
                recipient.memory.add(donor.unique_id)
                # Forgiveness: Clear memory if donor had recipient in memory
                if recipient.unique_id in donor.memory:
                    donor.memory.remove(recipient.unique_id)
        else:
            # Evolved assessment (Mode 2 and 3)
            if mode in ("mode2", "mode3"):
                action_val = 1 if cooperates else 0
                donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
                
                decision_score = self.igss_assessment_rule(action_val, donor_in_mem)
                
                if decision_score > 0:
                    recipient.memory.add(donor.unique_id)
                else:
                    recipient.memory.discard(donor.unique_id)
            else:
                # Mode 1: Hardcoded memory update
                if not cooperates:
                    recipient.memory.add(donor.unique_id)
                    if recipient.unique_id in donor.memory:
                        donor.memory.remove(recipient.unique_id)
    
    def _resolve_ir(self, donor: CoopAgent, recipient: CoopAgent, cooperates: bool):
        """Indirect Reciprocity resolution."""
        mode = self.config.mode
        
        if self.control_mode:
            # Hardcoded Standing norm
            if cooperates:
                donor.standing = 1
            else:
                if recipient.standing == 1:
                    donor.standing = 0
        else:
            # Evolved assessment (Mode 2 and 3)
            if mode in ("mode2", "mode3"):
                action_val = 1 if cooperates else 0
                
                # Get appropriate rule arguments based on primitive config
                prim = self.config.assessment_primitive
                if prim.arity == 2:
                    # Strict 2-variable: (DonorAction, RecipientStanding)
                    decision_score = self.igss_assessment_rule(
                        action_val, recipient.standing
                    )
                else:
                    # 3-variable: (DonorAction, DonorStanding, RecipientStanding)
                    decision_score = self.igss_assessment_rule(
                        action_val, donor.standing, recipient.standing
                    )
                
                donor.standing = 1 if decision_score > 0 else 0
            else:
                # Mode 1: Hardcoded standing update
                if cooperates:
                    donor.standing = 1
                else:
                    if recipient.standing == 1:
                        donor.standing = 0
    
    def _resolve_joint(self, donor: CoopAgent, recipient: CoopAgent, cooperates: bool):
        """Joint DR+IR resolution."""
        mode = self.config.mode
        
        if self.control_mode:
            # Hardcoded hybrid justice system
            if cooperates:
                donor.standing = 1
            else:
                recipient.memory.add(donor.unique_id)
                if recipient.unique_id in donor.memory:
                    donor.memory.remove(recipient.unique_id)
                if recipient.standing == 1:
                    donor.standing = 0
        else:
            # Evolved assessment
            if mode in ("mode2", "mode3"):
                action_val = 1 if cooperates else 0
                donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
                
                # Memory update (Tree 1 for mode2, Tree 1 for mode3)
                mem_score = self.igss_mem_rule(
                    action_val, donor_in_mem, donor.standing, recipient.standing
                )
                if mem_score > 0:
                    recipient.memory.add(donor.unique_id)
                else:
                    recipient.memory.discard(donor.unique_id)
                
                # Standing update (Tree 2 for mode2, Tree 2 for mode3)
                stand_score = self.igss_stand_rule(
                    action_val, donor_in_mem, donor.standing, recipient.standing
                )
                donor.standing = 1 if stand_score > 0 else 0
            else:
                # Mode 1: Hardcoded updates
                if cooperates:
                    donor.standing = 1
                else:
                    recipient.memory.add(donor.unique_id)
                    if recipient.unique_id in donor.memory:
                        donor.memory.remove(recipient.unique_id)
                    if recipient.standing == 1:
                        donor.standing = 0
    
    def _resolve_money(self, donor: CoopAgent, recipient: CoopAgent, cooperates: bool):
        """Money experiment resolution."""
        # Token settlement
        if cooperates:
            donor_cares_about_money = donor.agent_type in [
                self.config.igss_agent_type, 
                self.config.control_agent_type
            ]
            
            if donor_cares_about_money and recipient.token >= 1:
                recipient.token -= 1
                donor.token += 1
    
    def get_fitness_by_type(self) -> Dict[str, float]:
        """
        Calculate average payoff by agent type.
        
        Returns:
            Dictionary mapping agent types to average payoffs
        """
        fitness_data: Dict[str, float] = {}
        
        agent_types = [
            self.config.igss_agent_type,
            self.config.control_agent_type,
            "Unconditional Cooperator",
            "Defector"
        ]
        
        for a_type in agent_types:
            type_agents = [a for a in self.agents if a.agent_type == a_type]
            if type_agents:
                fitness_data[a_type] = sum(a.payoff for a in type_agents) / len(type_agents)
            else:
                fitness_data[a_type] = 0.0
        
        return fitness_data
    
    def run_simulation(self, num_rounds: Optional[int] = None) -> Dict[str, float]:
        """
        Run the simulation for specified rounds.
        
        Args:
            num_rounds: Number of rounds (defaults to config value)
            
        Returns:
            Final fitness by agent type
        """
        rounds = num_rounds or self.model_config.num_rounds
        
        for _ in range(rounds):
            self.step()
        
        return self.get_fitness_by_type()
