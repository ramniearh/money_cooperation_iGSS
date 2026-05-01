"""
Configuration classes for IGSS experiments.

All configurations are immutable dataclasses to ensure reproducibility
and prevent accidental mutation during experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class ModelConfig:
    """
    Agent-Based Model configuration parameters.
    
    These parameters control the simulation dynamics including
    population composition, payoff structure, and game length.
    """
    benefit_to_cost_ratio: float = 5.0
    cost: float = 1.0
    num_igss: int = 10
    num_uc: int = 10
    num_d: int = 10
    num_rounds: int = 100
    
    # Mode-specific parameters
    action_logic: str = "AND"  # For jointDRIR_mode2: "AND" or "OR"
    
    # Money-specific parameters
    initial_endowment: int = 1
    endowment_fraction: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for legacy compatibility."""
        return {
            "BENEFIT_TO_COST_RATIO": self.benefit_to_cost_ratio,
            "COST": self.cost,
            "NUM_IGSS": self.num_igss,
            "NUM_UC": self.num_uc,
            "NUM_D": self.num_d,
            "NUM_ROUNDS": self.num_rounds,
            "ACTION_LOGIC": self.action_logic,
            "INITIAL_ENDOWMENT": self.initial_endowment,
            "ENDOWMENT_FRACTION": self.endowment_fraction,
        }


@dataclass(frozen=True)
class EvoConfig:
    """
    Evolutionary Algorithm configuration parameters.
    
    These parameters control the genetic programming search including
    population size, generations, and selection pressure.
    """
    pop_size: int = 40
    max_gens: int = 100
    parsimony_tax: float = 0.1
    
    # Genetic operator probabilities
    crossover_prob: float = 0.5
    mutation_prob: float = 0.2
    
    # Evaluation parameters
    evaluation_runs: int = 3
    baseline_runs: int = 5
    
    # Selection parameters
    tournament_size: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for legacy compatibility."""
        return {
            "POP_SIZE": self.pop_size,
            "MAX_GENS": self.max_gens,
            "PARSIMONY_TAX": self.parsimony_tax,
        }


@dataclass(frozen=True)
class PrimitiveConfig:
    """
    Genetic Programming primitive set configuration.
    
    Defines the input arguments and available operations for evolved rules.
    """
    name: str
    arity: int
    argument_names: List[str]
    
    # Predefined configurations for each experiment type
    @classmethod
    def dr_action(cls) -> "PrimitiveConfig":
        """Direct Reciprocity Action Rule: PartnerInMemory -> decision."""
        return cls(
            name="ActionRule_DR",
            arity=1,
            argument_names=["PartnerInMemory"]
        )
    
    @classmethod
    def dr_assessment(cls) -> "PrimitiveConfig":
        """Direct Reciprocity Assessment Rule: (DonorAction, DonorInMemory) -> memory_update."""
        return cls(
            name="AssessmentRule_DR",
            arity=2,
            argument_names=["DonorAction", "DonorInMemory"]
        )
    
    @classmethod
    def ir_action(cls) -> "PrimitiveConfig":
        """Indirect Reciprocity Action Rule: PartnerStanding -> decision."""
        return cls(
            name="ActionRule_IR",
            arity=1,
            argument_names=["PartnerStanding"]
        )
    
    @classmethod
    def ir_assessment_donor(cls) -> "PrimitiveConfig":
        """IR Assessment with donor standing: (DonorAction, DonorStanding, RecipientStanding)."""
        return cls(
            name="AssessmentRule_IR",
            arity=3,
            argument_names=["DonorAction", "DonorStanding", "RecipientStanding"]
        )
    
    @classmethod
    def ir_assessment_strict(cls) -> "PrimitiveConfig":
        """IR Assessment without donor standing: (DonorAction, RecipientStanding)."""
        return cls(
            name="AssessmentRule_IR_Strict",
            arity=2,
            argument_names=["DonorAction", "RecipientStanding"]
        )
    
    @classmethod
    def joint_action(cls) -> "PrimitiveConfig":
        """Joint DR+IR Action Rule: (PartnerInMemory, PartnerStanding) -> decision."""
        return cls(
            name="ActionRule_Combined",
            arity=2,
            argument_names=["PartnerInMemory", "PartnerStanding"]
        )
    
    @classmethod
    def joint_assessment(cls) -> "PrimitiveConfig":
        """Joint DR+IR Assessment: (DonorAction, DonorInMemory, DonorStanding, RecipientStanding)."""
        return cls(
            name="AssessmentRule_Combined",
            arity=4,
            argument_names=["DonorAction", "DonorInMemory", "DonorStanding", "RecipientStanding"]
        )
    
    @classmethod
    def money_action(cls) -> "PrimitiveConfig":
        """Money Action Rule: (PartnerToken, MyToken) -> decision."""
        return cls(
            name="ActionRule_MoneyAcc",
            arity=2,
            argument_names=["PartnerToken", "MyToken"]
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Encapsulates all parameters needed to run a specific experimental mode
    with exact reproducibility.
    """
    name: str
    mode: str  # "mode1", "mode2", "mode3"
    mechanism: str  # "DR", "IR", "joint", "money"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    evolution: EvoConfig = field(default_factory=EvoConfig)
    
    # Primitive configurations for each tree
    action_primitive: Optional[PrimitiveConfig] = None
    assessment_primitive: Optional[PrimitiveConfig] = None
    
    # Random seed for exact reproducibility
    seed: int = 42
    
    # Number of trees in the individual (1, 2, or 3)
    num_trees: int = 1
    
    # Control agent type name
    control_agent_type: str = "Control-Agent"
    
    # iGSS agent type name
    igss_agent_type: str = "iGSS-Agent"
    
    def __post_init__(self):
        """Validate configuration consistency."""
        if self.mode == "mode1" and self.num_trees != 1:
            raise ValueError(f"Mode 1 requires 1 tree, got {self.num_trees}")
        if self.mode == "mode2" and self.num_trees != 1:
            raise ValueError(f"Mode 2 requires 1 tree, got {self.num_trees}")
        if self.mode == "mode3" and self.num_trees not in [2, 3]:
            raise ValueError(f"Mode 3 requires 2 or 3 trees, got {self.num_trees}")


# ============================================================================
# Predefined Experiment Configurations (Exact matches to original files)
# ============================================================================

DR_MODE1_CONFIG = ExperimentConfig(
    name="DR_mode1",
    mode="mode1",
    mechanism="DR",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=40, max_gens=100, parsimony_tax=0.1),
    action_primitive=PrimitiveConfig.dr_action(),
    seed=42,
    control_agent_type="Control-TFT",
)

DR_MODE2_CONFIG = ExperimentConfig(
    name="DR_mode2",
    mode="mode2",
    mechanism="DR",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=40, max_gens=100, parsimony_tax=0.1),
    assessment_primitive=PrimitiveConfig.dr_assessment(),
    seed=42,
)

DR_MODE3_CONFIG = ExperimentConfig(
    name="DR_mode3",
    mode="mode3",
    mechanism="DR",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=100, max_gens=300, parsimony_tax=0.05),
    action_primitive=PrimitiveConfig.dr_action(),
    assessment_primitive=PrimitiveConfig.dr_assessment(),
    seed=42,
    num_trees=2,
)

IR_MODE1_CONFIG = ExperimentConfig(
    name="IR_mode1",
    mode="mode1",
    mechanism="IR",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=40, max_gens=100, parsimony_tax=0.1),
    action_primitive=PrimitiveConfig.ir_action(),
    seed=42,
    control_agent_type="Control-Standing",
)

IR_MODE2_DONOR_CONFIG = ExperimentConfig(
    name="IR_mode2_donor",
    mode="mode2",
    mechanism="IR",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=40, max_gens=100, parsimony_tax=0.1),
    assessment_primitive=PrimitiveConfig.ir_assessment_donor(),
    seed=42,
)

IR_MODE2_NODONOR_CONFIG = ExperimentConfig(
    name="IR_mode2_nodonor",
    mode="mode2",
    mechanism="IR",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=40, max_gens=100, parsimony_tax=0.1),
    assessment_primitive=PrimitiveConfig.ir_assessment_strict(),
    seed=42,
)

IR_MODE3_CONFIG = ExperimentConfig(
    name="IR_mode3",
    mode="mode3",
    mechanism="IR",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=100, max_gens=200, parsimony_tax=0.05),
    action_primitive=PrimitiveConfig.ir_action(),
    assessment_primitive=PrimitiveConfig.ir_assessment_strict(),
    seed=42,
    num_trees=2,
)

JOINT_MODE1_CONFIG = ExperimentConfig(
    name="jointDRIR_mode1",
    mode="mode1",
    mechanism="joint",
    model=ModelConfig(),
    evolution=EvoConfig(pop_size=40, max_gens=100, parsimony_tax=0.1),
    action_primitive=PrimitiveConfig.joint_action(),
    seed=42,
    control_agent_type="Control-Hybrid",
)

JOINT_MODE2_CONFIG = ExperimentConfig(
    name="jointDRIR_mode2_AND_OR_twotrees",
    mode="mode2",
    mechanism="joint",
    model=ModelConfig(
        action_logic="OR",  # Toggle between "AND" and "OR"
    ),
    evolution=EvoConfig(pop_size=60, max_gens=150, parsimony_tax=0.05),
    assessment_primitive=PrimitiveConfig.joint_assessment(),
    seed=421,  # Note: different seed in original
)

JOINT_MODE3_CONFIG = ExperimentConfig(
    name="jointDRIR_mode3_threetrees",
    mode="mode3",
    mechanism="joint",
    model=ModelConfig(
        num_igss=30,
    ),
    evolution=EvoConfig(
        pop_size=500, 
        max_gens=1000, 
        parsimony_tax=0.01,
        tournament_size=7,
    ),
    action_primitive=PrimitiveConfig.joint_action(),
    assessment_primitive=PrimitiveConfig.joint_assessment(),
    seed=42,
    num_trees=3,
)

MONEY_MODE1_CONFIG = ExperimentConfig(
    name="MONEY_mode1_integer_exclusive_defenced",
    mode="mode1",
    mechanism="money",
    model=ModelConfig(
        benefit_to_cost_ratio=2,
        initial_endowment=1,
        endowment_fraction=0.5,
    ),
    evolution=EvoConfig(pop_size=40, max_gens=100, parsimony_tax=0.1),
    action_primitive=PrimitiveConfig.money_action(),
    seed=42,
)
