"""
Experiment definitions and registry.

Provides a centralized registry for all experimental modes with
factory functions for creating configured experiments.
"""

import random
from typing import Dict, Callable, Tuple, Optional, Any
from dataclasses import dataclass

from deap import base

from .config import (
    ExperimentConfig,
    DR_MODE1_CONFIG, DR_MODE2_CONFIG, DR_MODE3_CONFIG,
    IR_MODE1_CONFIG, IR_MODE2_DONOR_CONFIG, IR_MODE2_NODONOR_CONFIG, IR_MODE3_CONFIG,
    JOINT_MODE1_CONFIG, JOINT_MODE2_CONFIG, JOINT_MODE3_CONFIG,
    MONEY_MODE1_CONFIG,
)
from .primitives import PrimitiveSetFactory, IndividualFactory
from .evolution import EvolutionaryEngine, EvolutionHistory


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    best_individual: Any
    history: EvolutionHistory
    control_fitness: float
    config: ExperimentConfig


class ExperimentRegistry:
    """
    Registry for all experimental configurations.
    
    Provides factory methods for creating and running experiments
    with exact functional equivalence to original scripts.
    """
    
    # Registry of all predefined experiments
    _experiments: Dict[str, ExperimentConfig] = {
        # Direct Reciprocity
        "DR_mode1": DR_MODE1_CONFIG,
        "DR_mode2": DR_MODE2_CONFIG,
        "DR_mode3": DR_MODE3_CONFIG,
        
        # Indirect Reciprocity
        "IR_mode1": IR_MODE1_CONFIG,
        "IR_mode2_donor": IR_MODE2_DONOR_CONFIG,
        "IR_mode2_nodonor": IR_MODE2_NODONOR_CONFIG,
        "IR_mode3": IR_MODE3_CONFIG,
        
        # Joint DR+IR
        "jointDRIR_mode1": JOINT_MODE1_CONFIG,
        "jointDRIR_mode2_AND_OR_twotrees": JOINT_MODE2_CONFIG,
        "jointDRIR_mode3_threetrees": JOINT_MODE3_CONFIG,
        
        # Money
        "MONEY_mode1_integer_exclusive_defenced": MONEY_MODE1_CONFIG,
    }
    
    @classmethod
    def list_experiments(cls) -> list:
        """List all available experiment names."""
        return list(cls._experiments.keys())
    
    @classmethod
    def get_config(cls, name: str) -> ExperimentConfig:
        """
        Get configuration for named experiment.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment configuration
            
        Raises:
            KeyError: If experiment not found
        """
        if name not in cls._experiments:
            available = ", ".join(cls.list_experiments())
            raise KeyError(f"Unknown experiment '{name}'. Available: {available}")
        
        return cls._experiments[name]
    
    @classmethod
    def register(cls, name: str, config: ExperimentConfig):
        """
        Register a new experiment configuration.
        
        Args:
            name: Experiment name
            config: Experiment configuration
        """
        cls._experiments[name] = config
    
    @classmethod
    def create_experiment(cls, name: str) -> Tuple[ExperimentConfig, Any, Any, base.Toolbox]:
        """
        Create all components for an experiment.
        
        Args:
            name: Experiment name
            
        Returns:
            Tuple of (config, action_pset, assessment_pset, toolbox)
        """
        config = cls.get_config(name)
        
        # Set random seed for exact reproducibility
        random.seed(config.seed)
        
        # Create primitive sets
        action_pset = None
        assessment_pset = None
        
        if config.action_primitive:
            action_pset = PrimitiveSetFactory.create(
                config.action_primitive,
                ephemeral_prefix="rand_const_act"
            )
        
        if config.assessment_primitive:
            assessment_pset = PrimitiveSetFactory.create(
                config.assessment_primitive,
                ephemeral_prefix="rand_const_ass"
            )
        
        # Create toolbox
        factory = IndividualFactory()
        toolbox = factory.create_toolbox(
            action_pset=action_pset,
            assessment_pset=assessment_pset,
            num_trees=config.num_trees
        )
        
        return config, action_pset, assessment_pset, toolbox
    
    @classmethod
    def run_experiment(
        cls,
        name: str,
        verbose: bool = True,
        show_plots: bool = True
    ) -> ExperimentResult:
        """
        Run a complete experiment.
        
        Args:
            name: Experiment name
            verbose: Whether to print progress
            show_plots: Whether to display visualization
            
        Returns:
            Experiment results
        """
        # Create experiment components
        config, action_pset, assessment_pset, toolbox = cls.create_experiment(name)
        
        # Create evolutionary engine
        engine = EvolutionaryEngine(
            config=config,
            toolbox=toolbox,
            action_pset=action_pset,
            assessment_pset=assessment_pset
        )
        
        # Run evolution
        best_individual, history, control_fitness = engine.run(verbose=verbose)
        
        # Create result
        result = ExperimentResult(
            best_individual=best_individual,
            history=history,
            control_fitness=control_fitness,
            config=config
        )
        
        # Visualize if requested
        if show_plots:
            from .visualization import plot_dashboard
            plot_dashboard(
                best_individual,
                history,
                config,
                control_fitness
            )
        
        return result


# Convenience function for running experiments
def run_experiment(
    name: str,
    verbose: bool = True,
    show_plots: bool = True
) -> ExperimentResult:
    """
    Run an experiment by name.
    
    Args:
        name: Experiment name (e.g., "DR_mode1", "IR_mode3")
        verbose: Whether to print progress
        show_plots: Whether to display visualization
        
    Returns:
        Experiment results
        
    Example:
        >>> result = run_experiment("DR_mode1")
        >>> print(f"Best fitness: {result.history.max_igss[-1]}")
    """
    return ExperimentRegistry.run_experiment(name, verbose, show_plots)


def list_experiments() -> list:
    """List all available experiment names."""
    return ExperimentRegistry.list_experiments()


# ============================================================================
# Backward compatibility: Functions matching original script interfaces
# ============================================================================

def run_DR_mode1(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Direct Reciprocity Mode 1 (Action Rule evolution)."""
    return run_experiment("DR_mode1", verbose, show_plots)


def run_DR_mode2(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Direct Reciprocity Mode 2 (Assessment Rule evolution)."""
    return run_experiment("DR_mode2", verbose, show_plots)


def run_DR_mode3(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Direct Reciprocity Mode 3 (Co-evolution)."""
    return run_experiment("DR_mode3", verbose, show_plots)


def run_IR_mode1(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Indirect Reciprocity Mode 1 (Action Rule evolution)."""
    return run_experiment("IR_mode1", verbose, show_plots)


def run_IR_mode2_donor(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Indirect Reciprocity Mode 2 with donor standing."""
    return run_experiment("IR_mode2_donor", verbose, show_plots)


def run_IR_mode2_nodonor(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Indirect Reciprocity Mode 2 without donor standing."""
    return run_experiment("IR_mode2_nodonor", verbose, show_plots)


def run_IR_mode3(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Indirect Reciprocity Mode 3 (Co-evolution)."""
    return run_experiment("IR_mode3", verbose, show_plots)


def run_joint_mode1(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Joint DR+IR Mode 1 (Action Rule evolution)."""
    return run_experiment("jointDRIR_mode1", verbose, show_plots)


def run_joint_mode2(
    logic: str = "OR",
    verbose: bool = True,
    show_plots: bool = True
) -> ExperimentResult:
    """
    Run Joint DR+IR Mode 2 (Assessment evolution with AND/OR logic).
    
    Args:
        logic: "AND" or "OR" action logic
        verbose: Whether to print progress
        show_plots: Whether to display visualization
    """
    # Create custom config with specified logic
    from .config import ModelConfig, EvoConfig, PrimitiveConfig
    
    config = ExperimentConfig(
        name=f"jointDRIR_mode2_{logic}",
        mode="mode2",
        mechanism="joint",
        model=ModelConfig(action_logic=logic),
        evolution=EvoConfig(pop_size=60, max_gens=150, parsimony_tax=0.05),
        assessment_primitive=PrimitiveConfig.joint_assessment(),
        seed=421,
    )
    
    # Temporarily register and run
    ExperimentRegistry.register(config.name, config)
    return run_experiment(config.name, verbose, show_plots)


def run_joint_mode3(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Joint DR+IR Mode 3 (Total Co-evolution with 3 trees)."""
    return run_experiment("jointDRIR_mode3_threetrees", verbose, show_plots)


def run_money_mode1(verbose: bool = True, show_plots: bool = True) -> ExperimentResult:
    """Run Money Mode 1 (Closed economy with token accumulation)."""
    return run_experiment("MONEY_mode1_integer_exclusive_defenced", verbose, show_plots)
