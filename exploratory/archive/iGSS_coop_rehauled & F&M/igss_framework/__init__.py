"""
IGSS Framework: Integrated Genetic Programming for Social Simulation

A unified framework for evolutionary experiments on cooperation mechanisms,
including Direct Reciprocity (DR), Indirect Reciprocity (IR), Joint mechanisms,
and Money-based cooperation.

This package provides exact functional equivalence to the original 11 experimental
scripts while offering a modular, extensible architecture suitable for academic
publication.

Author: [Your Name]
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"

from .config import ExperimentConfig, ModelConfig, EvoConfig
from .experiments import ExperimentRegistry, run_experiment
from .evolution import EvolutionaryEngine
from .visualization import plot_dashboard, plot_tree

__all__ = [
    "ExperimentConfig",
    "ModelConfig", 
    "EvoConfig",
    "ExperimentRegistry",
    "run_experiment",
    "EvolutionaryEngine",
    "plot_dashboard",
    "plot_tree",
]
