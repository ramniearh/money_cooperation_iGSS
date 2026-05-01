"""
Genetic Programming primitive set definitions.

This module handles the creation and management of GP primitive sets
for evolving action and assessment rules.
"""

import operator
import random
from typing import List, Callable, Optional
from deap import gp, base, creator, tools

from .config import PrimitiveConfig


def random_constant() -> int:
    """
    Generate a random constant for GP ephemeral constants.
    
    Returns one of: -10, -1, 0, 1, 10
    """
    return random.choice([-10, -1, 0, 1, 10])


class PrimitiveSetFactory:
    """
    Factory for creating GP primitive sets.
    
    Creates primitive sets with arithmetic operations and ephemeral constants
    tailored to specific experiment configurations.
    """
    
    @staticmethod
    def create(config: PrimitiveConfig, ephemeral_prefix: str = "rand_const") -> gp.PrimitiveSet:
        """
        Create a primitive set from configuration.
        
        Args:
            config: Primitive configuration defining arguments and name
            ephemeral_prefix: Prefix for ephemeral constant naming
            
        Returns:
            Configured DEAP PrimitiveSet
        """
        pset = gp.PrimitiveSet(config.name, config.arity)
        
        # Rename arguments to meaningful names
        for i, name in enumerate(config.argument_names):
            pset.renameArguments(**{f"ARG{i}": name})
        
        # Add arithmetic primitives
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        
        # Add ephemeral constant
        pset.addEphemeralConstant(
            f"{ephemeral_prefix}_{config.name}", 
            random_constant
        )
        
        return pset


class IndividualFactory:
    """
    Factory for creating GP individuals.
    
    Handles the creation of single-tree and multi-tree individuals
    with proper DEAP registration.
    """
    
    def __init__(self):
        self._fitness_created = False
        self._individual_classes = {}
    
    def create_fitness_class(self, weights: tuple = (1.0,)) -> type:
        """
        Create or retrieve the FitnessMax class.
        
        Args:
            weights: Fitness weights (default maximizes single objective)
            
        Returns:
            Fitness class
        """
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=weights)
        return creator.FitnessMax
    
    def create_individual_class(self, num_trees: int) -> type:
        """
        Create or retrieve the Individual class for specified tree count.
        
        Args:
            num_trees: Number of trees in the individual (1, 2, or 3)
            
        Returns:
            Individual class
        """
        class_name = f"Individual_{num_trees}trees"
        
        if not hasattr(creator, class_name):
            fitness_class = self.create_fitness_class()
            
            if num_trees == 1:
                # Single tree: Individual is a PrimitiveTree
                creator.create(class_name, gp.PrimitiveTree, fitness=fitness_class)
            else:
                # Multiple trees: Individual is a list of PrimitiveTrees
                creator.create(class_name, list, fitness=fitness_class)
        
        return getattr(creator, class_name)
    
    def create_toolbox(
        self,
        action_pset: Optional[gp.PrimitiveSet],
        assessment_pset: Optional[gp.PrimitiveSet],
        num_trees: int,
        min_depth: int = 1,
        max_depth: int = 3,
    ) -> base.Toolbox:
        """
        Create a configured DEAP toolbox.
        
        Args:
            action_pset: Primitive set for action rules (or None)
            assessment_pset: Primitive set for assessment rules (or None)
            num_trees: Number of trees per individual
            min_depth: Minimum tree depth for initialization
            max_depth: Maximum tree depth for initialization
            
        Returns:
            Configured DEAP Toolbox
        """
        toolbox = base.Toolbox()
        individual_class = self.create_individual_class(num_trees)
        
        if num_trees == 1:
            # Single tree configuration
            pset = action_pset if action_pset else assessment_pset
            
            toolbox.register(
                "expr", 
                gp.genHalfAndHalf, 
                pset=pset, 
                min_=min_depth, 
                max_=max_depth
            )
            toolbox.register(
                "individual",
                tools.initIterate,
                individual_class,
                toolbox.expr
            )
            toolbox.register("compile", gp.compile, pset=pset)
            
        elif num_trees == 2:
            # Two trees: [action, assessment]
            toolbox.register(
                "expr_action",
                gp.genHalfAndHalf,
                pset=action_pset,
                min_=min_depth,
                max_=max_depth
            )
            toolbox.register(
                "expr_assessment",
                gp.genHalfAndHalf,
                pset=assessment_pset,
                min_=min_depth,
                max_=max_depth
            )
            
            def init_individual(container, func1, func2):
                return container([
                    gp.PrimitiveTree(func1()),
                    gp.PrimitiveTree(func2())
                ])
            
            toolbox.register(
                "individual",
                init_individual,
                individual_class,
                toolbox.expr_action,
                toolbox.expr_assessment
            )
            toolbox.register("compile_action", gp.compile, pset=action_pset)
            toolbox.register("compile_assessment", gp.compile, pset=assessment_pset)
            
        elif num_trees == 3:
            # Three trees: [action, memory_assessment, standing_assessment]
            toolbox.register(
                "expr_action",
                gp.genHalfAndHalf,
                pset=action_pset,
                min_=min_depth,
                max_=max_depth
            )
            toolbox.register(
                "expr_assessment",
                gp.genHalfAndHalf,
                pset=assessment_pset,
                min_=min_depth,
                max_=max_depth
            )
            
            def init_individual(container, func_act, func_ass):
                return container([
                    gp.PrimitiveTree(func_act()),
                    gp.PrimitiveTree(func_ass()),
                    gp.PrimitiveTree(func_ass())
                ])
            
            toolbox.register(
                "individual",
                init_individual,
                individual_class,
                toolbox.expr_action,
                toolbox.expr_assessment
            )
            toolbox.register("compile_action", gp.compile, pset=action_pset)
            toolbox.register("compile_assessment", gp.compile, pset=assessment_pset)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        return toolbox


class GeneticOperators:
    """
    Custom genetic operators for multi-tree individuals.
    """
    
    @staticmethod
    def cx_two_trees(ind1, ind2):
        """
        Crossover for two-tree individuals.
        
        Randomly selects one tree to crossover.
        """
        if random.random() < 0.5:
            ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
        else:
            ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
        return ind1, ind2
    
    @staticmethod
    def mut_two_trees(individual, toolbox, pset_action, pset_assessment):
        """
        Mutation for two-tree individuals.
        
        Randomly selects one tree to mutate.
        """
        if random.random() < 0.5:
            individual[0], = gp.mutUniform(
                individual[0],
                expr=toolbox.expr_action,
                pset=pset_action
            )
        else:
            individual[1], = gp.mutUniform(
                individual[1],
                expr=toolbox.expr_assessment,
                pset=pset_assessment
            )
        return individual,
    
    @staticmethod
    def cx_three_trees(ind1, ind2):
        """
        Crossover for three-tree individuals.
        
        Randomly selects one tree to crossover.
        """
        roll = random.random()
        if roll < 0.33:
            ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
        elif roll < 0.66:
            ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
        else:
            ind1[2], ind2[2] = gp.cxOnePoint(ind1[2], ind2[2])
        return ind1, ind2
    
    @staticmethod
    def mut_three_trees(individual, toolbox, pset_action, pset_assessment):
        """
        Mutation for three-tree individuals.
        
        Randomly selects one tree to mutate.
        """
        roll = random.random()
        if roll < 0.33:
            individual[0], = gp.mutUniform(
                individual[0],
                expr=toolbox.expr_action,
                pset=pset_action
            )
        elif roll < 0.66:
            individual[1], = gp.mutUniform(
                individual[1],
                expr=toolbox.expr_assessment,
                pset=pset_assessment
            )
        else:
            individual[2], = gp.mutUniform(
                individual[2],
                expr=toolbox.expr_assessment,
                pset=pset_assessment
            )
        return individual,
