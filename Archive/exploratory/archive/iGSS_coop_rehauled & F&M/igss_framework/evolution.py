"""
Evolutionary engine for genetic programming.

Implements the evolutionary algorithm for evolving cooperation rules
using DEAP with elitism and parsimony pressure.
"""

import random
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
from deap import tools, gp

from .config import ExperimentConfig, EvoConfig
from .models import CooperationModel
from .primitives import IndividualFactory, GeneticOperators


@dataclass
class EvolutionHistory:
    """Tracks evolutionary progress across generations."""
    max_igss: List[float]
    avg_igss: List[float]
    uc_scores: List[float]
    d_scores: List[float]
    fossil_record: Dict[int, str]
    
    @classmethod
    def create(cls) -> "EvolutionHistory":
        """Create empty history."""
        return cls(
            max_igss=[],
            avg_igss=[],
            uc_scores=[],
            d_scores=[],
            fossil_record={}
        )


class EvolutionaryEngine:
    """
    Evolutionary algorithm engine for evolving cooperation rules.
    
    Implements tournament selection, crossover, mutation, and elitism
    with parsimony pressure for bloat control.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        toolbox: Any,
        action_pset: Optional[Any] = None,
        assessment_pset: Optional[Any] = None,
    ):
        """
        Initialize the evolutionary engine.
        
        Args:
            config: Experiment configuration
            toolbox: Configured DEAP toolbox
            action_pset: Primitive set for action rules
            assessment_pset: Primitive set for assessment rules
        """
        self.config = config
        self.evo_config = config.evolution
        self.toolbox = toolbox
        self.action_pset = action_pset
        self.assessment_pset = assessment_pset
        
        # Setup genetic operators based on number of trees
        self._setup_operators()
    
    def _setup_operators(self):
        """Configure genetic operators for the number of trees."""
        num_trees = self.config.num_trees
        
        if num_trees == 1:
            # Single tree: standard operators
            self.toolbox.register("mate", gp.cxOnePoint)
            self.toolbox.register(
                "mutate",
                gp.mutUniform,
                expr=self.toolbox.expr,
                pset=self.action_pset or self.assessment_pset
            )
        elif num_trees == 2:
            # Two trees: custom operators
            self.toolbox.register("mate", GeneticOperators.cx_two_trees)
            self.toolbox.register(
                "mutate",
                GeneticOperators.mut_two_trees,
                toolbox=self.toolbox,
                pset_action=self.action_pset,
                pset_assessment=self.assessment_pset
            )
        elif num_trees == 3:
            # Three trees: custom operators
            self.toolbox.register("mate", GeneticOperators.cx_three_trees)
            self.toolbox.register(
                "mutate",
                GeneticOperators.mut_three_trees,
                toolbox=self.toolbox,
                pset_action=self.action_pset,
                pset_assessment=self.assessment_pset
            )
        
        # Selection with configurable tournament size
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.evo_config.tournament_size
        )
    
    def evaluate_individual(self, individual: Any) -> Tuple[float]:
        """
        Evaluate fitness of an individual.
        
        Compiles the GP tree(s) and runs multiple simulation trials,
        returning average payoff minus parsimony tax.
        
        Args:
            individual: GP individual to evaluate
            
        Returns:
            Tuple containing fitness value
        """
        # Compile rules based on number of trees
        if self.config.num_trees == 1:
            if self.config.mode == "mode1":
                func_action = self.toolbox.compile(expr=individual)
                func_assessment = None
            else:
                func_action = None
                func_assessment = self.toolbox.compile(expr=individual)
        elif self.config.num_trees == 2:
            func_action = self.toolbox.compile_action(expr=individual[0])
            func_assessment = self.toolbox.compile_assessment(expr=individual[1])
        elif self.config.num_trees == 3:
            func_action = self.toolbox.compile_action(expr=individual[0])
            func_assessment = None
        else:
            raise ValueError(f"Invalid num_trees: {self.config.num_trees}")
        
        # Run multiple evaluation trials
        tot_igss = tot_uc = tot_d = 0.0
        
        for _ in range(self.evo_config.evaluation_runs):
            if self.config.mechanism == "joint" and self.config.num_trees >= 2:
                # Joint mechanism needs separate mem and stand rules
                if self.config.num_trees == 2:
                    model = CooperationModel(
                        config=self.config,
                        igss_mem_rule=func_action,  # Actually mem rule
                        igss_stand_rule=func_assessment,
                    )
                else:  # 3 trees
                    func_mem = self.toolbox.compile_assessment(expr=individual[1])
                    func_stand = self.toolbox.compile_assessment(expr=individual[2])
                    model = CooperationModel(
                        config=self.config,
                        igss_action_rule=func_action,
                        igss_mem_rule=func_mem,
                        igss_stand_rule=func_stand,
                    )
            else:
                model = CooperationModel(
                    config=self.config,
                    igss_action_rule=func_action,
                    igss_assessment_rule=func_assessment,
                )
            
            fitness = model.run_simulation()
            tot_igss += fitness[self.config.igss_agent_type]
            tot_uc += fitness["Unconditional Cooperator"]
            tot_d += fitness["Defector"]
        
        # Store simulation results on individual
        individual.sim_igss = tot_igss / self.evo_config.evaluation_runs
        individual.sim_uc = tot_uc / self.evo_config.evaluation_runs
        individual.sim_d = tot_d / self.evo_config.evaluation_runs
        
        # Calculate parsimony tax
        if self.config.num_trees == 1:
            tree_size = len(individual)
        else:
            tree_size = sum(len(tree) for tree in individual)
        
        tax = tree_size * self.evo_config.parsimony_tax
        final_score = individual.sim_igss - tax
        
        return (final_score,)
    
    def get_control_baseline(self) -> float:
        """
        Calculate control baseline fitness.
        
        Runs hardcoded control strategy multiple times.
        
        Returns:
            Average control fitness
        """
        tot_control = 0.0
        
        for _ in range(self.evo_config.baseline_runs):
            model = CooperationModel(
                config=self.config,
                control_mode=True
            )
            fitness = model.run_simulation()
            tot_control += fitness[self.config.control_agent_type]
        
        return tot_control / self.evo_config.baseline_runs
    
    def run(self, verbose: bool = True) -> Tuple[Any, EvolutionHistory, float]:
        """
        Run the evolutionary algorithm.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_individual, history, control_baseline)
        """
        # Initialize population
        pop = self.toolbox.population(n=self.evo_config.pop_size)
        
        # Hall of Fame for elitism
        hof = tools.HallOfFame(1)
        
        # History tracking
        history = EvolutionHistory.create()
        
        # Get control baseline
        if verbose:
            print(f"Initializing {self.config.name}...")
        control_fitness = self.get_control_baseline()
        if verbose:
            print(f"Control Baseline established at: {control_fitness:.2f} average payoff.")
        
        # Register evaluation function
        self.toolbox.register("evaluate", self.evaluate_individual)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        hof.update(pop)
        
        # Evolution loop
        for gen in range(1, self.evo_config.max_gens + 1):
            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.evo_config.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.evo_config.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid_ind, map(self.toolbox.evaluate, invalid_ind)):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
            
            # Update Hall of Fame
            hof.update(pop)
            
            # Inject elite (elitism)
            pop[-1] = self.toolbox.clone(hof[0])
            
            # Track statistics
            best_in_gen = tools.selBest(pop, k=1)[0]
            history.max_igss.append(best_in_gen.sim_igss)
            history.avg_igss.append(np.mean([ind.sim_igss for ind in pop]))
            history.uc_scores.append(best_in_gen.sim_uc)
            history.d_scores.append(best_in_gen.sim_d)
            
            # Fossil record
            if gen % 10 == 0 or gen == 1:
                if self.config.num_trees == 1:
                    fossil = str(best_in_gen)
                elif self.config.num_trees == 2:
                    fossil = f"ACT: {best_in_gen[0]}  |  ASSESS: {best_in_gen[1]}"
                else:  # 3 trees
                    fossil = f"ACT: {best_in_gen[0]} | MEM: {best_in_gen[1]} | STAND: {best_in_gen[2]}"
                
                history.fossil_record[gen] = fossil
                
                if verbose:
                    gen_width = 3 if self.evo_config.max_gens >= 1000 else 2
                    print(f" > Gen {gen:0{gen_width}d} | "
                          f"iGSS: {best_in_gen.sim_igss:.1f} | "
                          f"UC: {best_in_gen.sim_uc:.1f} | "
                          f"D: {best_in_gen.sim_d:.1f}")
        
        return hof[0], history, control_fitness
