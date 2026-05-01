"""
Evolution Base Module for iGSS

Isolated DEAP setup - contains shared primitives and helper functions.
This is the ONLY file that touches DEAP's global state.

WARNING: DEAP uses global state (creator, toolbox). Don't refactor this unless
you absolutely have to - the global state is messy but functional.
"""

import operator
import random
from deap import base, creator, tools, gp


# =============================================================================
# PRIMITIVE FUNCTIONS
# =============================================================================

def if_then(condition, output_if_true):
    """
    Conditional primitive: returns output_if_true if condition > 0, else 0.
    
    This allows GP trees to implement conditional logic like:
    - if_then(ARG1, 10) -> cooperate if standing is good
    - if_then(ARG2, 5) -> cooperate if partner has tokens
    """
    return output_if_true if condition > 0 else 0


def random_constant():
    """
    Generate a random constant for GP ephemeral constants.
    Returns one of: -10, -1, 0, 1, 10
    """
    return random.choice([-10, -1, 0, 1, 10])


# =============================================================================
# PRIMITIVE SET FACTORIES
# =============================================================================

def create_action_pset(num_inputs=3, input_names=None):
    """
    Create a primitive set for action rules.
    
    Args:
        num_inputs: Number of input arguments (1, 2, or 3)
        input_names: Optional list of names for inputs (e.g., ["ARG0", "ARG1", "ARG2"])
                    If None, uses default ARG0, ARG1, etc.
    
    Returns:
        gp.PrimitiveSet: Configured primitive set
    """
    pset = gp.PrimitiveSet("ActionRule", num_inputs)
    
    # Rename arguments if names provided
    if input_names:
        for i, name in enumerate(input_names):
            pset.renameArguments(**{f"ARG{i}": name})
    
    # Add arithmetic primitives
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(if_then, 2)
    
    # Add ephemeral constants
    pset.addEphemeralConstant("rand_const_act", random_constant)
    
    return pset


def create_assessment_pset():
    """
    Create a primitive set for assessment rules.
    
    Assessment rules always take 3 inputs:
    - ARG0/Action: 1 if cooperated, 0 if defected
    - ARG1/HelperStanding: Helper's prior standing (1=Good, 0=Bad)
    - ARG2/RecipientStanding: Recipient's prior standing (1=Good, 0=Bad)
    
    Returns:
        gp.PrimitiveSet: Configured primitive set
    """
    pset = gp.PrimitiveSet("AssessRule", 3)
    pset.renameArguments(ARG0='Action', ARG1='HelperStanding', ARG2='RecipientStanding')
    
    # Add arithmetic primitives
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(if_then, 2)
    
    # Add ephemeral constants
    pset.addEphemeralConstant("rand_const_ass", random_constant)
    
    return pset


# =============================================================================
# FITNESS AND INDIVIDUAL SETUP
# =============================================================================

def setup_creator():
    """
    Set up DEAP's creator for single-tree individuals.
    
    This creates:
    - FitnessMax: Fitness class with single objective (maximize)
    - Individual: GP tree with fitness attribute
    
    Uses hasattr check to avoid re-creating if already exists.
    """
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def setup_creator_dual():
    """
    Set up DEAP's creator for dual-tree individuals (Mode 3).
    
    This creates:
    - FitnessMax: Fitness class with single objective (maximize)
    - Individual: LIST containing two GP trees [action_tree, assess_tree]
    
    Uses hasattr check to avoid re-creating if already exists.
    """
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)


# =============================================================================
# TOOLBOX FACTORIES
# =============================================================================

def create_toolbox_single(pset):
    """
    Create a DEAP toolbox for single-tree evolution.
    
    Args:
        pset: Primitive set to use
        
    Returns:
        base.Toolbox: Configured toolbox
    """
    toolbox = base.Toolbox()
    
    # Tree generation
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Compilation
    toolbox.register("compile", gp.compile, pset=pset)
    
    return toolbox


def create_toolbox_dual(pset_action, pset_assess):
    """
    Create a DEAP toolbox for dual-tree evolution (Mode 3).
    
    Args:
        pset_action: Primitive set for action rules
        pset_assess: Primitive set for assessment rules
        
    Returns:
        base.Toolbox: Configured toolbox with dual-tree operators
    """
    toolbox = base.Toolbox()
    
    # Tree generation for each type
    toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
    toolbox.register("expr_assess", gp.genHalfAndHalf, pset=pset_assess, min_=1, max_=3)
    toolbox.register("tree_action", tools.initIterate, gp.PrimitiveTree, toolbox.expr_action)
    toolbox.register("tree_assess", tools.initIterate, gp.PrimitiveTree, toolbox.expr_assess)
    
    # Individual initialization (list of two trees)
    def init_dual_individual(icls):
        return icls([toolbox.tree_action(), toolbox.tree_assess()])
    
    toolbox.register("individual", init_dual_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Custom crossover (randomly pick which tree to crossover)
    def cx_dual(ind1, ind2):
        if random.random() < 0.5:
            ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
        else:
            ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
        return ind1, ind2
    
    # Custom mutation (randomly pick which tree to mutate)
    def mut_dual(ind):
        if random.random() < 0.5:
            ind[0], = gp.mutUniform(ind[0], expr=toolbox.expr_action, pset=pset_action)
        else:
            ind[1], = gp.mutUniform(ind[1], expr=toolbox.expr_assess, pset=pset_assess)
        return ind,
    
    toolbox.register("mate", cx_dual)
    toolbox.register("mutate", mut_dual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Compilation (returns tuple of compiled functions)
    def compile_dual(individual):
        return (
            gp.compile(expr=individual[0], pset=pset_action),
            gp.compile(expr=individual[1], pset=pset_assess)
        )
    
    toolbox.register("compile", compile_dual)
    
    return toolbox


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate_single_tree(individual, toolbox, model_config, evo_config, 
                         action_rule=None, assessment_rule=None):
    """
    Evaluate a single-tree individual.
    
    Args:
        individual: DEAP individual (GP tree)
        toolbox: DEAP toolbox with compile function
        model_config: Model configuration dict
        evo_config: Evolution configuration dict
        action_rule: Optional action rule (for Mode 2)
        assessment_rule: Optional assessment rule (for Mode 1)
        
    Returns:
        tuple: (fitness_score,) - DEAP expects tuple
    """
    from core_abm import CooperationModel
    
    # Compile the individual
    if action_rule is None:
        # Mode 1: Individual IS the action rule
        func = toolbox.compile(expr=individual)
        rule_action = func
        rule_assess = None
    else:
        # Mode 2: Individual IS the assessment rule
        func = toolbox.compile(expr=individual)
        rule_action = None
        rule_assess = func
    
    # Run multiple simulations and average
    total_fitness = 0
    num_trials = 3
    for _ in range(num_trials):
        model = CooperationModel(
            config=model_config,
            action_rule=rule_action,
            assessment_rule=rule_assess
        )
        for _ in range(model_config["NUM_ROUNDS"]):
            model.step()
        total_fitness += model.get_igss_fitness()
    
    average_fitness = total_fitness / num_trials
    
    # Apply parsimony penalty
    tree_size = len(individual)
    final_score = average_fitness - (tree_size * evo_config["PARSIMONY_TAX"])
    
    return final_score,


def evaluate_dual_tree(individual, toolbox, model_config, evo_config):
    """
    Evaluate a dual-tree individual (Mode 3).
    
    Args:
        individual: DEAP individual (list of two GP trees)
        toolbox: DEAP toolbox with compile function
        model_config: Model configuration dict
        evo_config: Evolution configuration dict
        
    Returns:
        tuple: (fitness_score,) - DEAP expects tuple
    """
    from core_abm import CooperationModel
    
    # Compile both trees
    func_action, func_assess = toolbox.compile(individual)
    
    # Run multiple simulations and average
    total_fitness = 0
    num_trials = 3
    for _ in range(num_trials):
        model = CooperationModel(
            config=model_config,
            action_rule=func_action,
            assessment_rule=func_assess
        )
        for _ in range(model_config["NUM_ROUNDS"]):
            model.step()
        total_fitness += model.get_igss_fitness()
    
    average_fitness = total_fitness / num_trials
    
    # Apply parsimony penalty (sum of both tree sizes)
    total_tree_size = len(individual[0]) + len(individual[1])
    final_score = average_fitness - (total_tree_size * evo_config["PARSIMONY_TAX"])
    
    return final_score,


# =============================================================================
# EVOLUTIONARY LOOP (Shared Components)
# =============================================================================

def run_evolution_loop(toolbox, evo_config, evaluate_fn, dual_tree=False):
    """
    Run the main evolutionary loop.
    
    Args:
        toolbox: DEAP toolbox with genetic operators
        evo_config: Evolution configuration dict
        evaluate_fn: Function to evaluate individuals
        dual_tree: Whether this is dual-tree evolution (Mode 3)
        
    Returns:
        tuple: (best_individual, history_dict, final_population)
    """
    import numpy as np
    
    pop = toolbox.population(n=evo_config["POP_SIZE"])
    
    history = {
        "max_fitness": [],
        "avg_fitness": [],
        "fossil_record": {}
    }
    
    # Initial evaluation
    fitnesses = list(map(evaluate_fn, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Main loop
    for gen in range(1, evo_config["MAX_GENS"] + 1):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < evo_config.get("CX_PROB", 0.5):
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < evo_config.get("MUT_PROB", 0.2):
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(evaluate_fn, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        pop[:] = offspring
        
        # Record statistics
        fits = [ind.fitness.values[0] for ind in pop]
        history["max_fitness"].append(max(fits))
        history["avg_fitness"].append(np.mean(fits))
        
        # Fossil record every 10 generations
        if gen % 10 == 0 or gen == 1:
            best_so_far = tools.selBest(pop, k=1)[0]
            if dual_tree:
                history["fossil_record"][gen] = f"ACT: {str(best_so_far[0])} | ASSESS: {str(best_so_far[1])}"
            else:
                history["fossil_record"][gen] = str(best_so_far)
            print(f" > Gen {gen:02d} | Max Fit: {max(fits):.2f}")
    
    best_individual = tools.selBest(pop, k=1)[0]
    return best_individual, history, pop
