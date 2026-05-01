# =============================================================================
# FILE 2: evolution_core.py (BORROWED CODE, NOT VALIDATED)
# =============================================================================

import operator
import random
import numpy
from deap import base, creator, tools, gp

from model import DivisionLaborModel 

# -----------------------------------------------------------------------------
# PART 1: DEFINING THE ALPHABET
# -----------------------------------------------------------------------------
def if_then_else(condition, output_if_true, output_if_false):
    if condition > 0:
        return output_if_true
    else:
        return output_if_false

def random_constant():
    return random.choice([-1, 0, 1])

pset = gp.PrimitiveSet("MAIN", 2) 

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(if_then_else, 3) 
pset.addEphemeralConstant("rand_const", random_constant)

# -----------------------------------------------------------------------------
# PART 2: SETTING UP DEAP
# -----------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# -----------------------------------------------------------------------------
# PART 3: THE EVALUATION LOOP (With Parsimony Pressure)
# -----------------------------------------------------------------------------
def evaluate_rule(individual):
    rule_func = toolbox.compile(expr=individual)
    
    # 1. Run the economic simulation as normal
    total_fitness = 0
    for _ in range(5): 
        m = DivisionLaborModel(30, rule=rule_func)
        
        for _ in range(15): 
            m.step()
            
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 5
    
    # 2. CALCULATE THE PENALTY
    tree_size = len(individual)  # Counts every math operation and argument
    penalty_weight = 0.01        # We debit 0.01 points for every piece of code
    
    # 3. Apply the tax
    final_score = average_fitness - (tree_size * penalty_weight)
    
    return final_score, 

toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# -----------------------------------------------------------------------------
# PART 4: THE MAIN EVENT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42) 
    
    pop = toolbox.population(n=60) 
    print("Starting the Core Evolution...")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(1, 21): 
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values 
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
                
        pop[:] = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Gen {gen}: Max Fitness = {max(fits):.2f} | Avg Fitness = {numpy.mean(fits):.2f}")

    best_rule = tools.selBest(pop, k=1)[0]
    print("\nEvolution Complete!")
    print("Best Rule Discovered:", best_rule)