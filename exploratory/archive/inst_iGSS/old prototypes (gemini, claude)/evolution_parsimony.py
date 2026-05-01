# =============================================================================
# FILE: evolution_parsimony.py
# PURPOSE: Teaching the AI to write clean, short, human-readable institutions.
# =============================================================================

import operator
import random
import numpy
from deap import base, creator, tools, gp
from model_core import CoreModel

# --- THE ALPHABET (Same as before) ---
def safeDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

def if_then_else(condition, output_if_true, output_if_false):
    if condition > 0: return output_if_true
    else: return output_if_false

def random_constant():
    return random.choice([-1, 0, 1])

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(if_then_else, 3) 
pset.addEphemeralConstant("rand_const", random_constant)

# --- DEAP SETUP ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# =============================================================================
# THE LESSON: THE PARSIMONY PRESSURE (COMPLEXITY TAX)
# =============================================================================
def evaluate_rule(individual):
    # 1. Compile the math tree so we can test it
    rule_func = toolbox.compile(expr=individual)
    
    # 2. Count how many "blocks" or words are in the AI's equation
    # len() is a basic Python function that counts the length of a list or string.
    equation_length = len(individual)
    
    total_fitness = 0
    for _ in range(5): 
        m = CoreModel(30, rule=rule_func)
        for _ in range(15): 
            m.step()
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 5
    
    # 3. APPLY THE TAX!
    # We multiply the length of the equation by 0.5. 
    # So an equation with 10 words loses 5 points. An equation with 50 words loses 25 points.
    tax_penalty = equation_length * 0.3
    
    # 4. Calculate the Final Score
    final_score = average_fitness - tax_penalty
    
    return final_score, 

toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# --- THE MAIN LOOP ---
if __name__ == "__main__":
    random.seed(42) 
    
    pop = toolbox.population(n=60) 
    print("Starting Evolution with Parsimony Pressure (Complexity Tax)...")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(1, 21): 
        
        # ELITISM LESSON! We are fixing the "Tragedy of Generation 12" here.
        # We find the absolute #1 best rule from the population.
        best_of_generation = tools.selBest(pop, k=1)[0]
        # We clone it and put it in a safe box, so it cannot be mutated or broken.
        elite_clone = toolbox.clone(best_of_generation)
        
        offspring = toolbox.select(pop, len(pop) - 1) # We select 1 less to make room for the Elite
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
                
        # We put the newly mutated children into the population...
        pop[:] = offspring
        # ...AND we add our bulletproof Elite clone back in at the very end!
        pop.append(elite_clone)
        
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Gen {gen}: Max Fitness = {max(fits):.2f} | Avg = {numpy.mean(fits):.2f}")

    best_rule = tools.selBest(pop, k=1)[0]
    print("\nEvolution Complete!")
    print(f"Best Rule Discovered: {best_rule}")
    print(f"Equation Length: {len(best_rule)} words")