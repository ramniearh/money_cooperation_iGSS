# =============================================================================
# FILE 2: evolution_core.py
# PURPOSE: The Genetic Programming engine. It writes code, tests it, and evolves it.
# =============================================================================

import operator
import random
import numpy
from deap import base, creator, tools, gp
from model_core import CoreModel

# -----------------------------------------------------------------------------
# PART 1: DEFINING THE ALPHABET (The Primitive Set)
# GP LESSON: We have to tell the AI what words it is allowed to use to write rules.
# -----------------------------------------------------------------------------
def safeDiv(left, right):
    """If the AI tries to divide by zero, return 1 instead of crashing Python."""
    try: return left / right
    except ZeroDivisionError: return 1

def if_then_else(condition, output_if_true, output_if_false):
    """
    Logic Gate. 
    If the condition is > 0, do the first thing. Else, do the second thing.
    """
    if condition > 0:
        return output_if_true
    else:
        return output_if_false

def random_constant():
    """Allows the AI to randomly insert the number -1, 0, or 1 into equations."""
    return random.choice([-1, 0, 1])

# We create a "Primitive Set" named "MAIN" that takes 3 inputs (ARG0, ARG1, ARG2)
pset = gp.PrimitiveSet("MAIN", 3)

# We add our building blocks to the set. The '2' or '3' means how many inputs it takes.
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(if_then_else, 3) 
pset.addEphemeralConstant("rand_const", random_constant)

# -----------------------------------------------------------------------------
# PART 2: SETTING UP DEAP (The Framework)
# -----------------------------------------------------------------------------
# We tell DEAP we want to MAXIMIZE our fitness score (weights=1.0). 
# If we wanted to minimize it, we'd use weights=(-1.0,).
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# EXTENSION IDEA: To prevent "Bloat" (massive, unreadable equations), we limit 
# the maximum depth of the trees to 3 levels (max_=3).
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# This translates the DEAP tree into a real Python function we can run.
toolbox.register("compile", gp.compile, pset=pset)

# -----------------------------------------------------------------------------
# PART 3: THE EVALUATION LOOP (The Test)
# -----------------------------------------------------------------------------
def evaluate_rule(individual):
    """
    DEAP passes a single math equation to this function. 
    We plug it into Mesa, run the simulation, and return the final score.
    """
    rule_func = toolbox.compile(expr=individual)
    
    total_fitness = 0
    # Run the simulation 5 times and average it. 
    # Why? Because meetings are random! A good rule might just get unlucky once.
    for _ in range(5): 
        # 30 Agents. We pass the rule_func to them.
        m = CoreModel(30, rule=rule_func)
        
        # Run the economy for 15 ticks (days).
        for _ in range(15): 
            m.step()
            
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 5
    return average_fitness, # DEAP requires this to be a tuple, hence the comma

# We tell DEAP to use our custom evaluation function
toolbox.register("evaluate", evaluate_rule)
# Tournament size 3 means it picks 3 random rules, and the highest score gets to breed.
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# -----------------------------------------------------------------------------
# PART 4: THE MAIN EVENT (Running the Evolution)
# -----------------------------------------------------------------------------
# This line ensures the code only runs if you run this file directly.
if __name__ == "__main__":
    random.seed(42) # Keeps the randomness predictable so you can replicate results
    
    # We create 60 random equations to start.
    pop = toolbox.population(n=60) 
    print("Starting the Core Evolution...")
    
    # Evaluate Generation 0
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    # We are going to run this for 20 Generations! It will take a little longer.
    for gen in range(1, 51): 
        
        # 1. SELECT the winners from the previous generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # 2. MATE (Crossover) - 50% chance to swap branches with another winner
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values # Delete old score, the kid needs to be tested
                del child2.fitness.values

        # 3. MUTATE - 20% chance to randomly change a piece of math
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # 4. TEST THE NEW CHILDREN
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
                
        # 5. REPLACE the old generation with the new generation
        pop[:] = offspring
        
        # Print the progress to the terminal
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Gen {gen}: Max Fitness = {max(fits):.2f} | Avg Fitness = {numpy.mean(fits):.2f}")

    # The end! Find the single best equation and print it.
    best_rule = tools.selBest(pop, k=1)[0]
    print("\nEvolution Complete!")
    print("Best Rule Discovered:", best_rule)
    