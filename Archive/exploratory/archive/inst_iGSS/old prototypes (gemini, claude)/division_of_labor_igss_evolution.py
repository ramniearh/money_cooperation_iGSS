import operator
import random
import numpy
from deap import base, creator, tools, gp
from division_of_labor_igss_toy_model import ToyModel 

# --- PRIMITIVE SET DEFINITIONS ---
def safeDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

def if_then_else(condition, output_if_true, output_if_false):
    return output_if_true if condition > 0 else output_if_false

def random_constant():
    return random.choice([-1, 0, 1])

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(if_then_else, 3) 
pset.addEphemeralConstant("rand_const", random_constant)

# --- DEAP CONFIGURATION ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# --- EVALUATION AND FITNESS ---
def evaluate_rule(individual):
    rule_func = toolbox.compile(expr=individual)
    equation_length = len(individual)
    
    total_fitness = 0
    # Average across 5 runs to smooth stochastic interactions
    for _ in range(5): 
        m = ToyModel(N=30, num_freeloaders=5, rule=rule_func)
        for _ in range(20): # 20 ticks guarantees starvation without trade
            m.step()
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 5
    
    # Parsimony Pressure: Penalize complex rules to simulate bounded rationality
    tax_penalty = equation_length * 0.3
    final_score = average_fitness - tax_penalty
    
    return final_score, 

toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# --- EVOLUTIONARY LOOP ---
if __name__ == "__main__":
    random.seed(42) 
    pop = toolbox.population(n=60) 
    print("Starting Evolution: Scarcity + Freeloaders + Parsimony...")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(1, 21): 
        # Elitism: Preserve the best rule across generations
        best_of_generation = tools.selBest(pop, k=1)[0]
        elite_clone = toolbox.clone(best_of_generation)
        
        offspring = toolbox.select(pop, len(pop) - 1) 
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
        pop.append(elite_clone) 
        
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Gen {gen:02d}: Max Fitness = {max(fits):.2f} | Avg = {numpy.mean(fits):.2f}")

    best_rule = tools.selBest(pop, k=1)[0]
    print("\nEvolution Complete.")
    print(f"Best Heuristic Discovered: {best_rule}")
    print(f"Node Count: {len(best_rule)}")