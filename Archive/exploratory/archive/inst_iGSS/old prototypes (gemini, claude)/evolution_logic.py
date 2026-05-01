import operator
import random
import numpy
from deap import base, creator, tools, gp
from model_logic import LogicModel

# --- NEW FUNCTIONS FOR THE AI ---
def safeDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

def if_then_else(input, output_if_true, output_if_false):
    return output_if_true if input > 0 else output_if_false

# --- SETUP DEAP ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
# Add our new logic blocks!
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(if_then_else, 3) # 3 inputs: (condition, true_val, false_val)

def random_constant():
    return random.choice([-1, 1])
pset.addEphemeralConstant("rand_const", random_constant)

toolbox = base.Toolbox()
# We allow trees to grow a bit deeper (max 3) to accommodate if/then structures
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate_rule(individual):
    rule_func = toolbox.compile(expr=individual)
    m = LogicModel(30, rule=rule_func)
    for _ in range(10): m.step()
    return m.calculate_fitness(),

toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

if __name__ == "__main__":
    pop = toolbox.population(n=100) # Increased population for more complexity
    print("Starting Evolution with Logic Gates...")
    
    # Standard evolution loop (simplified for display)
    for gen in range(5):
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5: toolbox.mate(child1, child2); del child1.fitness.values; del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.2: toolbox.mutate(mutant); del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)): ind.fitness.values = fit
        pop[:] = offspring
        print(f"Gen {gen}: Max Fitness {max([ind.fitness.values[0] for ind in pop])}")

    print("\nBest Rule Found:")
    print(tools.selBest(pop, 1)[0])