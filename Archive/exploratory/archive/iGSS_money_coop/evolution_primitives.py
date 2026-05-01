import operator
import random
import numpy
from deap import base, creator, tools, gp
from model_primitives import PrimitiveModel

# --- ALPHABET ---
def safeDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

def if_then_else(condition, out_true, out_false):
    return out_true if condition > 0 else out_false

def random_constant():
    return random.choice([-1, 0, 1])

# 4 Inputs: Global_Given, Global_Received, Local_Given, Local_Received
pset = gp.PrimitiveSet("MAIN", 4)
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

# --- EVALUATION ---
def evaluate_rule(individual):
    rule_func = toolbox.compile(expr=individual)
    
    total_fitness = 0
    for _ in range(5): 
        # 30 AI Agents vs 10 Defectors
        # Adjust the bcr (Benefit-to-Cost Ratio) here to change evolutionary pressure!
        m = PrimitiveModel(N=40, num_defectors=10, rule=rule_func, bcr=5.0)
        
        # 50 interactions per agent to build up a rich action history
        for _ in range(50): 
            m.step()
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 5
    
    # 20% Complexity Tax - Encourages elegant, publishable sociology rules
    tax = len(individual) * 0.2 
    return (average_fitness - tax), 

toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# --- MAIN LOOP ---
if __name__ == "__main__":
    random.seed(42) 
    pop = toolbox.population(n=60) 
    print("Hunting for First-Principles Sociology...")
    print("ARG0: Global Given | ARG1: Global Received | ARG2: Local Given | ARG3: Local Received")
    
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit
        
    for gen in range(1, 21): 
        elite = toolbox.clone(tools.selBest(pop, k=1)[0])
        
        offspring = toolbox.select(pop, len(pop) - 1) 
        offspring = list(map(toolbox.clone, offspring))
        
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(c1, c2)
                del c1.fitness.values; del c2.fitness.values
        for m in offspring:
            if random.random() < 0.2:
                toolbox.mutate(m)
                del m.fitness.values
                
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
                
        pop[:] = offspring + [elite]
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Gen {gen:02d}: Max Fit = {max(fits):.2f} | Avg = {numpy.mean(fits):.2f}")

    best = tools.selBest(pop, k=1)[0]
    print("\nEvolution Complete.")
    print("Best Equation:", best)