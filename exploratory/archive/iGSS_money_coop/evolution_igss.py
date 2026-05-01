import operator
import random
import numpy
from deap import base, creator, tools, gp
from model_igss import IgssModel

# --- THE ALPHABET ---
def if_then_else(condition, out_true, out_false):
    return out_true if condition > 0 else out_false

def random_constant():
    return random.choice([-1, 0, 1])

# 3 Inputs: ARG0 (In Memory Flag), ARG1 (Reputation), ARG2 (Balance)
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
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
    # Run a few times to smooth out the stochastic matching
    for _ in range(5): 
        # 30 AI Agents vs 10 Hardcoded Defectors
        # Adjust bcr (Benefit-Cost Ratio) and liq (Liquidity) to shift the evolutionary pressure!
        m = IgssModel(N=40, num_defectors=10, rule=rule_func, bcr=5.0, liq=1.0)
        
        # Run for 50 rounds to let memory and reputation accumulate
        for _ in range(50): 
            m.step()
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 5
    
    # 10% Complexity Tax to favor the clean, classic heuristic rules
    tax = len(individual) * 0.1 
    return (average_fitness - tax), 

toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# --- EVOLUTION LOOP ---
if __name__ == "__main__":
    random.seed(42) 
    pop = toolbox.population(n=60) 
    print("Hunting for Reciprocity & Money Rules...")
    
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit
        
    for gen in range(1, 16): 
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
        print(f"Gen {gen:02d}: Max Fit = {max(fits):.2f}")

    best = tools.selBest(pop, k=1)[0]
    print("\nBest Heuristic:", best)