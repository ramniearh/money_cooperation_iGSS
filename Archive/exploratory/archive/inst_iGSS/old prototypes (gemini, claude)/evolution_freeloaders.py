import operator
import random
import numpy
from deap import base, creator, tools, gp
from model_freeloaders import FreeloaderModel # Importing the new model

# 1. SETUP DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 2. DEFINE THE "GENOME" ALPHABET
pset = gp.PrimitiveSet("MAIN", 3) 
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

def random_constant():
    return random.choice([-1, 1])
pset.addEphemeralConstant("rand_const", random_constant)

# 3. CONFIGURE THE EVOLUTION PROCESS
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate_rule_in_mesa(individual):
    rule_func = toolbox.compile(expr=individual)
    
    total_fitness = 0
    # Run 5 times instead of 3 to smooth out the randomness of meeting freeloaders
    for _ in range(5): 
        # 30 Total Agents, 5 of them are Freeloaders
        m = FreeloaderModel(30, 5, rule=rule_func)
        for _ in range(10):
            m.step()
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 5
    return average_fitness, 

toolbox.register("evaluate", evaluate_rule_in_mesa)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# 4. RUN THE ALGORITHM
if __name__ == "__main__":
    random.seed(42) 
    
    # Increased population and generations slightly to handle the harder problem
    pop = toolbox.population(n=60) 
    print("Starting Evolution with Freeloaders...")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(1, 8): 
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
        print(f"Generation {gen} - Max Fitness: {max(fits):.2f}, Avg Fitness: {numpy.mean(fits):.2f}")

    best_rule = tools.selBest(pop, k=1)[0]
    print("\nEvolution Complete!")
    print(f"Best Rule Discovered: {best_rule}")