import operator
import random
import numpy
from deap import base, creator, tools, gp
from model_debt import DebtModel  # Importing from the new file!

# 1. SETUP DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 2. DEFINE THE "GENOME" ALPHABET
# Set to 3 to accept my_supply(ARG0), partner_supply(ARG1), and partner_debt(ARG2)
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
    """Takes a generated rule, plugs it into Mesa, and returns the score."""
    rule_func = toolbox.compile(expr=individual)
    
    total_fitness = 0
    for _ in range(3):
        m = DebtModel(30, rule=rule_func)
        for _ in range(10):
            m.step()
        total_fitness += m.calculate_fitness()
        
    average_fitness = total_fitness / 3
    return average_fitness, 

toolbox.register("evaluate", evaluate_rule_in_mesa)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# 4. RUN THE ALGORITHM
if __name__ == "__main__":
    random.seed(42) 
    
    pop = toolbox.population(n=50)
    print("Starting Evolution with Debt Primitive...")
    
    # Evaluate the entire first generation
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    # Run for 5 Generations
    for gen in range(1, 6): 
        # Select the best and clone them
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Mutate and cross over
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate the new children
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
                
        # Replace the old population
        pop[:] = offspring
        
        # Print progress
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Generation {gen} - Max Fitness: {max(fits):.2f}, Avg Fitness: {numpy.mean(fits):.2f}")

    # Show the best rule found
    best_rule = tools.selBest(pop, k=1)[0]
    print("\nEvolution Complete!")
    print(f"Best Rule Discovered: {best_rule}")