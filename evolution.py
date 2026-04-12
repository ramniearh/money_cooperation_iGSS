import operator
import random
import numpy
from deap import base, creator, tools, gp
from model import CooperationModel, MODEL_CONFIG

EVO_CONFIG = {
    "POP_SIZE": 20,       
    "MAX_GENS": 50,         
    "PARSIMONY_TAX": 0.1  
}

def if_then(condition, output_if_true):
    if condition > 0: return output_if_true
    else: return 0

def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

pset = gp.PrimitiveSet("MAIN", 3) 
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(if_then, 2) 
pset.addEphemeralConstant("rand_const", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

def run_evolution(model_config=MODEL_CONFIG, evo_config=EVO_CONFIG):
    
    def evaluate_rule(individual):
        rule_func = toolbox.compile(expr=individual)
        total_fitness = 0
        for _ in range(3): 
            m = CooperationModel(igss_rule=rule_func, config=model_config)
            for _ in range(model_config["NUM_ROUNDS"]): 
                m.step()
            total_fitness += m.get_igss_fitness()
            
        average_fitness = total_fitness / 3
        tree_size = len(individual)  
        final_score = average_fitness - (tree_size * evo_config["PARSIMONY_TAX"])
        return final_score, 

    toolbox.register("evaluate", evaluate_rule)
    pop = toolbox.population(n=evo_config["POP_SIZE"]) 
    
    history = {
        "max_fitness": [],
        "avg_fitness": [],
        "fossil_record": {}
    }

    print("Initializing iGSS Evolutionary Engine...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(1, evo_config["MAX_GENS"] + 1): 
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
        history["max_fitness"].append(max(fits))
        history["avg_fitness"].append(numpy.mean(fits))
        
        if gen % 10 == 0:
            best_so_far = tools.selBest(pop, k=1)[0]
            history["fossil_record"][gen] = str(best_so_far)
            print(f" > Generation {gen:02d} evaluated. Max Fitness: {max(fits):.2f}")

    best_rule = tools.selBest(pop, k=1)[0]
    return best_rule, history, pop