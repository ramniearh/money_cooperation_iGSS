import operator
import random
import numpy
from deap import base, creator, tools, gp
from model import CooperationModel, MODEL_CONFIG

# =============================================================================
# EVOLUTION CONFIGURATION
# =============================================================================
EVO_CONFIG = {
    "POP_SIZE": 20,       
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1  
}
# =============================================================================

def if_then(condition, output_if_true):
    if condition > 0: return output_if_true
    else: return 0

def random_constant():
    return random.choice([-1, 0, 1])

pset = gp.PrimitiveSet("MAIN", 3) 
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(if_then, 2) 
pset.addEphemeralConstant("rand_const", random_constant)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate_rule(individual):
    rule_func = toolbox.compile(expr=individual)
    
    total_fitness = 0
    for _ in range(3): 
        m = CooperationModel(igss_rule=rule_func)
        for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): 
            m.step()
        total_fitness += m.get_igss_fitness()
        
    average_fitness = total_fitness / 3
    tree_size = len(individual)  
    final_score = average_fitness - (tree_size * EVO_CONFIG["PARSIMONY_TAX"])
    
    return final_score, 

toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

if __name__ == "__main__":
    random.seed(42) 
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"]) 
    print("Starting iGSS Discovery...")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
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
    
    # Calculate the Theoretical Maximum Score
    benefit = MODEL_CONFIG["BENEFIT_TO_COST_RATIO"] * MODEL_CONFIG["COST"]
    net_profit_per_round = benefit - MODEL_CONFIG["COST"]
    theoretical_max = MODEL_CONFIG["NUM_ROUNDS"] * net_profit_per_round
    
    # Calculate final metrics and indexing
    final_max_fitness = best_rule.fitness.values[0]
    final_avg_fitness = numpy.mean([ind.fitness.values[0] for ind in pop])
    
    max_efficiency = (final_max_fitness / theoretical_max) * 100
    avg_efficiency = (final_avg_fitness / theoretical_max) * 100
    
    # -------------------------------------------------------------------------
    # IMPROVED LAB REPORT OUTPUT
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE: LAB REPORT")
    print("="*60)
    
    print("\nMODEL CONFIGURATION:")
    for key, value in MODEL_CONFIG.items():
        print(f"  > {key}: {value}")
    
    print("\niGSS ENGINE CONFIGURATION:")
    for key, value in EVO_CONFIG.items():
        print(f"  > {key}: {value}")
        
    print("\n" + "-"*60)
    print("RULE TRANSLATION KEY:")
    print(f"  ARG0 (Standing): Partner's reputation (1=Good, 0=Bad). Active: {MODEL_CONFIG['USE_STANDING']}")
    print(f"  ARG1 (Memory):   Is partner in blacklist? (1=Yes, 0=No). Active: {MODEL_CONFIG['USE_MEMORY']}")
    print(f"  ARG2 (Tokens):   Partner's token balance (Integer).      Active: {MODEL_CONFIG['USE_TOKENS']}")
    print(f"  Logic Gate:      Agent cooperates if Rule Output > 0")
    print("-"*60)
    print("="*60)
    print("FINAL PERFORMANCE METRICS:")
    print(f"  > Theoretical Max Score:   {theoretical_max:.2f}")
    print(f"  > Max Fitness (Best Rule): {final_max_fitness:.2f} ({max_efficiency:.1f}% Efficiency)")
    print(f"  > Avg Fitness (Final Gen): {final_avg_fitness:.2f} ({avg_efficiency:.1f}% Efficiency)")
    print(f"\nBEST RULE DISCOVERED: {best_rule}")
    print("="*60)