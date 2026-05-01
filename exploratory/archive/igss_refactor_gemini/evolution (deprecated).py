import operator
import random
import uuid
import numpy
from deap import base, creator, tools, gp
from core import CooperationModel, DEFAULT_CONFIG

EVO_CONFIG = {
    "POP_SIZE": 40,       
    "MAX_GENS": 50,         
    "PARSIMONY_TAX": 0.1  
}

# DEAP Global Registry (Safe to run once)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def if_then(condition, output_if_true):
    return output_if_true if condition > 0 else 0

def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

def create_action_pset(config):
    active_args = sum([config["USE_MEMORY"], config["USE_STANDING"], config["USE_TOKENS"]])
    active_args = max(1, active_args) 
    
    # FIX: Unique name bypasses DEAP's global caching in interactive IDEs
    pset_name = f"ActionRule_{uuid.uuid4().hex[:6]}"
    pset = gp.PrimitiveSet(pset_name, active_args)
    
    arg_idx = 0
    if config["USE_MEMORY"]:
        pset.renameArguments(**{f"ARG{arg_idx}": "Is_In_Memory"})
        arg_idx += 1
    if config["USE_STANDING"]:
        pset.renameArguments(**{f"ARG{arg_idx}": "Partner_Standing"})
        arg_idx += 1
    if config["USE_TOKENS"]:
        pset.renameArguments(**{f"ARG{arg_idx}": "Partner_Tokens"})

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(if_then, 2)
    
    try: pset.addEphemeralConstant("rand_const_act", random_constant)
    except Exception: pass 
        
    return pset

def wrap_action_rule(gp_tree, pset, config):
    compiled_func = gp.compile(expr=gp_tree, pset=pset)
    def action_wrapper(memory, standing, tokens):
        args = []
        if config["USE_MEMORY"]: args.append(memory)
        if config["USE_STANDING"]: args.append(standing)
        if config["USE_TOKENS"]: args.append(tokens)
        if not args: args.append(0)
        return compiled_func(*args)
    return action_wrapper

def evaluate_action_rule(individual, pset, model_config, evo_config):
    action_func = wrap_action_rule(individual, pset, model_config)
    total_fitness = 0
    
    for _ in range(3): 
        m = CooperationModel(config=model_config, action_rule=action_func)
        for _ in range(model_config["NUM_ROUNDS"]): 
            m.step()
        total_fitness += m.get_igss_fitness()
        
    average_fitness = total_fitness / 3
    final_score = average_fitness - (len(individual) * evo_config["PARSIMONY_TAX"])
    return final_score, 

def run_evolution(model_config=DEFAULT_CONFIG, evo_config=EVO_CONFIG):
    pset = create_action_pset(model_config)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("evaluate", evaluate_action_rule, pset=pset, model_config=model_config, evo_config=evo_config)

    pop = toolbox.population(n=evo_config["POP_SIZE"]) 
    history = {"max_fitness": [], "avg_fitness": [], "fossil_record": {}}

    print(f"\nInitializing iGSS Evolutionary Engine...")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
        
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
        for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
                
        pop[:] = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        history["max_fitness"].append(max(fits))
        history["avg_fitness"].append(numpy.mean(fits))
        
        if gen % 10 == 0 or gen == 1:
            best_so_far = tools.selBest(pop, k=1)[0]
            history["fossil_record"][gen] = str(best_so_far)
            print(f" > Gen {gen:02d} evaluated. Max Fitness: {max(fits):.2f}")

    best_rule = tools.selBest(pop, k=1)[0]
    return best_rule, history, pop