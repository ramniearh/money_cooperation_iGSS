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

# =============================================================================
# 1. DEAP GLOBAL SETUP (Universal List Architecture)
# =============================================================================
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Individuals are always LISTS. Mode 1/2: length 1. Mode 3: length 2.
    creator.create("Individual", list, fitness=creator.FitnessMax)

def if_then(condition, output_if_true):
    return output_if_true if condition > 0 else 0

def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# =============================================================================
# 2. UNIVERSAL PSET FACTORIES
# =============================================================================
def create_action_pset(config):
    active_args = max(1, sum([config["USE_MEMORY"], config["USE_STANDING"], config["USE_TOKENS"]]))
    pset = gp.PrimitiveSet(f"Action_{uuid.uuid4().hex[:6]}", active_args)
    
    arg_idx = 0
    if config["USE_MEMORY"]: pset.renameArguments(**{f"ARG{arg_idx}": "Is_In_Memory"}); arg_idx += 1
    if config["USE_STANDING"]: pset.renameArguments(**{f"ARG{arg_idx}": "Partner_Standing"}); arg_idx += 1
    if config["USE_TOKENS"]: pset.renameArguments(**{f"ARG{arg_idx}": "Partner_Tokens"})

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(if_then, 2)
    try: pset.addEphemeralConstant("rand_act", random_constant)
    except Exception: pass 
    return pset

def create_assess_pset(config):
    pset_name = f"Assess_{uuid.uuid4().hex[:6]}"
    if config["USE_STANDING"]: # IR
        pset = gp.PrimitiveSet(pset_name, 3)
        pset.renameArguments(ARG0='Action', ARG1='Helper_Standing', ARG2='Recipient_Standing')
    elif config["USE_MEMORY"]: # DR
        pset = gp.PrimitiveSet(pset_name, 2)
        pset.renameArguments(ARG0='Action', ARG1='Is_In_Memory')
    elif config["USE_TOKENS"]: # Money
        pset = gp.PrimitiveSet(pset_name, 3)
        pset.renameArguments(ARG0='Action', ARG1='Helper_Tokens', ARG2='Recipient_Tokens')
    else:
        pset = gp.PrimitiveSet(pset_name, 1)

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(if_then, 2)
    try: pset.addEphemeralConstant("rand_ass", random_constant)
    except Exception: pass
    return pset

# =============================================================================
# 3. UNIVERSAL WRAPPERS (Translating Math to MESA Physics)
# =============================================================================
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

def wrap_assess_rule(gp_tree, pset, config):
    compiled_func = gp.compile(expr=gp_tree, pset=pset)
    
    def assess_wrapper(helper, recipient, action_signal):
        if config["USE_STANDING"]: # IR Update
            score = compiled_func(action_signal, helper.standing, recipient.standing)
            helper.standing = 1 if score > 0 else 0
            
        elif config["USE_MEMORY"]: # DR Update
            is_in_mem = 1 if helper.unique_id in recipient.memory else 0
            score = compiled_func(action_signal, is_in_mem)
            if score > 0: recipient.memory.add(helper.unique_id)
            elif helper.unique_id in recipient.memory: recipient.memory.remove(helper.unique_id)
                
        elif config["USE_TOKENS"]: # Money Update
            score = compiled_func(action_signal, helper.tokens, recipient.tokens)
            if score > 0 and recipient.tokens > 0:
                recipient.tokens -= 1
                helper.tokens += 1
    return assess_wrapper

# Hardcoded Action Rules for Mode 2
FIXED_ACTIONS = {
    "IR": lambda mem, stand, tok: stand,               
    "DR": lambda mem, stand, tok: -1 if mem > 0 else 1,
    "MONEY": lambda mem, stand, tok: tok               
}

# =============================================================================
# 4. UNIVERSAL ENGINE
# =============================================================================
def evaluate_rules(individual, psets, model_config, evo_config):
    mode = model_config.get("EVO_MODE", 1)
    
    action_func = None
    assess_func = None
    
    if mode == 1:
        action_func = wrap_action_rule(individual[0], psets["action"], model_config)
    elif mode == 2:
        # Determine which fixed rule to use
        if model_config["USE_STANDING"]: fixed_key = "IR"
        elif model_config["USE_MEMORY"]: fixed_key = "DR"
        else: fixed_key = "MONEY"
        
        action_func = FIXED_ACTIONS[fixed_key]
        assess_func = wrap_assess_rule(individual[0], psets["assess"], model_config)
    elif mode == 3:
        action_func = wrap_action_rule(individual[0], psets["action"], model_config)
        assess_func = wrap_assess_rule(individual[1], psets["assess"], model_config)
    
    total_fitness = 0
    for _ in range(3): 
        m = CooperationModel(config=model_config, action_rule=action_func, assessment_rule=assess_func)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        total_fitness += m.get_igss_fitness()
        
    average_fitness = total_fitness / 3
    
    # Calculate parsimony based on how many trees we evolved
    total_tree_size = sum(len(tree) for tree in individual)
    final_score = average_fitness - (total_tree_size * evo_config["PARSIMONY_TAX"])
    return final_score, 

def run_universal_engine(model_config=DEFAULT_CONFIG, evo_config=EVO_CONFIG):
    mode = model_config.get("EVO_MODE", 1)
    print(f"\nInitializing iGSS Universal Engine (Mode {mode})...")
    
    toolbox = base.Toolbox()
    psets = {}
    
    # Setup Psets and Tree Generators based on Mode
    if mode == 1:
        psets["action"] = create_action_pset(model_config)
        toolbox.register("expr_act", gp.genHalfAndHalf, pset=psets["action"], min_=1, max_=3)
        toolbox.register("tree_act", tools.initIterate, gp.PrimitiveTree, toolbox.expr_act)
        def init_ind(icls): return icls([toolbox.tree_act()])
        
    elif mode == 2:
        psets["assess"] = create_assess_pset(model_config)
        toolbox.register("expr_ass", gp.genHalfAndHalf, pset=psets["assess"], min_=1, max_=3)
        toolbox.register("tree_ass", tools.initIterate, gp.PrimitiveTree, toolbox.expr_ass)
        def init_ind(icls): return icls([toolbox.tree_ass()])
        
    elif mode == 3:
        psets["action"] = create_action_pset(model_config)
        psets["assess"] = create_assess_pset(model_config)
        toolbox.register("expr_act", gp.genHalfAndHalf, pset=psets["action"], min_=1, max_=3)
        toolbox.register("expr_ass", gp.genHalfAndHalf, pset=psets["assess"], min_=1, max_=3)
        toolbox.register("tree_act", tools.initIterate, gp.PrimitiveTree, toolbox.expr_act)
        toolbox.register("tree_ass", tools.initIterate, gp.PrimitiveTree, toolbox.expr_ass)
        def init_ind(icls): return icls([toolbox.tree_act(), toolbox.tree_ass()])

    toolbox.register("individual", init_ind, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Universal Operators for Lists
    def list_crossover(ind1, ind2):
        idx = 0 if mode in [1, 2] else random.choice([0, 1])
        ind1[idx], ind2[idx] = gp.cxOnePoint(ind1[idx], ind2[idx])
        return ind1, ind2

    def list_mutate(ind):
        idx = 0 if mode in [1, 2] else random.choice([0, 1])
        active_expr = toolbox.expr_act if (mode == 1 or (mode == 3 and idx == 0)) else toolbox.expr_ass
        active_pset = psets["action"] if (mode == 1 or (mode == 3 and idx == 0)) else psets["assess"]
        ind[idx], = gp.mutUniform(ind[idx], expr=active_expr, pset=active_pset)
        return ind,

    toolbox.register("mate", list_crossover)
    toolbox.register("mutate", list_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_rules, psets=psets, model_config=model_config, evo_config=evo_config)

    pop = toolbox.population(n=evo_config["POP_SIZE"]) 
    history = {"max_fitness": [], "avg_fitness": [], "fossil_record": {}}

    # Generational Loop
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
        
    for gen in range(1, evo_config["MAX_GENS"] + 1): 
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values; del child2.fitness.values

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
            # Format fossil string based on mode
            if mode == 1: f_str = f"ACT: {str(best_so_far[0])}"
            elif mode == 2: f_str = f"ASSESS: {str(best_so_far[0])}"
            else: f_str = f"ACT: {str(best_so_far[0])} | ASSESS: {str(best_so_far[1])}"
            
            history["fossil_record"][gen] = f_str
            print(f" > Gen {gen:02d} evaluated. Max Fitness: {max(fits):.2f}")

    best_individual = tools.selBest(pop, k=1)[0]
    return best_individual, history, pop