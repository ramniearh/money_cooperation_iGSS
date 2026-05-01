#$## TEMP< NOPT PRO
# 
import operator
import random
import numpy
import matplotlib.pyplot as plt
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (IR 3-Tribe Model)
# =============================================================================
MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 25,
    "COST": 1,
    "TRIBE_SIZE": 10,
    "NUM_TRIBES": 3,
    "NUM_ROUNDS": 500 
}

EVO_CONFIG = {
    "MAX_GENS": 500,         
    "PARSIMONY_TAX": 0.02  
}

# =============================================================================
# 2. GP SETUP (Indirect Reciprocity Primitives)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# ACTION RULE: Inputs are now self-reputation and partner-reputation
pset_action = gp.PrimitiveSet("ActionRule_IR", 2) 
pset_action.renameArguments(ARG0='MyReputation', ARG1='PartnerReputation')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

# ASSESSMENT RULE: How society updates a RECIPIENT's reputation based on an action
# Inputs: ActionTaken (1/0), RecipientOldRep, HelperOldRep
pset_memory = gp.PrimitiveSet("AssessmentRule_IR", 3)
pset_memory.renameArguments(ARG0='ActionTaken', ARG1='RecipientRep', ARG2='HelperRep')
pset_memory.addPrimitive(operator.add, 2)
pset_memory.addPrimitive(operator.sub, 2)
pset_memory.addPrimitive(operator.mul, 2)
pset_memory.addEphemeralConstant("rand_const_mem", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
toolbox.register("expr_memory", gp.genHalfAndHalf, pset=pset_memory, min_=1, max_=3)
toolbox.register("tree_action", tools.initIterate, gp.PrimitiveTree, toolbox.expr_action)
toolbox.register("tree_memory", tools.initIterate, gp.PrimitiveTree, toolbox.expr_memory)

def init_dual_individual(icls):
    return icls([toolbox.tree_action(), toolbox.tree_memory()])

toolbox.register("individual", init_dual_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", gp.cxOnePoint) # Simplified for stability

def mut_dual(ind):
    if random.random() < 0.5: ind[0], = gp.mutUniform(ind[0], expr=toolbox.expr_action, pset=pset_action)
    else: ind[1], = gp.mutUniform(ind[1], expr=toolbox.expr_memory, pset=pset_memory)
    return ind,

toolbox.register("mutate", mut_dual)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. MESA MODEL (The Global Reputation Ledger)
# =============================================================================
class IRAgent(mesa.Agent):
    def __init__(self, model, global_index, tribe_id, action_rule, assessment_rule):
        super().__init__(model) 
        self.global_index = global_index
        self.tribe_id = tribe_id
        self.action_rule = action_rule
        self.assessment_rule = assessment_rule
        self.payoff = 0
        self.reputation = 0 # Starts at neutral

    def decide(self, partner):
        # Decision based on public reputation, not private memory
        score = self.action_rule(self.reputation, partner.reputation)
        return score > 0

class IRModel(mesa.Model):
    def __init__(self, flat_pop_rules, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        for i, rules in enumerate(flat_pop_rules):
            tribe_id = i // config["TRIBE_SIZE"]
            IRAgent(self, i, tribe_id, rules[0], rules[1])

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for i in range(0, len(agents) - 1, 2):
            a_idx, b_idx = i, i+1
            agent_A = agents[a_idx]; agent_B = agents[b_idx]

            # Decisions
            a_coops = agent_A.decide(agent_B)
            b_coops = agent_B.decide(agent_A)

            # Resolution and Reputation Update
            self.resolve_ir(agent_A, agent_B, a_coops)
            self.resolve_ir(agent_B, agent_A, b_coops)
            
    def resolve_ir(self, helper, recipient, cooperates):
        if cooperates:
            helper.payoff -= self.cost
            recipient.payoff += self.benefit
            
        action_val = 1 if cooperates else 0
        
        # In IR, the SOCIETY (represented by the helper's tribe rule) 
        # updates the helper's reputation based on their action toward the recipient
        new_rep_score = helper.assessment_rule(action_val, recipient.reputation, helper.reputation)
        
        # Normalize reputation to binary for simplicity (Good/Bad)
        helper.reputation = 1 if new_rep_score > 0 else 0

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def run_evolution():
    tribes = [toolbox.population(n=MODEL_CONFIG["TRIBE_SIZE"]) for _ in range(MODEL_CONFIG["NUM_TRIBES"])]
    history = {"max_T0": [], "max_T1": [], "max_T2": [], "avg_eco": [], "fossil_record": {}}
    
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        compiled_rules = []
        for pop in tribes:
            for ind in pop:
                compiled_rules.append((gp.compile(ind[0], pset_action), gp.compile(ind[1], pset_memory)))
                
        total_payoffs = [0] * (MODEL_CONFIG["TRIBE_SIZE"] * MODEL_CONFIG["NUM_TRIBES"])
        for _ in range(3):
            m = IRModel(compiled_rules)
            for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): m.step()
            for a in m.agents: total_payoffs[a.global_index] += a.payoff
                
        global_idx = 0
        all_fits = []
        tribe_maxes = []
        for pop in tribes:
            t_fits = []
            for ind in pop:
                fit = (total_payoffs[global_idx]/3) - (len(ind[0])+len(ind[1]))*EVO_CONFIG["PARSIMONY_TAX"]
                ind.fitness.values = (fit,)
                t_fits.append(fit); all_fits.append(fit)
                global_idx += 1
            tribe_maxes.append(max(t_fits))

        history["max_T0"].append(tribe_maxes[0]); history["max_T1"].append(tribe_maxes[1]); history["max_T2"].append(tribe_maxes[2])
        history["avg_eco"].append(numpy.mean(all_fits))
        
        best_overall = tools.selBest([ind for t in tribes for ind in t], 1)[0]
        if gen % 10 == 0 or gen == 1:
            history["fossil_record"][gen] = f"ACT: {best_overall[0]} | ASSESS: {best_overall[1]}"
            print(f"Gen {gen} | Tribe Maxes: {[round(m) for m in tribe_maxes]}")

        if gen < EVO_CONFIG["MAX_GENS"]:
            for i in range(len(tribes)):
                tribes[i] = list(map(toolbox.clone, toolbox.select(tribes[i], len(tribes[i]))))
                # Apply crossover/mutation per tribe... (Standard DEAP loop omitted for brevity, same as DR)
                for c1, c2 in zip(tribes[i][::2], tribes[i][1::2]):
                    if random.random() < 0.5: gp.cxOnePoint(c1[0], c2[0]); gp.cxOnePoint(c1[1], c2[1]); del c1.fitness.values; del c2.fitness.values
                for mutant in tribes[i]:
                    if random.random() < 0.2: toolbox.mutate(mutant); del mutant.fitness.values

    return best_overall, history

# =============================================================================
# 5. SYMPY & PLOT
# =============================================================================
def simplify_ir(gp_str, type):
    if type == "action":
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 
                   'MyReputation': sp.Symbol('MyRep'), 'PartnerReputation': sp.Symbol('PartRep')}
    else:
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 
                   'ActionTaken': sp.Symbol('Act'), 'RecipientRep': sp.Symbol('RecRep'), 'HelperRep': sp.Symbol('HlpRep')}
    return sp.simplify(eval(gp_str, {"__builtins__": {}}, mapping))

def plot_ir(best, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.plot(history["max_T0"], label="Tribe A"); ax1.plot(history["max_T1"], label="Tribe B"); ax1.plot(history["max_T2"], label="Tribe C")
    ax1.plot(history["avg_eco"], label="Global Avg", linestyle="--", color="black")
    ax1.set_title("Indirect Reciprocity: The Gossip Economy"); ax1.legend()
    
    report = (f"--- APEX IR SURVIVOR ---\nACTION: {simplify_ir(str(best[0]), 'action')}\nASSESS: {simplify_ir(str(best[1]), 'assess')}\n\n"
              f"CONFIG: Tribes=3, TribeSize=10, Rounds={MODEL_CONFIG['NUM_ROUNDS']}\nParsimony={EVO_CONFIG['PARSIMONY_TAX']}")
    ax2.text(0, 1, report, transform=ax2.transAxes, va='top', family='monospace')
    ax2.axis('off'); plt.show()

if __name__ == "__main__":
    best, history = run_evolution()
    plot_ir(best, history)