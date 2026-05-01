import operator
import random
import numpy
import matplotlib.pyplot as plt
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (2 Tribes of 10 = 20 Agents)
# =============================================================================
MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 500,
    "COST": 1,
    "TRIBE_SIZE": 10,
    "NUM_TRIBES": 2,     # Changed to 2 Tribes
    "NUM_ROUNDS": 500 
}

EVO_CONFIG = {
    "MAX_GENS": 500,         
    "PARSIMONY_TAX": 0.05  
}

# =============================================================================
# 2. DUAL-TREE GP SETUP (PURE MATH ONLY)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

pset_action = gp.PrimitiveSet("ActionRule_DR", 1) 
pset_action.renameArguments(ARG0='PartnerInMemory')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

pset_memory = gp.PrimitiveSet("MemoryRule_DR", 2)
pset_memory.renameArguments(ARG0='PartnerAction', ARG1='PartnerWasInMemory')
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

def cx_dual(ind1, ind2):
    if random.random() < 0.5: ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else: ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mut_dual(ind):
    if random.random() < 0.5: ind[0], = gp.mutUniform(ind[0], expr=toolbox.expr_action, pset=pset_action)
    else: ind[1], = gp.mutUniform(ind[1], expr=toolbox.expr_memory, pset=pset_memory)
    return ind,

toolbox.register("mate", cx_dual)
toolbox.register("mutate", mut_dual)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. AGENT-BASED MODEL (MESA 3.0+)
# =============================================================================
class TribeAgent(mesa.Agent):
    def __init__(self, model, global_index, tribe_id, action_rule, memory_rule):
        super().__init__(model) 
        self.global_index = global_index # Maps back to the flat list of all compiled rules
        self.tribe_id = tribe_id         # Tribe A (0) or B (1)
        self.action_rule = action_rule
        self.memory_rule = memory_rule
        self.payoff = 0
        self.memory = set() 
        
    def evaluate_partner(self, partner):
        in_memory_signal = 1 if partner.unique_id in self.memory else 0
        decision_score = self.action_rule(in_memory_signal)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, flat_pop_rules, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        for i, rules in enumerate(flat_pop_rules):
            tribe_id = i // config["TRIBE_SIZE"] # Will be 0 or 1
            TribeAgent(self, global_index=i, tribe_id=tribe_id, action_rule=rules[0], memory_rule=rules[1])

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for i in range(0, len(agents) - 1, 2):
            agent_A = agents[i]
            agent_B = agents[i+1]

            a_coops = agent_A.evaluate_partner(agent_B)
            b_coops = agent_B.evaluate_partner(agent_A)

            self.resolve(helper=agent_A, recipient=agent_B, cooperates=a_coops)
            self.resolve(helper=agent_B, recipient=agent_A, cooperates=b_coops)
            
    def resolve(self, helper, recipient, cooperates):
        if cooperates:
            helper.payoff -= self.cost
            recipient.payoff += self.benefit
            
        partner_action = 1 if cooperates else 0
        partner_was_in_memory = 1 if helper.unique_id in recipient.memory else 0
        
        memory_score = recipient.memory_rule(partner_action, partner_was_in_memory)
        
        if memory_score > 0: recipient.memory.add(helper.unique_id)
        else:
            if helper.unique_id in recipient.memory: recipient.memory.remove(helper.unique_id)

# =============================================================================
# 4. EVOLUTIONARY ENGINE LOOP (MULTI-TRIBE)
# =============================================================================
def run_evolution():
    tribes = [toolbox.population(n=MODEL_CONFIG["TRIBE_SIZE"]) for _ in range(MODEL_CONFIG["NUM_TRIBES"])]
    history = {"max_T0": [], "max_T1": [], "avg_eco": [], "fossil_record": {}}
    
    print("Initializing Mode 4: 2-Tribe Co-Evolution...")
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        
        # 1. Compile all rules into one flat list for MESA
        compiled_rules = []
        for pop in tribes:
            for ind in pop:
                f_act = gp.compile(expr=ind[0], pset=pset_action)
                f_mem = gp.compile(expr=ind[1], pset=pset_memory)
                compiled_rules.append((f_act, f_mem))
                
        # 2. Run the Global Economy
        total_payoffs = [0] * (MODEL_CONFIG["TRIBE_SIZE"] * MODEL_CONFIG["NUM_TRIBES"])
        for _ in range(3):
            m = CooperationModel(flat_pop_rules=compiled_rules, config=MODEL_CONFIG)
            for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): m.step()
            for agent in m.agents:
                total_payoffs[agent.global_index] += agent.payoff
                
        # 3. Assign Fitness back to the distinct Tribes
        global_idx = 0
        all_fits_this_gen = []
        max_tribe_fits = []
        
        for pop in tribes:
            tribe_fits = []
            for ind in pop:
                avg_raw_score = total_payoffs[global_idx] / 3
                tax = (len(ind[0]) + len(ind[1])) * EVO_CONFIG["PARSIMONY_TAX"]
                final_fit = avg_raw_score - tax
                ind.fitness.values = (final_fit, )
                
                tribe_fits.append(final_fit)
                all_fits_this_gen.append(final_fit)
                global_idx += 1
            max_tribe_fits.append(max(tribe_fits))

        # 4. Logging and Reporting
        history["max_T0"].append(max_tribe_fits[0])
        history["max_T1"].append(max_tribe_fits[1])
        history["avg_eco"].append(numpy.mean(all_fits_this_gen))
        
        all_individuals_flat = [ind for pop in tribes for ind in pop]
        best_overall = tools.selBest(all_individuals_flat, k=1)[0]
        
        if gen % 10 == 0 or gen == 1:
            history["fossil_record"][gen] = f"ACT: {str(best_overall[0])} | MEM: {str(best_overall[1])}"
            print(f" > Gen {gen:02d} | Max Fits [T1:{max_tribe_fits[0]:.0f}, T2:{max_tribe_fits[1]:.0f}] | Avg: {numpy.mean(all_fits_this_gen):.0f}")

        # 5. Breed the Tribes
        if gen < EVO_CONFIG["MAX_GENS"]:
            for t_idx in range(MODEL_CONFIG["NUM_TRIBES"]):
                pop = tribes[t_idx]
                offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.5: toolbox.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
                for mutant in offspring:
                    if random.random() < 0.2: toolbox.mutate(mutant); del mutant.fitness.values
                tribes[t_idx][:] = offspring

    return best_overall, history

# =============================================================================
# 5. SYMPY SIMPLIFIER & DASHBOARD
# =============================================================================
def simplify_rule(gp_string, rule_type):
    mapping = {'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y}
    if rule_type == "action": mapping['PartnerInMemory'] = sp.Symbol('PartnerInMemory')
    elif rule_type == "memory": 
        mapping['PartnerAction'] = sp.Symbol('PartnerAction')
        mapping['PartnerWasInMemory'] = sp.Symbol('PartnerWasInMemory')
    try: return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: return f"SymPy Error: {e}"

def plot_dashboard(best_ind, history, model_config, evo_config):
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.5, 1.5]})
    
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    theoretical_max = model_config["NUM_ROUNDS"] * (benefit - model_config["COST"])
    
    def to_eff(scores): return [(s / theoretical_max) * 100 for s in scores]
    
    ax_plot.plot(to_eff(history["max_T0"]), label='Tribe A (Max)', color='royalblue', linewidth=2)
    ax_plot.plot(to_eff(history["max_T1"]), label='Tribe B (Max)', color='seagreen', linewidth=2)
    ax_plot.plot(to_eff(history["avg_eco"]), label='Global Average', color='gray', linestyle='--')
    
    ax_plot.set_title('Mode 4 DR: 2-Tribe Island Dynamics')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Efficiency (% of Theoretical Utopia)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)

    ax_text.axis('off') 
    raw_action = str(best_ind[0]); raw_memory = str(best_ind[1])
    
    fossils_str = "\n".join([f" Gen {g:02d}:\n  {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    # Restored experimental configurations to the print out
    report_text = (
        f"--- APEX GLOBAL SURVIVOR ---\n"
        f"ACTION RULE:\n  Raw: {raw_action}\n  Sim: {simplify_rule(raw_action, 'action')}\n\n"
        f"MEMORY RULE:\n  Raw: {raw_memory}\n  Sim: {simplify_rule(raw_memory, 'memory')}\n\n"
        f"--- ECOSYSTEM SETTINGS ---\n"
        f"Tribes: {model_config['NUM_TRIBES']} | Agents per Tribe: {model_config['TRIBE_SIZE']}\n"
        f"Benefit: {model_config['BENEFIT_TO_COST_RATIO']} | Cost: {model_config['COST']} | Rounds: {model_config['NUM_ROUNDS']}\n"
        f"Max Gens: {evo_config['MAX_GENS']} | Parsimony Tax: {evo_config['PARSIMONY_TAX']}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
    )
    
    ax_text.text(0.0, 0.95, report_text, fontsize=9, family='monospace', verticalalignment='top', transform=ax_text.transAxes, wrap=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_ind, history = run_evolution()
    
    raw_act = str(best_ind[0]); raw_mem = str(best_ind[1])
    print("\n" + "="*50)
    print("      MODE 4 (2-TRIBE) FINAL REPORT")
    print("="*50)
    print(f"ACTION: {simplify_rule(raw_act, 'action')}")
    print(f"MEMORY: {simplify_rule(raw_mem, 'memory')}")
    print("==================================================")
    
    plot_dashboard(best_ind, history, MODEL_CONFIG, EVO_CONFIG)