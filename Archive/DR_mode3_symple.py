import operator
import random
import numpy
import matplotlib.pyplot as plt
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (3x10 Balanced Ecosystem)
# =============================================================================
MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,  
    "NUM_UC": 10,    
    "NUM_D": 10,     
    "NUM_ROUNDS": 100 
}

EVO_CONFIG = {
    "POP_SIZE": 40,        
    "MAX_GENS": 500,         
    "PARSIMONY_TAX": 0.1  
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
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.memory = set() 
        
    def evaluate_partner(self, partner, action_rule):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        in_memory_signal = 1 if partner.unique_id in self.memory else 0
        decision_score = action_rule(in_memory_signal)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, action_rule, memory_rule, config=MODEL_CONFIG):
        super().__init__()
        self.action_rule = action_rule
        self.memory_rule = memory_rule 
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        for _ in range(config["NUM_IGSS"]): CoopAgent(self, "iGSS-Agent")
        for _ in range(config["NUM_UC"]): CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): CoopAgent(self, "Defector")

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for i in range(0, len(agents) - 1, 2):
            agent_A = agents[i]
            agent_B = agents[i+1]

            a_coops = agent_A.evaluate_partner(agent_B, self.action_rule)
            b_coops = agent_B.evaluate_partner(agent_A, self.action_rule)

            self.resolve(helper=agent_A, recipient=agent_B, cooperates=a_coops)
            self.resolve(helper=agent_B, recipient=agent_A, cooperates=b_coops)
            
    def resolve(self, helper, recipient, cooperates):
        if cooperates:
            helper.payoff -= self.cost
            recipient.payoff += self.benefit
            
        partner_action = 1 if cooperates else 0
        partner_was_in_memory = 1 if helper.unique_id in recipient.memory else 0
        
        memory_score = self.memory_rule(partner_action, partner_was_in_memory)
        
        if memory_score > 0: recipient.memory.add(helper.unique_id)
        else:
            if helper.unique_id in recipient.memory: recipient.memory.remove(helper.unique_id)

    def get_all_fitnesses(self):
        igss = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        ucs = [a for a in self.agents if a.agent_type == "Unconditional Cooperator"]
        ds = [a for a in self.agents if a.agent_type == "Defector"]
        
        f_igss = sum(a.payoff for a in igss) / len(igss) if igss else 0
        f_uc = sum(a.payoff for a in ucs) / len(ucs) if ucs else 0
        f_d = sum(a.payoff for a in ds) / len(ds) if ds else 0
        return f_igss, f_uc, f_d

# =============================================================================
# 4. EVOLUTIONARY ENGINE LOOP
# =============================================================================
def evaluate_rules(individual, model_config, evo_config):
    func_action = gp.compile(expr=individual[0], pset=pset_action)
    func_memory = gp.compile(expr=individual[1], pset=pset_memory)
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    for _ in range(3): 
        m = CooperationModel(action_rule=func_action, memory_rule=func_memory, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        i_fit, u_fit, d_fit = m.get_all_fitnesses()
        tot_igss += i_fit; tot_uc += u_fit; tot_d += d_fit
        
    tax = (len(individual[0]) + len(individual[1])) * evo_config["PARSIMONY_TAX"]
    
    # Store the raw, untaxed scores on the individual so we can extract them for the plot later
    individual.sim_igss = tot_igss / 3
    individual.sim_uc = tot_uc / 3
    individual.sim_d = tot_d / 3
    
    final_score = individual.sim_igss - tax
    return final_score, 

toolbox.register("evaluate", evaluate_rules, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"]) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Mode 3 DR (Ecosystem Tracking Enabled)...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5: toolbox.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
        for mutant in offspring:
            if random.random() < 0.2: toolbox.mutate(mutant); del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)): ind.fitness.values = fit
                
        pop[:] = offspring
        
        # Logging
        best = tools.selBest(pop, k=1)[0]
        history["max_igss"].append(best.sim_igss)
        history["avg_igss"].append(numpy.mean([ind.sim_igss for ind in pop]))
        history["uc_scores"].append(best.sim_uc)
        history["d_scores"].append(best.sim_d)
        
        if gen % 10 == 0 or gen == 1:
            history["fossil_record"][gen] = f"ACT: {str(best[0])} | MEM: {str(best[1])}"
            print(f" > Gen {gen:02d} | iGSS: {best.sim_igss:.1f} | UC: {best.sim_uc:.1f} | D: {best.sim_d:.1f}")

    return tools.selBest(pop, k=1)[0], history

# =============================================================================
# 5. SYMPY SIMPLIFIER
# =============================================================================
def simplify_rule(gp_string, rule_type):
    mapping = {'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y}
    if rule_type == "action": mapping['PartnerInMemory'] = sp.Symbol('PartnerInMemory')
    elif rule_type == "memory":
        mapping['PartnerAction'] = sp.Symbol('PartnerAction')
        mapping['PartnerWasInMemory'] = sp.Symbol('PartnerWasInMemory')
    try: return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: return f"SymPy Error: {e}"

# =============================================================================
# 6. VISUALIZATION & EXECUTION
# =============================================================================
def plot_dashboard(best_ind, history, model_config):
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.5, 1.5]})
    
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    theoretical_max_points = model_config["NUM_ROUNDS"] * (benefit - model_config["COST"])
    
    # Calculate efficiencies for all three populations
    eff_igss_max = [(score / theoretical_max_points) * 100 for score in history["max_igss"]]
    eff_igss_avg = [(score / theoretical_max_points) * 100 for score in history["avg_igss"]]
    eff_uc = [(score / theoretical_max_points) * 100 for score in history["uc_scores"]]
    eff_d = [(score / theoretical_max_points) * 100 for score in history["d_scores"]]
    
    ax_plot.plot(eff_igss_max, label='Apex iGSS', color='darkorange', linewidth=2.5)
    ax_plot.plot(eff_uc, label='Unconditional Cooperators', color='mediumseagreen', linewidth=2)
    ax_plot.plot(eff_d, label='Defectors', color='crimson', linewidth=2)
    ax_plot.plot(eff_igss_avg, label='Avg iGSS', color='navajowhite', linestyle='--')
    
    ax_plot.set_title('Mode 3 DR: Multi-Population Ecological Dynamics')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Efficiency (% of Theoretical Utopia)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)

    ax_text.axis('off') 
    
    raw_action = str(best_ind[0])
    raw_memory = str(best_ind[1])
    simp_action = simplify_rule(raw_action, "action")
    simp_memory = simplify_rule(raw_memory, "memory")
    
    fossils_str = "\n".join([f" Gen {g:02d}:\n  {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    report_text = (
        f"--- BEST EVOLVED DR INSTITUTION ---\n"
        f"ACTION RULE:\n  Raw: {raw_action}\n  Sim: {simp_action}\n\n"
        f"MEMORY RULE:\n  Raw: {raw_memory}\n  Sim: {simp_memory}\n\n"
        f"--- ENVIRONMENT SETTINGS ---\n"
        f"iGSS: {model_config['NUM_IGSS']} | UC: {model_config['NUM_UC']} | Defectors: {model_config['NUM_D']}\n"
        f"Benefit: {model_config['BENEFIT_TO_COST_RATIO']} | Cost: {model_config['COST']} | Rounds: {model_config['NUM_ROUNDS']}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
    )
    
    ax_text.text(0.0, 0.95, report_text, fontsize=9, family='monospace', verticalalignment='top', transform=ax_text.transAxes, wrap=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_ind, history = run_evolution()
    
    print("\n" + "="*50)
    print("      MODE 3 DR FINAL LAB REPORT")
    print("="*50)
    print("--- ACTION RULE ---")
    print(f"Sim: {simplify_rule(str(best_ind[0]), 'action')}")
    print("--- MEMORY RULE ---")
    print(f"Sim: {simplify_rule(str(best_ind[1]), 'memory')}")
    print("==================================================")
    
    plot_dashboard(best_ind, history, MODEL_CONFIG)