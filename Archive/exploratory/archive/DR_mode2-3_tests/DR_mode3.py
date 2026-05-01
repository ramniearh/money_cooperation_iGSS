import operator
import random
import numpy
import matplotlib.pyplot as plt
import mesa
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (DR High Pressure)
# =============================================================================
MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,  
    "NUM_UC": 10,    
    "NUM_D": 20,     # High defectors to stress-test memory mechanics
    "NUM_ROUNDS": 100 
}

EVO_CONFIG = {
    "POP_SIZE": 40,        
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1  
}

# =============================================================================
# 2. DUAL-TREE GP SETUP (DEAP) - DIRECT RECIPROCITY
# =============================================================================
def if_then(condition, output_if_true):
    return output_if_true if condition > 0 else 0

def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# TREE 1: Action Rule (Input: Is Partner in Memory?)
# NOTE: 1 = In Memory (Danger), 0 = Not in Memory (Safe)
pset_action = gp.PrimitiveSet("ActionRule_DR", 1) 
pset_action.renameArguments(ARG0='PartnerInMemory')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addPrimitive(if_then, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

# TREE 2: Memory Update Rule (Inputs: Partner's Just-Played Action, Were they already in memory?)
# OUTPUT: > 0 means Add/Keep in Memory (Grudge), <= 0 means Remove (Forgive)
pset_memory = gp.PrimitiveSet("MemoryRule_DR", 2)
pset_memory.renameArguments(ARG0='PartnerAction', ARG1='PartnerWasInMemory')
pset_memory.addPrimitive(operator.add, 2)
pset_memory.addPrimitive(operator.sub, 2)
pset_memory.addPrimitive(operator.mul, 2)
pset_memory.addPrimitive(if_then, 2)
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
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mut_dual(ind):
    if random.random() < 0.5:
        ind[0], = gp.mutUniform(ind[0], expr=toolbox.expr_action, pset=pset_action)
    else:
        ind[1], = gp.mutUniform(ind[1], expr=toolbox.expr_memory, pset=pset_memory)
    return ind,

toolbox.register("mate", cx_dual)
toolbox.register("mutate", mut_dual)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. AGENT-BASED MODEL (MESA)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.memory = set() # DR uses personal memory sets
        
    def evaluate_partner(self, partner, action_rule):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # iGSS DR Action Rule: Check if partner is in personal memory
        in_memory_signal = 1 if partner.unique_id in self.memory else 0
        decision_score = action_rule(in_memory_signal)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, action_rule, memory_rule, config=MODEL_CONFIG):
        super().__init__()
        self.action_rule = action_rule
        self.memory_rule = memory_rule # This replaces "assessment_rule"
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
        # 1. Economic Exchange
        if cooperates:
            helper.payoff -= self.cost
            recipient.payoff += self.benefit
            
        # 2. DR Memory Update (The "Assessment")
        # How does the recipient process the helper's action?
        partner_action = 1 if cooperates else 0
        partner_was_in_memory = 1 if helper.unique_id in recipient.memory else 0
        
        memory_score = self.memory_rule(partner_action, partner_was_in_memory)
        
        if memory_score > 0:
            # Grudge: Add to (or keep in) memory
            recipient.memory.add(helper.unique_id)
        else:
            # Forgive: Remove from memory if they are in it
            if helper.unique_id in recipient.memory:
                recipient.memory.remove(helper.unique_id)

    def get_igss_fitness(self):
        igss_agents = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        if not igss_agents: return 0
        return sum(a.payoff for a in igss_agents) / len(igss_agents)

# =============================================================================
# 4. EVOLUTIONARY ENGINE & EXECUTION 
# (Boilerplate execution logic truncated for brevity, same as mode3_IR)
# =============================================================================
def evaluate_rules(individual, model_config, evo_config):
    func_action = gp.compile(expr=individual[0], pset=pset_action)
    func_memory = gp.compile(expr=individual[1], pset=pset_memory)
    
    total_fitness = 0
    for _ in range(3): 
        m = CooperationModel(action_rule=func_action, memory_rule=func_memory, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        total_fitness += m.get_igss_fitness()
        
    final_score = (total_fitness / 3) - ((len(individual[0]) + len(individual[1])) * evo_config["PARSIMONY_TAX"])
    return final_score, 

toolbox.register("evaluate", evaluate_rules, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"]) 
    history = {"max_fitness": [], "avg_fitness": [], "fossil_record": {}}
    
    print("Initializing Mode 3 DR (Co-Evolving Action and Memory/Forgiveness)...")
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
        fits = [ind.fitness.values[0] for ind in pop]
        history["max_fitness"].append(max(fits)); history["avg_fitness"].append(numpy.mean(fits))
        
        if gen % 10 == 0 or gen == 1:
            best = tools.selBest(pop, k=1)[0]
            history["fossil_record"][gen] = f"ACT: {str(best[0])} | MEM: {str(best[1])}"
            print(f" > Gen {gen:02d} | Max Fit: {max(fits):.2f}")

    return tools.selBest(pop, k=1)[0], history

# =============================================================================
# 5. VISUALIZATION & EXECUTION
# =============================================================================
def plot_dashboard(best_ind, history, model_config, title="Mode 3 DR: Co-Evolutionary Institutional Discovery"):
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.8, 1.2]})
    
    # Calculate Theoretical Limits
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    theoretical_max_points = model_config["NUM_ROUNDS"] * (benefit - model_config["COST"])
    
    max_efficiency = [(score / theoretical_max_points) * 100 for score in history["max_fitness"]]
    avg_efficiency = [(score / theoretical_max_points) * 100 for score in history["avg_fitness"]]
    
    # Left Side: The Chart
    ax_plot.plot(max_efficiency, label='Max Efficiency', color='darkorange', linewidth=2)
    ax_plot.plot(avg_efficiency, label='Avg Efficiency', color='navajowhite', linestyle='--')
    
    ax_plot.set_title(title)
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Efficiency (% of Total Possible Points)')
    
    # We don't cap ylim at 100 because DR agents against defectors might briefly go negative
    # or score differently depending on the exact mix. 
    ax_plot.legend()
    ax_plot.grid(True, alpha=0.3)

    # Right Side: The Report
    ax_text.axis('off') 
    
    # Format the fossil record to be readable
    fossil_gens = sorted(history["fossil_record"].keys())
    fossils_str = "\n".join([f" Gen {g:02d}:\n  {history['fossil_record'][g]}" for g in fossil_gens])
    
    report_text = (
        f"--- BEST EVOLVED DR INSTITUTION ---\n"
        f"ACTION RULE:\n  {str(best_ind[0])}\n\n"
        f"MEMORY RULE:\n  {str(best_ind[1])}\n\n"
        f"--- ENVIRONMENT SETTINGS ---\n"
        f"iGSS: {model_config['NUM_IGSS']} | UC: {model_config['NUM_UC']} | Defectors: {model_config['NUM_D']}\n"
        f"Benefit: {model_config['BENEFIT_TO_COST_RATIO']} | Cost: {model_config['COST']} | Rounds: {model_config['NUM_ROUNDS']}\n\n"
        f"--- FOSSIL RECORD ---\n"
        f"{fossils_str}\n"
    )
    
    ax_text.text(0.0, 0.95, report_text, fontsize=9, family='monospace', 
                 verticalalignment='top', transform=ax_text.transAxes, wrap=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_ind, history = run_evolution()
    
    # Print to terminal
    print("\n--- BEST EVOLVED DR INSTITUTION ---")
    print(f"ACTION RULE: {best_ind[0]} \n[ARG0=PartnerInMemory]")
    print(f"MEMORY RULE: {best_ind[1]} \n[ARG0=PartnerAction, ARG1=PartnerWasInMemory]")
    
    # Launch Dashboard
    plot_dashboard(best_ind, history, MODEL_CONFIG)