import operator
import random
import numpy
import matplotlib.pyplot as plt
import mesa
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (HIGH PRESSURE)
# =============================================================================
MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,  
    "NUM_UC": 10,    
    "NUM_D": 20,     # Increased Defectors to prevent the "Unconditional Love" trap
    "NUM_ROUNDS": 20 
}

EVO_CONFIG = {
    "POP_SIZE": 40,        
    "MAX_GENS": 50,         
    "PARSIMONY_TAX": 0.1  
}

# =============================================================================
# 2. DUAL-TREE GP SETUP (DEAP)
# =============================================================================
def if_then(condition, output_if_true):
    return output_if_true if condition > 0 else 0

def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# TREE 1: Action Rule (Input: Partner Standing)
pset_action = gp.PrimitiveSet("ActionRule", 1) 
pset_action.renameArguments(ARG0='PartnerStanding')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addPrimitive(if_then, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

# TREE 2: Assessment Rule (Inputs: Action, Helper Prior Standing, Recipient Prior Standing)
pset_assess = gp.PrimitiveSet("AssessRule", 3)
pset_assess.renameArguments(ARG0='Action', ARG1='HelperStanding', ARG2='RecipientStanding')
pset_assess.addPrimitive(operator.add, 2)
pset_assess.addPrimitive(operator.sub, 2)
pset_assess.addPrimitive(operator.mul, 2)
pset_assess.addPrimitive(if_then, 2)
pset_assess.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Individual is a LIST holding two separate GP trees
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
toolbox.register("expr_assess", gp.genHalfAndHalf, pset=pset_assess, min_=1, max_=3)
toolbox.register("tree_action", tools.initIterate, gp.PrimitiveTree, toolbox.expr_action)
toolbox.register("tree_assess", tools.initIterate, gp.PrimitiveTree, toolbox.expr_assess)

def init_dual_individual(icls):
    return icls([toolbox.tree_action(), toolbox.tree_assess()])

toolbox.register("individual", init_dual_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Custom Operators for Dual-Tree
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
        ind[1], = gp.mutUniform(ind[1], expr=toolbox.expr_assess, pset=pset_assess)
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
        self.standing = 1  # 1 = Good, 0 = Bad
        
    def evaluate_partner(self, partner, action_rule):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # iGSS Action Rule Evaluation
        decision_score = action_rule(partner.standing)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, action_rule, assessment_rule, config=MODEL_CONFIG):
        super().__init__()
        self.action_rule = action_rule
        self.assessment_rule = assessment_rule
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
            
        # MODE 3: Endogenous Assessment logic overrides hardcoded rules
        action_signal = 1 if cooperates else 0
        assessment_score = self.assessment_rule(action_signal, helper.standing, recipient.standing)
        helper.standing = 1 if assessment_score > 0 else 0

    def get_igss_fitness(self):
        igss_agents = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        if not igss_agents: return 0
        return sum(a.payoff for a in igss_agents) / len(igss_agents)

# =============================================================================
# 4. EVOLUTIONARY ENGINE LOOP
# =============================================================================
def evaluate_rules(individual, model_config, evo_config):
    func_action = gp.compile(expr=individual[0], pset=pset_action)
    func_assess = gp.compile(expr=individual[1], pset=pset_assess)
    
    total_fitness = 0
    for _ in range(3): 
        m = CooperationModel(action_rule=func_action, assessment_rule=func_assess, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): 
            m.step()
        total_fitness += m.get_igss_fitness()
        
    average_fitness = total_fitness / 3
    total_tree_size = len(individual[0]) + len(individual[1])
    final_score = average_fitness - (total_tree_size * evo_config["PARSIMONY_TAX"])
    return final_score, 

toolbox.register("evaluate", evaluate_rules, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"]) 
    history = {"max_fitness": [], "avg_fitness": [], "fossil_record": {}}
    print("Initializing Mode 3 (Co-Evolutionary Dual-Tree Engine)...")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
        
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
        for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
                
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        
        history["max_fitness"].append(max(fits))
        history["avg_fitness"].append(numpy.mean(fits))
        
        if gen % 10 == 0 or gen == 1:
            best_so_far = tools.selBest(pop, k=1)[0]
            # Compact formatting for dual-tree fossils
            history["fossil_record"][gen] = f"ACT: {str(best_so_far[0])} | ASSESS: {str(best_so_far[1])}"
            print(f" > Gen {gen:02d} | Max Fit: {max(fits):.2f}")

    best_individual = tools.selBest(pop, k=1)[0]
    return best_individual, history

# =============================================================================
# 5. VISUALIZATION & EXECUTION
# =============================================================================
def plot_dashboard(best_ind, history, model_config, title="Mode 3: Co-Evolutionary Institutional Discovery"):
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.8, 1.2]})
    
    # Calculate Theoretical Limits
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    theoretical_max_points = model_config["NUM_ROUNDS"] * (benefit - model_config["COST"])
    
    # Calculate Achievable Ceiling based on population
    total_agents = model_config["NUM_IGSS"] + model_config["NUM_UC"] + model_config["NUM_D"]
    cooperative_agents = model_config["NUM_IGSS"] + model_config["NUM_UC"]
    achievable_ceiling = (cooperative_agents / total_agents) * 100
    
    max_efficiency = [(score / theoretical_max_points) * 100 for score in history["max_fitness"]]
    avg_efficiency = [(score / theoretical_max_points) * 100 for score in history["avg_fitness"]]
    
    # Left Side: The Chart
    ax_plot.plot(max_efficiency, label='Max Efficiency', color='darkmagenta', linewidth=2)
    ax_plot.plot(avg_efficiency, label='Avg Efficiency', color='orchid', linestyle='--')
    
    # Plot the Achievable Ceiling Line
    ax_plot.axhline(y=achievable_ceiling, color='red', linestyle=':', linewidth=2, 
                    label=f'Achievable Max ({achievable_ceiling:.1f}%)')
    
    ax_plot.set_title(title)
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Efficiency (% of Total Possible Points)')
    ax_plot.set_ylim(0, 105) 
    ax_plot.legend()
    ax_plot.grid(True, alpha=0.3)

    # Right Side: The Report
    ax_text.axis('off') 
    
    fossil_gens = sorted(history["fossil_record"].keys())
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in fossil_gens])
    
    report_text = (
        f"--- BEST EVOLVED INSTITUTION ---\n"
        f"ACTION RULE:\n  {str(best_ind[0])}\n\n"
        f"ASSESSMENT RULE:\n  {str(best_ind[1])}\n\n"
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
    plot_dashboard(best_ind, history, MODEL_CONFIG)