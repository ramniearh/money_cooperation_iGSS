import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (Mode 2 / 3 Benchmark)
# =============================================================================
MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 100,
    "INITIAL_TOKENS": 1
}

EVO_CONFIG = {
    "POP_SIZE": 25,
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1  
}

# =============================================================================
# 2. GP SETUP (Single Tree - Supply Curve)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

pset = gp.PrimitiveSet("ActionRule_Money", 1) 
pset.renameArguments(ARG0='MyTokens')
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addEphemeralConstant("rand_const", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. MESA MODEL (The Fiat Economy)
# =============================================================================
class MoneyAgent(mesa.Agent):
    def __init__(self, model, agent_type, action_rule=None):
        super().__init__(model) 
        self.agent_type = agent_type
        self.action_rule = action_rule
        self.payoff = 0                             
        self.tokens = MODEL_CONFIG["INITIAL_TOKENS"] 
        
    def will_sell_help(self):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        return self.action_rule(self.tokens) > 0

class MoneyModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        for _ in range(config["NUM_IGSS"]): MoneyAgent(self, "iGSS-Agent", igss_rule)
        for _ in range(config["NUM_UC"]): MoneyAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): MoneyAgent(self, "Defector")

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for i in range(0, len(agents) - 1, 2):
            agent_A = agents[i]
            agent_B = agents[i+1]

            # A requests help from B
            if agent_A.tokens >= 1 and agent_B.will_sell_help():
                agent_A.tokens -= 1
                agent_B.tokens += 1
                agent_A.payoff += self.benefit
                agent_B.payoff -= self.cost

            # B requests help from A
            if agent_B.tokens >= 1 and agent_A.will_sell_help():
                agent_B.tokens -= 1
                agent_A.tokens += 1
                agent_B.payoff += self.benefit
                agent_A.payoff -= self.cost

    def get_all_fitnesses(self):
        igss = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        ucs = [a for a in self.agents if a.agent_type == "Unconditional Cooperator"]
        ds = [a for a in self.agents if a.agent_type == "Defector"]
        
        f_igss = sum(a.payoff for a in igss) / len(igss) if igss else 0
        f_uc = sum(a.payoff for a in ucs) / len(ucs) if ucs else 0
        f_d = sum(a.payoff for a in ds) / len(ds) if ds else 0
        return f_igss, f_uc, f_d

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rule(individual, model_config, evo_config):
    compiled_rule = toolbox.compile(individual)
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    for _ in range(3): 
        m = MoneyModel(igss_rule=compiled_rule, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        i_fit, u_fit, d_fit = m.get_all_fitnesses()
        tot_igss += i_fit; tot_uc += u_fit; tot_d += d_fit
        
    tax = len(individual) * evo_config["PARSIMONY_TAX"]
    
    individual.sim_igss = tot_igss / 3
    individual.sim_uc = tot_uc / 3
    individual.sim_d = tot_d / 3
    
    return (individual.sim_igss - tax), 

toolbox.register("evaluate", evaluate_rule, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Mode 2 Money Benchmark...")
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
        
        best = tools.selBest(pop, k=1)[0]
        history["max_igss"].append(best.sim_igss)
        history["avg_igss"].append(np.mean([ind.sim_igss for ind in pop]))
        history["uc_scores"].append(best.sim_uc)
        history["d_scores"].append(best.sim_d)
        
        if gen % 10 == 0 or gen == 1:
            history["fossil_record"][gen] = f"RULE: {str(best)}"
            print(f" > Gen {gen:02d} | iGSS: {best.sim_igss:.1f} | UC: {best.sim_uc:.1f} | D: {best.sim_d:.1f}")

    # Extract final wealth distribution of the apex environment
    final_market = MoneyModel(toolbox.compile(best), MODEL_CONFIG)
    for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): final_market.step()
    wealth_dist = [a.tokens for a in final_market.agents]

    return best, history, wealth_dist

# =============================================================================
# 5. SYMPY SIMPLIFIER & DASHBOARD
# =============================================================================
def simplify_rule(gp_string):
    mapping = {'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y, 'MyTokens': sp.Symbol('Tokens')}
    try: return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: return f"Error: {e}"

def plot_dashboard(best_ind, history, wealth_dist, model_config, evo_config):
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 1])
    
    ax1 = fig.add_subplot(gs[0])
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    t_max = model_config["NUM_ROUNDS"] * (benefit - model_config["COST"])
    
    ax1.plot([(s/t_max)*100 for s in history["max_igss"]], label='Apex iGSS', color='darkorange', linewidth=2.5)
    ax1.plot([(s/t_max)*100 for s in history["uc_scores"]], label='Unconditional Cooperators', color='mediumseagreen', linewidth=2)
    ax1.plot([(s/t_max)*100 for s in history["d_scores"]], label='Defectors', color='crimson', linewidth=2)
    
    ax1.set_title('Mode 2 Money: Population Biological Fitness'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    bins = np.arange(0, max(wealth_dist) + 2) - 0.5 
    ax2.hist(wealth_dist, bins=bins, color='gold', edgecolor='black', alpha=0.8)
    ax2.set_title('Apex Final Wealth Distribution (Tokens)')
    ax2.set_xlabel('Tokens Held')
    ax2.set_ylabel('Number of Agents')
    ax2.set_xticks(range(max(wealth_dist) + 1))
    ax2.grid(axis='y', alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off') 
    
    fossils_str = "\n".join([f" Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    report_text = (
        f"--- APEX MONETARY STRATEGY ---\n"
        f"RAW: {str(best_ind)}\n"
        f"SIM: {simplify_rule(str(best_ind))}\n\n"
        f"--- ECOSYSTEM SETTINGS ---\n"
        f"iGSS: {model_config['NUM_IGSS']} | UC: {model_config['NUM_UC']} | D: {model_config['NUM_D']}\n"
        f"Initial Tokens: {model_config['INITIAL_TOKENS']} | Rounds: {model_config['NUM_ROUNDS']}\n"
        f"Max Gens: {evo_config['MAX_GENS']} | Parsimony Tax: {evo_config['PARSIMONY_TAX']}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
    )
    ax3.text(0.0, 0.95, report_text, fontsize=9, family='monospace', verticalalignment='top', transform=ax3.transAxes, wrap=True)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    best_ind, history, wealth_dist = run_evolution()
    
    print("\n" + "="*50)
    print("      MODE 2 (MONEY BENCHMARK) FINAL REPORT")
    print("="*50)
    print(f"RAW RULE: {str(best_ind)}")
    print(f"SIMPLIFIED: {simplify_rule(str(best_ind))}")
    print("==================================================")
    
    plot_dashboard(best_ind, history, wealth_dist, MODEL_CONFIG, EVO_CONFIG)