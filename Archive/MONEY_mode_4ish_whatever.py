import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (The Fiat Economy)
# =============================================================================
MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "TRIBE_SIZE": 10,
    "NUM_TRIBES": 3,
    "NUM_ROUNDS": 100,
    "INITIAL_TOKENS": 2  # The primordial wealth given to each agent
}

EVO_CONFIG = {
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1  
}

# =============================================================================
# 2. GP SETUP (Single Tree - The Supply Curve)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# Agents only have one cognitive tree now. 
# "Given my current token balance, am I willing to sell my help for 1 token?"
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
# 3. MESA MODEL (The Vending Machine Exchange)
# =============================================================================
class MoneyAgent(mesa.Agent):
    def __init__(self, model, global_index, tribe_id, action_rule):
        super().__init__(model) 
        self.global_index = global_index
        self.tribe_id = tribe_id
        self.action_rule = action_rule
        
        # Dual Ledgers
        self.payoff = 0                             # Biological Fitness (DEAP cares about this)
        self.tokens = MODEL_CONFIG["INITIAL_TOKENS"] # Fiat Wealth (MESA physics care about this)

class MoneyModel(mesa.Model):
    def __init__(self, flat_pop_rules, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        for i, rule in enumerate(flat_pop_rules):
            tribe_id = i // config["TRIBE_SIZE"]
            MoneyAgent(self, i, tribe_id, rule)

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for i in range(0, len(agents) - 1, 2):
            agent_A = agents[i]
            agent_B = agents[i+1]

            # INTERACTION 1: Agent A asks Agent B for help
            # 1. Does A have a token to pay?
            a_can_afford = agent_A.tokens >= 1
            # 2. Is B willing to sell help based on their current bank account?
            b_willing_to_sell = agent_B.action_rule(agent_B.tokens) > 0

            if a_can_afford and b_willing_to_sell:
                agent_A.tokens -= 1
                agent_B.tokens += 1
                agent_A.payoff += self.benefit
                agent_B.payoff -= self.cost

            # INTERACTION 2: Agent B asks Agent A for help
            b_can_afford = agent_B.tokens >= 1
            a_willing_to_sell = agent_A.action_rule(agent_A.tokens) > 0

            if b_can_afford and a_willing_to_sell:
                agent_B.tokens -= 1
                agent_A.tokens += 1
                agent_B.payoff += self.benefit
                agent_A.payoff -= self.cost

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def run_evolution():
    tribes = [toolbox.population(n=MODEL_CONFIG["TRIBE_SIZE"]) for _ in range(MODEL_CONFIG["NUM_TRIBES"])]
    history = {"max_T0": [], "max_T1": [], "max_T2": [], "avg_eco": [], "fossil_record": {}}
    
    print("Initializing Mode 4: The Fiat Economy...")
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        compiled_rules = [toolbox.compile(ind) for pop in tribes for ind in pop]
                
        total_payoffs = [0] * (MODEL_CONFIG["TRIBE_SIZE"] * MODEL_CONFIG["NUM_TRIBES"])
        
        for _ in range(3):
            m = MoneyModel(compiled_rules)
            for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): m.step()
            for a in m.agents: total_payoffs[a.global_index] += a.payoff
                
        global_idx = 0
        all_fits = []
        tribe_maxes = []
        
        for pop in tribes:
            t_fits = []
            for ind in pop:
                fit = (total_payoffs[global_idx]/3) - (len(ind)*EVO_CONFIG["PARSIMONY_TAX"])
                ind.fitness.values = (fit,)
                t_fits.append(fit); all_fits.append(fit)
                global_idx += 1
            tribe_maxes.append(max(t_fits))

        history["max_T0"].append(tribe_maxes[0])
        history["max_T1"].append(tribe_maxes[1])
        history["max_T2"].append(tribe_maxes[2])
        history["avg_eco"].append(np.mean(all_fits))
        
        best_overall = tools.selBest([ind for t in tribes for ind in t], 1)[0]
        
        if gen % 10 == 0 or gen == 1:
            history["fossil_record"][gen] = f"RULE: {str(best_overall)}"
            print(f" > Gen {gen:02d} | Tribe Fitness: {[round(m) for m in tribe_maxes]}")

        if gen < EVO_CONFIG["MAX_GENS"]:
            for i in range(len(tribes)):
                offspring = list(map(toolbox.clone, toolbox.select(tribes[i], len(tribes[i]))))
                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.5: toolbox.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
                for mutant in offspring:
                    if random.random() < 0.2: toolbox.mutate(mutant); del mutant.fitness.values
                tribes[i][:] = offspring

    # --- POST-EVOLUTION: Final Token Run ---
    # We run the final ecosystem one last time to pull the exact geometric wealth distribution.
    final_compiled = [toolbox.compile(ind) for pop in tribes for ind in pop]
    final_market = MoneyModel(final_compiled)
    for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): final_market.step()
    final_wealth_distribution = [a.tokens for a in final_market.agents]

    return best_overall, history, final_wealth_distribution

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
    
    # Plot 1: Biological Fitness
    ax1 = fig.add_subplot(gs[0])
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    t_max = model_config["NUM_ROUNDS"] * (benefit - model_config["COST"])
    
    ax1.plot([(s/t_max)*100 for s in history["max_T0"]], label='Tribe A', color='royalblue')
    ax1.plot([(s/t_max)*100 for s in history["max_T1"]], label='Tribe B', color='seagreen')
    ax1.plot([(s/t_max)*100 for s in history["max_T2"]], label='Tribe C', color='crimson')
    ax1.plot([(s/t_max)*100 for s in history["avg_eco"]], label='Global Avg', color='gray', linestyle='--')
    ax1.set_title('Evolution of Biological Fitness (Efficiency %)'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Plot 2: Fiat Wealth Distribution
    ax2 = fig.add_subplot(gs[1])
    # The histogram bins are aligned to whole integers (0 tokens, 1 token, etc.)
    bins = np.arange(0, max(wealth_dist) + 2) - 0.5 
    ax2.hist(wealth_dist, bins=bins, color='gold', edgecolor='black', alpha=0.8)
    ax2.set_title('Final Wealth Distribution (Tokens)')
    ax2.set_xlabel('Tokens Held')
    ax2.set_ylabel('Number of Agents')
    ax2.set_xticks(range(max(wealth_dist) + 1))
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Text Lab Report
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off') 
    raw_rule = str(best_ind)
    fossils_str = "\n".join([f" Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    report_text = (
        f"--- APEX MONETARY STRATEGY ---\n"
        f"RAW: {raw_rule}\n"
        f"SIM: {simplify_rule(raw_rule)}\n\n"
        f"--- ECOSYSTEM SETTINGS ---\n"
        f"Tribes: {model_config['NUM_TRIBES']} | Agents per Tribe: {model_config['TRIBE_SIZE']}\n"
        f"Initial Tokens: {model_config['INITIAL_TOKENS']} | Rounds: {model_config['NUM_ROUNDS']}\n"
        f"Max Gens: {evo_config['MAX_GENS']} | Parsimony Tax: {evo_config['PARSIMONY_TAX']}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
    )
    ax3.text(0.0, 0.95, report_text, fontsize=9, family='monospace', verticalalignment='top', transform=ax3.transAxes, wrap=True)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    best_ind, history, wealth_dist = run_evolution()
    
    print("\n" + "="*50)
    print("      MODE 4 (MONEY PROTOCOL) FINAL REPORT")
    print("="*50)
    print(f"SUPPLY CURVE RULE: {simplify_rule(str(best_ind))}")
    print("==================================================")
    
    plot_dashboard(best_ind, history, wealth_dist, MODEL_CONFIG, EVO_CONFIG)