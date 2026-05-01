import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATIONS & REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_CONFIG = {
    "NUM_AGENTS": 40,      # 20 Apple Producers, 20 Banana Producers
    "NUM_ROUNDS": 50,      
    "CONSUMPTION_REWARD": 1.0,  # Points for successfully eating what you need
    "OFFER_COST": 0.05          # Friction: Proposing a trade costs energy
}

EVO_CONFIG = {
    "POP_SIZE": 60,        
    "MAX_GENS": 30,         
    "PARSIMONY_TAX": 0.1   
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP
# =============================================================================
def random_constant():
    return random.choice([-1, 1])

# PRIMITIVES: The agent knows what it wants, and it can see what the partner holds.
# Goods are represented as math integers: -1 (Apples) and 1 (Bananas)
pset_barter = gp.PrimitiveSet("BarterRule", 2) 
pset_barter.renameArguments(ARG0='My_Need', ARG1='Partner_Inv')

pset_barter.addPrimitive(operator.add, 2)
pset_barter.addPrimitive(operator.sub, 2)
pset_barter.addPrimitive(operator.mul, 2)
pset_barter.addEphemeralConstant("rand_const", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_barter", gp.genHalfAndHalf, pset=pset_barter, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_barter)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_barter)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_barter, pset=pset_barter)
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL (The Barter Economy)
# =============================================================================
class BarterAgent(mesa.Agent):
    def __init__(self, model, unique_id, produce, need, agent_type="iGSS"):
        super().__init__(model) 
        self.unique_id = unique_id
        self.my_produce = produce   # What I make (-1 or 1)
        self.my_need = need         # What I eat (1 or -1)
        self.inventory = produce    # I start the day holding what I made
        
        self.agent_type = agent_type
        self.fitness_score = 0.0    # Accumulates rewards and costs
        
    def get_action(self, partner_inventory):
        # The Control Baseline represents the perfect "Rational" Barterer
        if self.agent_type == "Control-Perfect":
            return 1 if self.my_need == partner_inventory else -1
            
        # iGSS evaluates the mathematical tree. >0 means OFFER, <=0 means REFUSE.
        score = self.model.igss_rule(self.my_need, partner_inventory)
        return 1 if score > 0 else -1

class BarterModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        
        # FIX: Renamed from self.agents to self.traders to avoid Mesa 3.0+ namespace collision
        self.traders = [] 
        
        a_type = "Control-Perfect" if control_mode else "iGSS"
        
        # Populate the economy: 50% produce -1 (need 1), 50% produce 1 (need -1)
        for i in range(config["NUM_AGENTS"]):
            if i % 2 == 0:
                self.traders.append(BarterAgent(self, i, produce=-1, need=1, agent_type=a_type))
            else:
                self.traders.append(BarterAgent(self, i, produce=1, need=-1, agent_type=a_type))

    def step(self):
        # The Random Soup: Agents bump into each other blindly
        random.shuffle(self.traders)
        
        for i in range(0, len(self.traders), 2):
            a0 = self.traders[i]
            a1 = self.traders[i+1]
            
            # 1. Agents evaluate if they want to offer a trade based on partner's inventory
            act_0 = a0.get_action(a1.inventory)
            act_1 = a1.get_action(a0.inventory)
            
            # 2. Apply Friction (Offer Cost)
            if act_0 > 0: a0.fitness_score -= self.config["OFFER_COST"]
            if act_1 > 0: a1.fitness_score -= self.config["OFFER_COST"]
            
            # 3. MUTUAL CONSENT: Trade only happens if BOTH agree
            if act_0 > 0 and act_1 > 0:
                # Swap goods
                a0.inventory, a1.inventory = a1.inventory, a0.inventory
                
                # Check for successful consumption
                if a0.inventory == a0.my_need:
                    a0.fitness_score += self.config["CONSUMPTION_REWARD"]
                    a0.inventory = a0.my_produce # Reset inventory after eating
                    
                if a1.inventory == a1.my_need:
                    a1.fitness_score += self.config["CONSUMPTION_REWARD"]
                    a1.inventory = a1.my_produce # Reset inventory after eating

    def get_average_fitness(self):
        # FIX: Updated to use self.traders
        return sum(a.fitness_score for a in self.traders) / len(self.traders)

# =============================================================================
# 4. EVOLUTIONARY ENGINE & DASHBOARD
# =============================================================================
def evaluate_rule(individual, config):
    func = toolbox.compile(expr=individual)
    tot_score = 0
    runs = 3 # Smooth out the noise of random matching
    for _ in range(runs):
        m = BarterModel(igss_rule=func, config=config)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_fitness()
        
    avg_score = tot_score / runs
    tax = len(individual) * EVO_CONFIG["PARSIMONY_TAX"]
    return (avg_score - tax),

toolbox.register("evaluate", evaluate_rule, config=MODEL_CONFIG)

def get_control_baseline(config):
    tot_score = 0
    runs = 5
    for _ in range(runs):
        m = BarterModel(igss_rule=None, config=config, control_mode=True)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_fitness()
    return tot_score / runs

def simplify_rule(gp_string):
    mapping = {
        'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y, 
        'My_Need': sp.Symbol('Need'), 'Partner_Inv': sp.Symbol('Inv')
    }
    try: return sp.simplify(eval(gp_string, {'__builtins__': {}}, mapping))
    except Exception: return "Parse Error"

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1)
    history = {"max_igss": [], "avg_igss": [], "fossil_record": {}}
    
    print("\nInitializing Phase 2: The Barter Economy...")
    control_fit = get_control_baseline(MODEL_CONFIG)
    print(f"Theoretical Barter Ceiling: {control_fit:.2f}\n")

    for gen in range(EVO_CONFIG["MAX_GENS"]):
        offspring = tools.selBest(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6: toolbox.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
        for m in offspring:
            if random.random() < 0.3: toolbox.mutate(m); del m.fitness.values
            
        for ind in [i for i in offspring if not i.fitness.valid]:
            ind.fitness.values = toolbox.evaluate(ind)
            
        pop[:] = offspring
        hof.update(pop)
        pop[-1] = toolbox.clone(hof[0]) 
        
        fits = [ind.fitness.values[0] for ind in pop]
        gen_max, gen_avg = max(fits), sum(fits) / len(fits)
        history["max_igss"].append(gen_max); history["avg_igss"].append(gen_avg)
        
        if gen % 3 == 0 or gen == EVO_CONFIG["MAX_GENS"] - 1:
            history["fossil_record"][gen] = str(hof[0])
            
        print(f" > Gen {gen:02d} | Max Fit: {gen_max:5.2f} | Avg Fit: {gen_avg:5.2f}")

    return hof[0], history, control_fit

def plot_dashboard(best_ind, history, config, control_fit):
    raw_action = str(best_ind)
    simp_action = simplify_rule(raw_action)
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    report_text = (
        f"\n{'='*60}\n"
        f"      iGSS PHASE 2: DOUBLE COINCIDENCE OF WANTS\n"
        f"{'='*60}\n"
        f"--- BEST EVOLVED RULE ---\n"
        f"Raw:   {raw_action}\n"
        f"SymPy: {simp_action}\n\n"
        f"--- FINAL PERFORMANCE ---\n"
        f"Control Baseline (Perfect Barter): {control_fit:.2f}\n"
        f"iGSS Final Max Fitness: {history['max_igss'][-1]:.2f}\n\n"
        f"--- FOSSIL RECORD (Sampled) ---\n{fossils_str}\n"
        f"{'='*60}\n"
    )
    print(report_text)

    fig, ax_plot = plt.subplots(figsize=(10, 6))
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    ax_plot.axhline(y=control_fit, color='black', linestyle=':', label=f'Control Baseline: {control_fit:.1f}')
    
    ax_plot.set_title('iGSS Emergence of Division of Labor Trade')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Net Consumption (Reward - Frictions)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, control_fit)