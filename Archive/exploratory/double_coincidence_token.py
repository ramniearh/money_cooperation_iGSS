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
    "NUM_AGENTS": 60,           # 20 of each species (0, 1, 2)
    "NUM_ROUNDS": 100,          # Longer rounds required for indirect reciprocity to circulate
    "CONSUMPTION_REWARD": 1.0,  
    "CONSENT_COST": 0.05,       # Friction: Evaluating/Saying 'Yes' costs a tiny bit of energy
    "INITIAL_TOKENS": 1         # The Genesis Block
}

EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 50,         
    "PARSIMONY_TAX": 0.05       # Slight tax to encourage the elegant "Bal + Val*dTok + 1" rule
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (The Physics of a Transaction)
# =============================================================================
def random_constant():
    return random.choice([-1, 1])

# PRIMITIVES: The agent evaluates the proposed transaction.
# Val_Good: 1 if I need it, 0 if I am giving my own away.
# Delta_Tok: 1 if I am receiving a token, -1 if I am paying one.
# My_Tok: My current token bank balance.
pset_money = gp.PrimitiveSet("MoneyRule", 3) 
pset_money.renameArguments(ARG0='Val_Good', ARG1='Delta_Tok', ARG2='My_Tok')

pset_money.addPrimitive(operator.add, 2)
pset_money.addPrimitive(operator.sub, 2)
pset_money.addPrimitive(operator.mul, 2)
pset_money.addEphemeralConstant("rand_const", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_money", gp.genHalfAndHalf, pset=pset_money, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_money)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_money)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_money, pset=pset_money)
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL (The Ring of Frustration)
# =============================================================================
class TokenAgent(mesa.Agent):
    def __init__(self, model, unique_id, species_id, agent_type="iGSS"):
        super().__init__(model) 
        self.unique_id = unique_id
        
        # The Ring: Species 0 needs 1. Species 1 needs 2. Species 2 needs 0.
        self.my_produce = species_id
        self.my_need = (species_id + 1) % 3 
        
        self.tokens = model.config["INITIAL_TOKENS"]
        self.agent_type = agent_type
        self.fitness_score = 0.0
        
    def consent_to_trade(self, val_good, delta_tok):
        if self.agent_type == "Control-Perfect":
            # The Perfect Fiat Strategy Baseline:
            # Buy if I need it and can afford it. Sell if I get a token.
            if val_good == 1 and delta_tok == -1: return 1 if self.tokens > 0 else -1
            if val_good == 0 and delta_tok == 1:  return 1 
            return -1
            
        # iGSS evaluates the transaction physics
        score = self.model.igss_rule(val_good, delta_tok, self.tokens)
        return 1 if score > 0 else -1

class TokenModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        
        # Uses self.traders to avoid Mesa 3.0+ protected namespace issues
        self.traders = []
        
        a_type = "Control-Perfect" if control_mode else "iGSS"
        
        # Populate the economy evenly with the 3 species
        for i in range(config["NUM_AGENTS"]):
            self.traders.append(TokenAgent(self, i, species_id=(i % 3), agent_type=a_type))

    def step(self):
        random.shuffle(self.traders)
        
        for i in range(0, len(self.traders) - 1, 2):
            a0 = self.traders[i]
            a1 = self.traders[i+1]
            
            buyer, seller = None, None
            
            # Check for Asymmetric Supply/Demand
            if a1.my_produce == a0.my_need:
                buyer, seller = a0, a1
            elif a0.my_produce == a1.my_need:
                buyer, seller = a1, a0
                
            if buyer and seller:
                # 1. Agents evaluate the transaction independently
                # Buyer evaluates losing 1 token to gain 1 biological value
                act_buyer = buyer.consent_to_trade(val_good=1, delta_tok=-1)
                
                # Seller evaluates gaining 1 token for surrendering 0 biological value
                act_seller = seller.consent_to_trade(val_good=0, delta_tok=1)
                
                # 2. Friction Costs
                if act_buyer > 0: buyer.fitness_score -= self.config["CONSENT_COST"]
                if act_seller > 0: seller.fitness_score -= self.config["CONSENT_COST"]
                
                # 3. MUTUAL CONSENT & LIQUIDITY CHECK
                if act_buyer > 0 and act_seller > 0 and buyer.tokens >= 1:
                    # Execute Trade
                    buyer.tokens -= 1
                    seller.tokens += 1
                    
                    # Reward Consumption
                    buyer.fitness_score += self.config["CONSUMPTION_REWARD"]

    def get_average_fitness(self):
        return sum(a.fitness_score for a in self.traders) / len(self.traders)

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rule(individual, config):
    func = toolbox.compile(expr=individual)
    tot_score = 0
    runs = 3 
    for _ in range(runs):
        m = TokenModel(igss_rule=func, config=config)
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
        m = TokenModel(igss_rule=None, config=config, control_mode=True)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_fitness()
    return tot_score / runs

def simplify_rule(gp_string):
    mapping = {
        'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y, 
        'Val_Good': sp.Symbol('Val'), 'Delta_Tok': sp.Symbol('dTok'), 'My_Tok': sp.Symbol('Bal')
    }
    try: return sp.simplify(eval(gp_string, {'__builtins__': {}}, mapping))
    except Exception: return "Parse Error"

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1)
    history = {"max_igss": [], "avg_igss": [], "fossil_record": {}}
    
    print("\nInitializing Phase 4: The Token Economy (Ring of Frustration)...")
    control_fit = get_control_baseline(MODEL_CONFIG)
    print(f"Theoretical Fiat Economy Ceiling: {control_fit:.2f}\n")

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
        
        if gen % 5 == 0 or gen == EVO_CONFIG["MAX_GENS"] - 1:
            history["fossil_record"][gen] = str(hof[0])
            
        print(f" > Gen {gen:02d} | Max Fit: {gen_max:5.2f} | Avg Fit: {gen_avg:5.2f}")

    return hof[0], history, control_fit

# =============================================================================
# 5. DASHBOARD & VISUALIZATION
# =============================================================================
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Pure Python hierarchical layout for networkx."""
    if not nx.is_tree(G): raise TypeError('Graph must be a tree')
    if root is None: root = next(iter([n for n, d in G.in_degree() if d == 0]))
    def _hierarchy_pos(G, node, width, vert_gap, vert_loc, xcenter, pos=None, parent=None):
        if pos is None: pos = {node: (xcenter, vert_loc)}
        else: pos[node] = (xcenter, vert_loc)
        children = list(G.successors(node))
        if len(children) != 0:
            dx = width / len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=node)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def plot_deap_tree(individual):
    nodes, edges, labels = gp.graph(individual)
    g = nx.DiGraph() 
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    
    try: pos = hierarchy_pos(g)
    except TypeError: pos = nx.spring_layout(g, seed=42) 
        
    fig_tree, ax_tree = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, ax=ax_tree, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g, pos, ax=ax_tree, arrows=False)
    nx.draw_networkx_labels(g, pos, labels, ax=ax_tree, font_size=10)
    ax_tree.set_title("Evolved Fiat Money Rule (AST)", fontsize=14)
    ax_tree.axis("off")
    plt.show(block=False)

def plot_dashboard(best_ind, history, config, control_fit):
    raw_action = str(best_ind)
    simp_action = simplify_rule(raw_action)
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    report_text = (
        f"\n{'='*60}\n"
        f"      iGSS PHASE 4: EMERGENCE OF FIAT MONEY\n"
        f"{'='*60}\n"
        f"--- BEST EVOLVED RULE ---\n"
        f"Raw:   {raw_action}\n"
        f"SymPy: {simp_action}\n\n"
        f"--- FINAL PERFORMANCE ---\n"
        f"Control Baseline (Perfect Fiat): {control_fit:.2f}\n"
        f"iGSS Final Max Fitness: {history['max_igss'][-1]:.2f}\n\n"
        f"--- FOSSIL RECORD (Sampled) ---\n{fossils_str}\n"
        f"{'='*60}\n"
    )
    print(report_text)

    fig, ax_plot = plt.subplots(figsize=(10, 6))
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    ax_plot.axhline(y=control_fit, color='black', linestyle=':', label=f'Control Baseline: {control_fit:.1f}')
    
    ax_plot.set_title('iGSS Emergence of a Token Economy')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Net Consumption (Reward - Frictions)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    
    plot_deap_tree(best_ind)
    plt.show() # Blocks until windows are closed

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, control_fit)