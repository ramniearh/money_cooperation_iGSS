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
# Setting seeds ensures that the evolutionary search path is reproducible.
# This is crucial when sharing results with co-authors.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_CONFIG = {
    "NUM_PAIRS": 20,       # 40 agents total (20 with ID=0, 20 with ID=1)
    "NUM_ROUNDS": 50,      # A perfectly equitable split over 50 rounds is 25 gives each.
    "COLLISION_PENALTY": 0.5 
}

EVO_CONFIG = {
    "POP_SIZE": 60,        
    "MAX_GENS": 30,         
    "PARSIMONY_TAX": 0.1   # Penalizes bloated math trees to encourage human-readable rules.
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (Continuous Math)
# =============================================================================
# Fix for DEAP's lambda pickling warning. This provides random constants to the GP.
def random_constant():
    return random.choice([-1, 1])

# THE PRIMITIVES: The agents are given two raw, meaningless variables.
# 1. 'Env_Clock': A global 0/1 alternator.
# 2. 'My_ID': A static 0/1 tag assigned at birth.
# It is up to the evolutionary process to figure out how to combine them to break symmetry.
pset_action = gp.PrimitiveSet("ActionRule_Clock", 2) 
pset_action.renameArguments(ARG0='Env_Clock', ARG1='My_ID')

# THE COMBINATORS: Basic arithmetic.
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const", random_constant)

# DEAP Boilerplate for maximizing a single objective (Fitness)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_action)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_action)

# Standard evolutionary operators
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_action, pset=pset_action)
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL (Synchronous Matching)
# =============================================================================
class CoordAgent(mesa.Agent):
    def __init__(self, model, my_id, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.my_id = my_id # 0 or 1
        self.agent_type = agent_type
        
        # Trackers for the Leontief Macro-Target
        self.successful_gives = 0
        self.collisions = 0
        
    def get_action(self, clock):
        if self.agent_type == "Control-Perfect":
            # Baseline explicitly programs an XOR gate to set the theoretical ceiling
            return 1 if self.my_id != clock else -1
            
        # iGSS evaluates its evolved mathematical tree synchronously.
        # Output > 0 implies "Give" (1), <= 0 implies "Receive" (-1)
        score = self.model.igss_rule(clock, self.my_id)
        return 1 if score > 0 else -1

class CoordinationModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.clock = 0 # The external environmental metronome
        
        a_type = "Control-Perfect" if control_mode else "iGSS-Agent"
        
        # Populate the world with two distinct "species" of IDs
        self.agents_0 = [CoordAgent(self, 0, a_type) for _ in range(config["NUM_PAIRS"])]
        self.agents_1 = [CoordAgent(self, 1, a_type) for _ in range(config["NUM_PAIRS"])]

    def step(self):
        # 1. Toggle the External Alternator
        self.clock = 1 - self.clock
        
        # 2. Random Pairwise Matching (ID=0 always meets ID=1)
        random.shuffle(self.agents_0)
        random.shuffle(self.agents_1)

        for a0, a1 in zip(self.agents_0, self.agents_1):
            # Both agents evaluate the SAME rule at the SAME time.
            act_0 = a0.get_action(self.clock)
            act_1 = a1.get_action(self.clock)
            
            # 3. Resolve the synchronous interaction
            if act_0 != act_1:
                # Handshake successful. Track who gave.
                if act_0 == 1: a0.successful_gives += 1
                else:          a1.successful_gives += 1
            else: 
                # Collision. They crashed into each other.
                a0.collisions += 1
                a1.collisions += 1

    def get_average_fitness(self):
        total_fit = 0
        for a0, a1 in zip(self.agents_0, self.agents_1):
            # MACRO-TARGET: THE LEONTIEF FITNESS FUNCTION
            # We take the minimum of the two agents' gives. 
            # This mathematically punishes static "Parasite/Host" behavior.
            # To maximize this, agents MUST learn to balance the giving perfectly.
            pair_score = min(a0.successful_gives, a1.successful_gives)
            pair_score -= (a0.collisions * self.config["COLLISION_PENALTY"])
            total_fit += pair_score
            
        return total_fit / len(self.agents_0)

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rule(individual, config):
    func = toolbox.compile(expr=individual)
    m = CoordinationModel(igss_rule=func, config=config)
    for _ in range(config["NUM_ROUNDS"]): 
        m.step()
    
    score = m.get_average_fitness()
    tax = len(individual) * EVO_CONFIG["PARSIMONY_TAX"]
    return (score - tax),

toolbox.register("evaluate", evaluate_rule, config=MODEL_CONFIG)

def get_control_baseline(config):
    """Calculates the max theoretical fitness using perfectly programmed alternators."""
    m = CoordinationModel(igss_rule=None, config=config, control_mode=True)
    for _ in range(config["NUM_ROUNDS"]): 
        m.step()
    return m.get_average_fitness()

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1)
    
    # Data tracking for the dashboard visualization
    history = {"max_igss": [], "avg_igss": [], "fossil_record": {}}
    
    print("\nInitializing iGSS External Clock Sandbox...")
    control_fit = get_control_baseline(MODEL_CONFIG)
    print(f"Control Baseline (Perfect XOR): {control_fit:.2f}\n")

    for gen in range(EVO_CONFIG["MAX_GENS"]):
        offspring = tools.selBest(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover & Mutation
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6: 
                toolbox.mate(c1, c2)
                del c1.fitness.values; del c2.fitness.values
        for m in offspring:
            if random.random() < 0.3: 
                toolbox.mutate(m)
                del m.fitness.values
            
        # Evaluate valid individuals
        for ind in [i for i in offspring if not i.fitness.valid]:
            ind.fitness.values = toolbox.evaluate(ind)
            
        pop[:] = offspring
        hof.update(pop)
        pop[-1] = toolbox.clone(hof[0]) # Elitism: Guarantee the best survives
        
        # Record keeping
        fits = [ind.fitness.values[0] for ind in pop]
        gen_max = max(fits)
        gen_avg = sum(fits) / len(fits)
        history["max_igss"].append(gen_max)
        history["avg_igss"].append(gen_avg)
        
        best_rule = hof[0]
        
        # Save to fossil record periodically for the final report
        if gen % 3 == 0 or gen == EVO_CONFIG["MAX_GENS"] - 1:
            history["fossil_record"][gen] = str(best_rule)
        
        print(f" > Gen {gen:02d} | Max Fit: {gen_max:5.2f} | Avg Fit: {gen_avg:5.2f}")

    return hof[0], history, control_fit

# =============================================================================
# 5. REPORTING & VISUALIZATION (AST & Dashboard)
# =============================================================================
def simplify_rule(gp_string):
    """Converts the DEAP tree string into a simplified algebraic formula."""
    mapping = {
        'add': lambda x, y: x + y, 
        'sub': lambda x, y: x - y, 
        'mul': lambda x, y: x * y, 
        'Env_Clock': sp.Symbol('Clock'), 
        'My_ID': sp.Symbol('ID')
    }
    try: 
        # Using single quotes for the inner dict to avoid f-string escaping errors
        return sp.simplify(eval(gp_string, {'__builtins__': {}}, mapping))
    except Exception as e: 
        return f"Parse Error: {e}"

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Pure Python hierarchical layout for networkx (bypasses graphviz dependencies)."""
    if not nx.is_tree(G): raise TypeError('Graph must be a tree to use hierarchy_pos')
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
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=node)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def plot_deap_tree(individual):
    """Draws the abstract syntax tree cleanly to visualize the 'brain' of the agent."""
    nodes, edges, labels = gp.graph(individual)
    g = nx.DiGraph() 
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    
    try:
        pos = hierarchy_pos(g)
    except TypeError:
        # Fallback if the evolutionary engine generates a disconnected graph somehow
        pos = nx.spring_layout(g, seed=42) 
        
    fig_tree, ax_tree = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, ax=ax_tree, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g, pos, ax=ax_tree, arrows=False)
    nx.draw_networkx_labels(g, pos, labels, ax=ax_tree, font_size=10)
    ax_tree.set_title("Evolved Cognitive Rule (Abstract Syntax Tree)", fontsize=14)
    ax_tree.axis("off")
    plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    """Compiles the evolutionary history into a publication-style report and chart."""
    raw_action = str(best_ind)
    simp_action = simplify_rule(raw_action)
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    # In a 50-round game, perfect turn taking means each agent gives 25 times.
    theoretical_max = model_config["NUM_ROUNDS"] / 2.0 
    final_igss_avg = history["max_igss"][-1]
    
    # 1. Terminal Report
    report_text = (
        f"\n{'='*60}\n"
        f"      iGSS SYNCHRONOUS COORDINATION REPORT\n"
        f"{'='*60}\n"
        f"--- BEST EVOLVED RULE ---\n"
        f"Raw:   {raw_action}\n"
        f"SymPy: {simp_action}\n\n"
        f"--- ABM CONFIGURATIONS ---\n"
        f"Pairs: {model_config['NUM_PAIRS']} | Rounds: {model_config['NUM_ROUNDS']} | Target: Leontief Min()\n\n"
        f"--- EVO CONFIGURATIONS ---\n"
        f"Pop: {evo_config['POP_SIZE']} | Gens: {evo_config['MAX_GENS']} | Parsimony Tax: {evo_config['PARSIMONY_TAX']}\n\n"
        f"--- FINAL PERFORMANCE ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline (Perfect XOR): {control_fit:.2f}\n"
        f"iGSS Final Max Fitness: {final_igss_avg:.2f}\n\n"
        f"--- FOSSIL RECORD (Sampled) ---\n{fossils_str}\n"
        f"{'='*60}\n"
    )
    print(report_text)

    # 2. Main Fitness Chart
    fig, ax_plot = plt.subplots(figsize=(10, 6))
    
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    
    # Baselines
    ax_plot.axhline(y=control_fit, color='black', linestyle=':', label=f'Control Baseline (XOR): {control_fit:.1f}')
    
    ax_plot.set_title('iGSS Emergence of Reciprocal Turn-Taking')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Leontief Fitness Score')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # 3. Popup Tree Visualization
    plot_deap_tree(best_ind)
    plt.show() # Blocks until all matplotlib windows are closed

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)