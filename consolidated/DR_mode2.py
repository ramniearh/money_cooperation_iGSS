# note slight shift in the place where DR forgiveness and punishment happen in the sequential code.

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
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 100
}

EVO_CONFIG = {
    "POP_SIZE": 40,        
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (MEMORY LEDGER SEARCH)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

pset_assessment = gp.PrimitiveSet("AssessmentRule_DR", 2) 
pset_assessment.renameArguments(ARG0='DonorAction', ARG1='DonorInMemory')
pset_assessment.addPrimitive(operator.add, 2)
pset_assessment.addPrimitive(operator.sub, 2)
pset_assessment.addPrimitive(operator.mul, 2)
pset_assessment.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_assessment", gp.genHalfAndHalf, pset=pset_assessment, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_assessment)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_assessment)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_assessment, pset=pset_assessment)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. AGENT-BASED MODEL (Asymmetric Topology & Memory Mechanics)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.memory = set() 
        
    def evaluate_partner(self, partner):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # MODE 2: Action Rule is fixed to Strict Tit-for-Tat for both classes
        return partner.unique_id not in self.memory

class CooperationModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.control_mode = control_mode
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-Agent" if control_mode else "iGSS-Agent"
        
        for _ in range(config["NUM_IGSS"]): CoopAgent(self, target_agent)
        for _ in range(config["NUM_UC"]): CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): CoopAgent(self, "Defector")

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for donor in agents:
            possible_recipients = [a for a in agents if a.unique_id != donor.unique_id]
            recipient = random.choice(possible_recipients)

            donor_action = donor.evaluate_partner(recipient)
            self.resolve(donor=donor, recipient=recipient, cooperates=donor_action)
            
    def resolve(self, donor, recipient, cooperates):
        # 1. Economic Execution
        if cooperates:
            donor.payoff -= self.cost
            recipient.payoff += self.benefit
            
        # 2. Institutional Assessment (Updating Memory)
        if self.control_mode:
            # HARDCODED CONTROL: The strict pseudocode implementation
            if not cooperates:
                recipient.memory.add(donor.unique_id)
                if recipient.unique_id in donor.memory:
                    donor.memory.remove(recipient.unique_id)
        else:
            # EVOLVED iGSS ASSESSMENT: The Physics of the Grudge
            action_val = 1 if cooperates else 0
            donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
            
            decision_score = self.igss_rule(action_val, donor_in_mem)
            
            # > 0 means "Add/Keep in Memory" (Hold Grudge)
            # <= 0 means "Remove/Keep out of Memory" (Forgive/Trust)
            if decision_score > 0:
                recipient.memory.add(donor.unique_id)
            else:
                recipient.memory.discard(donor.unique_id)

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Control-Agent", "Unconditional Cooperator", "Defector"]:
            type_agents = [a for a in self.agents if a.agent_type == a_type]
            if type_agents:
                fitness_data[a_type] = sum(a.payoff for a in type_agents) / len(type_agents)
            else:
                fitness_data[a_type] = 0
        return fitness_data

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rule(individual, model_config, evo_config):
    func_assessment = toolbox.compile(expr=individual)
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_rule=func_assessment, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    tax = len(individual) * evo_config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

toolbox.register("evaluate", evaluate_rule, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def get_control_baseline(model_config):
    tot_control = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(igss_rule=None, config=model_config, control_mode=True)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_control += m.get_fitness_by_type()["Control-Agent"]
    return tot_control / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Direct Reciprocity [MODE 2 - Memory Assessment Search]...")
    control_fitness = get_control_baseline(MODEL_CONFIG)
    print(f"Control Baseline established at: {control_fitness:.2f} average payoff.")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    
    hof.update(pop)
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values; del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
            ind.fitness.values = fit
                
        pop[:] = offspring
        hof.update(pop) 
        
        pop[-1] = toolbox.clone(hof[0])
        
        best_in_gen = tools.selBest(pop, k=1)[0]
        history["max_igss"].append(best_in_gen.sim_igss)
        history["avg_igss"].append(np.mean([ind.sim_igss for ind in pop]))
        history["uc_scores"].append(best_in_gen.sim_uc)
        history["d_scores"].append(best_in_gen.sim_d)
        
        if gen % 10 == 0 or gen == 1:
            history["fossil_record"][gen] = str(best_in_gen)
            print(f" > Gen {gen:02d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def simplify_rule(gp_string):
    mapping = {
        'add': lambda x, y: x + y, 
        'sub': lambda x, y: x - y, 
        'mul': lambda x, y: x * y,
        'DonorAction': sp.Symbol('DonorAction'),
        'DonorInMemory': sp.Symbol('DonorInMemory')
    }
    try: 
        return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: 
        return f"SymPy Parsing Error: {e}"

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
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
    nx.draw_networkx_nodes(g, pos, ax=ax_tree, node_size=1000, node_color="lightgreen")
    nx.draw_networkx_edges(g, pos, ax=ax_tree, arrows=False)
    nx.draw_networkx_labels(g, pos, labels, ax=ax_tree, font_size=10)
    ax_tree.set_title("GP Assessment Rule (Memory Ledger)", fontsize=14)
    ax_tree.axis("off")
    plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    raw_action = str(best_ind)
    simp_action = simplify_rule(raw_action)
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    final_igss_avg = history["max_igss"][-1]
    final_uc_avg = history["uc_scores"][-1]
    final_d_avg = history["d_scores"][-1]
    
    report_text = (
        f"\n{'='*50}\n"
        f"      DIRECT RECIPROCITY [MODE 2] REPORT\n"
        f"{'='*50}\n"
        f"--- BEST EVOLVED ASSESSMENT RULE (MEMORY) ---\n"
        f"Raw:   {raw_action}\n"
        f"SymPy: {simp_action}\n\n"
        f"--- ABM CONFIGURATIONS ---\n"
        f"iGSS: {model_config['NUM_IGSS']} | UC: {model_config['NUM_UC']} | Defectors: {model_config['NUM_D']}\n"
        f"Benefit: {model_config['BENEFIT_TO_COST_RATIO']} | Cost: {model_config['COST']} | Rounds: {model_config['NUM_ROUNDS']}\n\n"
        f"--- EVO CONFIGURATIONS ---\n"
        f"Pop: {evo_config['POP_SIZE']} | Gens: {evo_config['MAX_GENS']} | Parsimony Tax: {evo_config['PARSIMONY_TAX']}\n\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline (Pseudocode Memory): {control_fit:.1f}\n"
        f"iGSS Max Payoff: {final_igss_avg:.1f}\n"
        f"Uncond. Cooperators Avg Payoff: {final_uc_avg:.1f}\n"
        f"Defectors Avg Payoff: {final_d_avg:.1f}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
        f"{'='*50}\n"
    )
    print(report_text)

    fig, ax_plot = plt.subplots(figsize=(10, 6))
    
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    ax_plot.plot(history["uc_scores"], label='Unconditional Cooperators', color='green', linewidth=1.5, linestyle='--')
    ax_plot.plot(history["d_scores"], label='Defectors', color='red', linewidth=1.5, linestyle='-.')
    
    ax_plot.axhline(y=control_fit, color='black', linestyle=':', label=f'Control Baseline: {control_fit:.1f}')
    ax_plot.axhline(y=theoretical_max, color='purple', alpha=0.3, label=f'Maximum Theoretical Fitness: {theoretical_max}')
    
    ax_plot.set_title('Direct Reciprocity [Mode 2]: Search for Memory Mechanics')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    plot_deap_tree(best_ind)
    plt.show() 

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)