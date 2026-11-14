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
SEED = 421
random.seed(SEED)
np.random.seed(SEED)

MODEL_CONFIG = {
    "ACTION_LOGIC": "OR",   # Toggle this between "AND" and "OR" to test both variants!
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 100
}

EVO_CONFIG = {
    "POP_SIZE": 60,        # Slightly larger pop to handle dual-tree search
    "MAX_GENS": 150,       # More generations needed for assessment stabilization
    "PARSIMONY_TAX": 0.05
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (DUAL ASSESSMENT TREES)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# Shared Primitive Set for BOTH Assessment Trees
pset_assess = gp.PrimitiveSet("AssessmentRule_Combined", 4) 
pset_assess.renameArguments(ARG0='DonorAction', ARG1='DonorInMemory', ARG2='DonorStanding', ARG3='RecipientStanding')
pset_assess.addPrimitive(operator.add, 2)
pset_assess.addPrimitive(operator.sub, 2)
pset_assess.addPrimitive(operator.mul, 2)
pset_assess.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Individual is a LIST of two trees: [Memory_Tree, Standing_Tree]
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_assess", gp.genHalfAndHalf, pset=pset_assess, min_=1, max_=3)

def init_individual(container, func):
    return container([gp.PrimitiveTree(func()), gp.PrimitiveTree(func())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_assess)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compiler
toolbox.register("compile", gp.compile, pset=pset_assess)

# Custom Genetic Operators for Multi-Tree DNA
def cxTwoTrees(ind1, ind2):
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0]) # Crossover Memory trees
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1]) # Crossover Standing trees
    return ind1, ind2

def mutTwoTrees(individual):
    if random.random() < 0.5:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_assess, pset=pset_assess)
    else:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_assess, pset=pset_assess)
    return individual,

toolbox.register("mate", cxTwoTrees)
toolbox.register("mutate", mutTwoTrees)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. AGENT-BASED MODEL 
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.memory = set()
        self.standing = 1 
        
    def evaluate_partner(self, partner):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # In Mode 2, BOTH Control and iGSS use the same hardcoded Action rule!
        logic = self.model.config["ACTION_LOGIC"]
        if logic == "AND":
            return (partner.unique_id not in self.memory) and (partner.standing == 1)
        elif logic == "OR":
            return (partner.unique_id not in self.memory) or (partner.standing == 1)
        return True

class CooperationModel(mesa.Model):
    def __init__(self, igss_mem_rule, igss_stand_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_mem_rule = igss_mem_rule
        self.igss_stand_rule = igss_stand_rule
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
            
        # 2. Institutional Assessment
        if self.control_mode:
            # HARDCODED CONTROL JUSTICE SYSTEM
            if cooperates:
                donor.standing = 1
            else:
                recipient.memory.add(donor.unique_id)
                if recipient.unique_id in donor.memory:
                    donor.memory.remove(recipient.unique_id)
                if recipient.standing == 1:
                    donor.standing = 0
        else:
            # EVOLVED iGSS ASSESSMENT (Dual Trees)
            action_val = 1 if cooperates else 0
            donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
            
            # Tree 0: Memory Ledger Update
            mem_score = self.igss_mem_rule(action_val, donor_in_mem, donor.standing, recipient.standing)
            if mem_score > 0:
                recipient.memory.add(donor.unique_id)
            else:
                recipient.memory.discard(donor.unique_id)
                
            # Tree 1: Public Standing Update
            stand_score = self.igss_stand_rule(action_val, donor_in_mem, donor.standing, recipient.standing)
            donor.standing = 1 if stand_score > 0 else 0

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
def evaluate_rules(individual, model_config, evo_config):
    func_mem = toolbox.compile(expr=individual[0])
    func_stand = toolbox.compile(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_mem_rule=func_mem, igss_stand_rule=func_stand, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    # Combined Parsimony Tax for both trees
    tax = (len(individual[0]) + len(individual[1])) * evo_config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

toolbox.register("evaluate", evaluate_rules, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def get_control_baseline(model_config):
    tot_control = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(igss_mem_rule=None, igss_stand_rule=None, config=model_config, control_mode=True)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_control += m.get_fitness_by_type()["Control-Agent"]
    return tot_control / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    logic = MODEL_CONFIG["ACTION_LOGIC"]
    print(f"Initializing Combined DR & IR [MODE 2 - Assessment Search] with {logic} Action Logic...")
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
            history["fossil_record"][gen] = f"MEM: {best_in_gen[0]}  |  STAND: {best_in_gen[1]}"
            print(f" > Gen {gen:02d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def parse_sympy(gp_string):
    mapping = {
        'add': lambda x,y: x+y, 
        'sub': lambda x,y: x-y, 
        'mul': lambda x,y: x*y, 
        'DonorAction': sp.Symbol('DonorAction'), 
        'DonorInMemory': sp.Symbol('DonorInMemory'),
        'DonorStanding': sp.Symbol('DonorStanding'),
        'RecipientStanding': sp.Symbol('RecipientStanding')
    }
    try: return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: return f"Error: {e}"

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

def plot_dual_trees(individual):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Memory Assessment Tree
    n1, e1, l1 = gp.graph(individual[0])
    g1 = nx.DiGraph(); g1.add_nodes_from(n1); g1.add_edges_from(e1)
    try: p1 = hierarchy_pos(g1)
    except TypeError: p1 = nx.spring_layout(g1)
    nx.draw_networkx_nodes(g1, p1, ax=ax1, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g1, p1, ax=ax1, arrows=False)
    nx.draw_networkx_labels(g1, p1, l1, ax=ax1, font_size=10)
    ax1.set_title("Evolved MEMORY Update Rule", fontsize=14); ax1.axis("off")
    
    # Standing Assessment Tree
    n2, e2, l2 = gp.graph(individual[1])
    g2 = nx.DiGraph(); g2.add_nodes_from(n2); g2.add_edges_from(e2)
    try: p2 = hierarchy_pos(g2)
    except TypeError: p2 = nx.spring_layout(g2)
    nx.draw_networkx_nodes(g2, p2, ax=ax2, node_size=1000, node_color="lightgreen")
    nx.draw_networkx_edges(g2, p2, ax=ax2, arrows=False)
    nx.draw_networkx_labels(g2, p2, l2, ax=ax2, font_size=10)
    ax2.set_title("Evolved STANDING Update Rule", fontsize=14); ax2.axis("off")
    
    plt.tight_layout()
    plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    raw_mem = str(best_ind[0]); raw_stand = str(best_ind[1])
    simp_mem = parse_sympy(raw_mem); simp_stand = parse_sympy(raw_stand)
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*60}\n"
        f"      COMBINED DR & IR [MODE 2 - ASSESSMENT] REPORT\n"
        f"{'='*60}\n"
        f"--- BEST EVOLVED RULES ---\n"
        f"[MEMORY UPDATE]\nRaw:   {raw_mem}\nSymPy: {simp_mem}\n\n"
        f"[STANDING UPDATE]\nRaw:   {raw_stand}\nSymPy: {simp_stand}\n\n"
        f"--- ABM CONFIGURATIONS ---\n"
        f"Action Logic: {model_config['ACTION_LOGIC']}\n"
        f"iGSS: {model_config['NUM_IGSS']} | UC: {model_config['NUM_UC']} | Defectors: {model_config['NUM_D']}\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline: {control_fit:.1f}\n"
        f"iGSS Max Payoff: {history['max_igss'][-1]:.1f}\n"
        f"Uncond. Cooperators Avg Payoff: {history['uc_scores'][-1]:.1f}\n"
        f"Defectors Avg Payoff: {history['d_scores'][-1]:.1f}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
        f"{'='*60}\n"
    )
    print(report_text)

    fig, ax_plot = plt.subplots(figsize=(10, 6))
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    ax_plot.plot(history["uc_scores"], label='Unconditional Cooperators', color='green', linewidth=1.5, linestyle='--')
    ax_plot.plot(history["d_scores"], label='Defectors', color='red', linewidth=1.5, linestyle='-.')
    ax_plot.axhline(y=control_fit, color='black', linestyle=':', label=f'Control Baseline: {control_fit:.1f}')
    
    ax_plot.set_title(f'Combined DR & IR [Mode 2]: Assessment Evolution (Logic: {model_config["ACTION_LOGIC"]})')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_dual_trees(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)