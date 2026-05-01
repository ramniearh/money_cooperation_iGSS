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
    "BENEFIT_TO_COST_RATIO": 2,
    "COST": 1,
    "NUM_IGSS": 30, 
    "NUM_UC": 10,    # Active
    "NUM_D": 10,     # Active
    "NUM_ROUNDS": 100,
    "INITIAL_ENDOWMENT": 2,        
    "ENDOWMENT_FRACTION": 0.8      
}

EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.05,
    "TOURNAMENT_SIZE": 3,
    "MUTATION_RATE": 0.2,
    "CROSSOVER_RATE": 0.5,
    "TREE_MIN_DEPTH": 1,
    "TREE_MAX_DEPTH": 3
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (CLAMPED TREES)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

pset_donor = gp.PrimitiveSet("DonorRule", 1) 
pset_donor.renameArguments(ARG0='MyToken')
pset_donor.addPrimitive(operator.add, 2)
pset_donor.addPrimitive(operator.sub, 2)
pset_donor.addPrimitive(operator.mul, 2)
pset_donor.addEphemeralConstant("rand_const_d", random_constant)

pset_recipient = gp.PrimitiveSet("RecipientRule", 1) 
pset_recipient.renameArguments(ARG0='MyToken')
pset_recipient.addPrimitive(operator.add, 2)
pset_recipient.addPrimitive(operator.sub, 2)
pset_recipient.addPrimitive(operator.mul, 2)
pset_recipient.addEphemeralConstant("rand_const_r", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_donor", gp.genHalfAndHalf, pset=pset_donor, min_=EVO_CONFIG["TREE_MIN_DEPTH"], max_=EVO_CONFIG["TREE_MAX_DEPTH"])
toolbox.register("expr_recipient", gp.genHalfAndHalf, pset=pset_recipient, min_=EVO_CONFIG["TREE_MIN_DEPTH"], max_=EVO_CONFIG["TREE_MAX_DEPTH"])

def init_individual(container, func1, func2):
    return container([gp.PrimitiveTree(func1()), gp.PrimitiveTree(func2())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_donor, toolbox.expr_recipient)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile_donor", gp.compile, pset=pset_donor)
toolbox.register("compile_recipient", gp.compile, pset=pset_recipient)

def cxTwoTrees(ind1, ind2):
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutTwoTrees(individual):
    if random.random() < 0.5:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_donor, pset=pset_donor)
    else:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_recipient, pset=pset_recipient)
    return individual,

toolbox.register("mate", cxTwoTrees)
toolbox.register("mutate", mutTwoTrees)
toolbox.register("select", tools.selTournament, tournsize=EVO_CONFIG["TOURNAMENT_SIZE"])

# =============================================================================
# 3. AGENT-BASED MODEL (Pure Ontology Collision)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.token = 0 
        
    def evaluate_donor_tree(self):
        # ONLY runs for iGSS agents.
        try:
            raw_val = self.model.igss_donor_rule(self.token)
            if raw_val > 0: return 1
            elif raw_val < 0: return -1
            else: return 0
        except: return 0

    def evaluate_recipient_tree(self):
        # ONLY runs for iGSS agents.
        try:
            raw_val = self.model.igss_recipient_rule(self.token)
            if raw_val > 0: return 1
            elif raw_val < 0: return -1
            else: return 0
        except: return 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_donor_rule, igss_recipient_rule, config=MODEL_CONFIG):
        super().__init__()
        self.igss_donor_rule = igss_donor_rule
        self.igss_recipient_rule = igss_recipient_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        for _ in range(config["NUM_IGSS"]): self.agents.add(CoopAgent(self, "iGSS-Agent"))
        for _ in range(config["NUM_UC"]): self.agents.add(CoopAgent(self, "Unconditional Cooperator"))
        for _ in range(config["NUM_D"]): self.agents.add(CoopAgent(self, "Defector"))

        # Fiat Genesis (Tokens only exist for iGSS agents)
        economic_agents = [a for a in self.agents if a.agent_type == "iGSS-Agent"]
        random.shuffle(economic_agents)
        cutoff = int(len(economic_agents) * config["ENDOWMENT_FRACTION"])
        for i, agent in enumerate(economic_agents):
            if i < cutoff:
                agent.token = config["INITIAL_ENDOWMENT"]

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for donor in agents:
            possible_recipients = [a for a in agents if a.unique_id != donor.unique_id]
            recipient = random.choice(possible_recipients)

            # 1. UC Donor Logic (Unconditional Payload)
            if donor.agent_type == "Unconditional Cooperator":
                self.execute_payload(donor, recipient)
            
            # 2. Defector Donor Logic (Does nothing)
            elif donor.agent_type == "Defector":
                pass
                
            # 3. iGSS Donor Logic (Attempts Consensus Protocol)
            elif donor.agent_type == "iGSS-Agent":
                donor_demand = donor.evaluate_donor_tree()
                
                # Interaction with another iGSS Agent
                if recipient.agent_type == "iGSS-Agent":
                    recipient_offer = recipient.evaluate_recipient_tree()
                    if donor_demand + recipient_offer == 0:
                        self.execute_transfer_and_payload(donor, recipient, donor_demand)
                
                # Interaction with "Barter" Agents (UC or D)
                else:
                    # The institutional handshake fails unless the iGSS agent is offering free charity (0)
                    if donor_demand == 0:
                        self.execute_payload(donor, recipient)

    def execute_payload(self, donor, recipient):
        donor.payoff -= self.cost
        recipient.payoff += self.benefit

    def execute_transfer_and_payload(self, donor, recipient, transfer_val):
        # Strict Fiat Constraint: No Negative Balances
        if transfer_val == 1 and recipient.token < 1: return
        if transfer_val == -1 and donor.token < 1: return

        # Execute
        self.execute_payload(donor, recipient)
        if transfer_val != 0:
            donor.token += transfer_val
            recipient.token -= transfer_val

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Unconditional Cooperator", "Defector"]:
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
    func_donor = toolbox.compile_donor(expr=individual[0])
    func_recipient = toolbox.compile_recipient(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_donor_rule=func_donor, igss_recipient_rule=func_recipient, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    tax = (len(individual[0]) + len(individual[1])) * evo_config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

toolbox.register("evaluate", evaluate_rules, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing CLAMPED CONSENSUS [PURE ONTOLOGY] Module...")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    
    hof.update(pop)
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < EVO_CONFIG["CROSSOVER_RATE"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values; del child2.fitness.values

        for mutant in offspring:
            if random.random() < EVO_CONFIG["MUTATION_RATE"]:
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
            history["fossil_record"][gen] = f"DONOR: {best_in_gen[0]}  |  RECIPIENT: {best_in_gen[1]}"
            print(f" > Gen {gen:03d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def parse_sympy(gp_string):
    mapping = {
        'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y, 'MyToken': sp.Symbol('MyToken')
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
    
    n1, e1, l1 = gp.graph(individual[0])
    g1 = nx.DiGraph(); g1.add_nodes_from(n1); g1.add_edges_from(e1)
    try: p1 = hierarchy_pos(g1)
    except TypeError: p1 = nx.spring_layout(g1)
    nx.draw_networkx_nodes(g1, p1, ax=ax1, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g1, p1, ax=ax1, arrows=False)
    nx.draw_networkx_labels(g1, p1, l1, ax=ax1, font_size=10)
    ax1.set_title("Evolved DONOR Rule", fontsize=14); ax1.axis("off")
    
    n2, e2, l2 = gp.graph(individual[1])
    g2 = nx.DiGraph(); g2.add_nodes_from(n2); g2.add_edges_from(e2)
    try: p2 = hierarchy_pos(g2)
    except TypeError: p2 = nx.spring_layout(g2)
    nx.draw_networkx_nodes(g2, p2, ax=ax2, node_size=1000, node_color="lightgreen")
    nx.draw_networkx_edges(g2, p2, ax=ax2, arrows=False)
    nx.draw_networkx_labels(g2, p2, l2, ax=ax2, font_size=10)
    ax2.set_title("Evolved RECIPIENT Rule", fontsize=14); ax2.axis("off")
    
    plt.tight_layout()
    plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config):
    raw_d = str(best_ind[0]); raw_r = str(best_ind[1])
    simp_d = parse_sympy(raw_d); simp_r = parse_sympy(raw_r)
    fossils_str = "\n".join([f"  Gen {g:03d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*70}\n"
        f"      CLAMPED CONSENSUS [PURE ONTOLOGY] REPORT\n"
        f"{'='*70}\n"
        f"--- ECONOMIC PHYSICS (ASSUMPTIONS) ---\n"
        f"Defector Constraint: Unconditional Defection (No Tokens)\n"
        f"UC Constraint:       Unconditional Cooperation (No Tokens)\n"
        f"Fiat Constraint:     No Negative Balances Allowed\n"
        f"Init Endowment:      {model_config['INITIAL_ENDOWMENT']}\n"
        f"Endowment Fraction:  {model_config['ENDOWMENT_FRACTION']} (Liquidity)\n\n"
        f"--- EVOLUTIONARY PHYSICS (ASSUMPTIONS) ---\n"
        f"Population Size:     {evo_config['POP_SIZE']}\n"
        f"Max Generations:     {evo_config['MAX_GENS']}\n"
        f"Tournament Size:     {evo_config['TOURNAMENT_SIZE']}\n"
        f"Mutation/Crossover:  {evo_config['MUTATION_RATE']} / {evo_config['CROSSOVER_RATE']}\n"
        f"Parsimony Tax:       {evo_config['PARSIMONY_TAX']}\n"
        f"Tree Depth Bounds:   {evo_config['TREE_MIN_DEPTH']} to {evo_config['TREE_MAX_DEPTH']}\n\n"
        f"--- BEST EVOLVED RULES ---\n"
        f"[DONOR]     Raw: {raw_d}\n            SymPy: {simp_d}\n\n"
        f"[RECIPIENT] Raw: {raw_r}\n            SymPy: {simp_r}\n\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"iGSS Max Payoff: {history['max_igss'][-1]:.1f}\n"
        f"Uncond. Cooperators Avg Payoff: {history['uc_scores'][-1]:.1f}\n"
        f"Defectors Avg Payoff: {history['d_scores'][-1]:.1f}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
        f"{'='*70}\n"
    )
    print(report_text)

    fig, ax_plot = plt.subplots(figsize=(10, 6))
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    ax_plot.plot(history["uc_scores"], label='Unconditional Cooperators', color='green', linewidth=1.5, linestyle='--')
    ax_plot.plot(history["d_scores"], label='Defectors', color='red', linewidth=1.5, linestyle='-.')
    
    ax_plot.set_title('Clamped Consensus [Pure Ontology]')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_dual_trees(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG)