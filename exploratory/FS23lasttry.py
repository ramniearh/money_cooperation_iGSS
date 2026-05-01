import operator
import random
import math
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
    "NUM_ROUNDS": 100,
    # High liquidity to grease the wheels of a strict consensus economy
    "INITIAL_ENDOWMENT": 1,        
    "ENDOWMENT_FRACTION": 0.5      
}

EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (THE CONSENSUS TREE)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# Single Tree: Evaluates to an INTEGER representing the Desired Token Delta
# >0 means "I demand X tokens"
# <0 means "I offer X tokens"
pset = gp.PrimitiveSet("ConsensusRule", 3) 
pset.renameArguments(ARG0='MyToken', ARG1='PartnerToken', ARG2='MyRole')
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
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL (Strict F&M Consensus Physics)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.token = 0 
        
    def evaluate_consensus(self, partner_token, my_role):
        if self.agent_type == "Unconditional Cooperator": 
            return 0 # Demands 0 to help, offers 0 to receive help
        if self.agent_type == "Defector": 
            # Absurd demand to help, offers 0 to receive help
            return 100 if my_role == 1 else 0 
            
        if self.agent_type == "Control-Merchant":
            # The perfect baseline: Demands 1 as Donor, Offers 1 as Recipient
            return 1 if my_role == 1 else -1
            
        # EVOLVED iGSS RULE
        try:
            raw_val = self.model.igss_rule(self.token, partner_token, my_role)
            # Clip to prevent infinity math crashes
            if math.isnan(raw_val) or math.isinf(raw_val): return 0
            return int(np.clip(raw_val, -10, 10))
        except:
            return 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-Merchant" if control_mode else "iGSS-Agent"
        
        agents_list = []
        for _ in range(config["NUM_IGSS"]): agents_list.append(CoopAgent(self, target_agent))
        for _ in range(config["NUM_UC"]): agents_list.append(CoopAgent(self, "Unconditional Cooperator"))
        for _ in range(config["NUM_D"]): agents_list.append(CoopAgent(self, "Defector"))

        # Fiat Genesis (No defector funding)
        economic_agents = [a for a in agents_list if a.agent_type in ["iGSS-Agent", "Control-Merchant"]]
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

            # THE F&M CONSENSUS CHECK
            # Donor uses role = 1
            donor_demand = donor.evaluate_consensus(recipient.token, my_role=1)
            # Recipient uses role = -1
            recipient_offer = recipient.evaluate_consensus(donor.token, my_role=-1)
            
            # 1. Do their mathematical equations perfectly mirror each other?
            if donor_demand == -recipient_offer:
                transfer = donor_demand
                
                cares_about_money = donor.agent_type in ["iGSS-Agent", "Control-Merchant"] and recipient.agent_type in ["iGSS-Agent", "Control-Merchant"]
                
                valid_trade = True
                if cares_about_money:
                    # 2. Do they actually have the funds they just mathematically agreed upon?
                    if transfer > 0 and recipient.token < transfer: valid_trade = False
                    if transfer < 0 and donor.token < abs(transfer): valid_trade = False
                
                # 3. Execute the Consented State Transition
                if valid_trade:
                    donor.payoff -= self.cost
                    recipient.payoff += self.benefit
                    
                    if cares_about_money and transfer != 0:
                        donor.token += transfer
                        recipient.token -= transfer

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Control-Merchant", "Unconditional Cooperator", "Defector"]:
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
    func = toolbox.compile(expr=individual)
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_rule=func, config=model_config)
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
        m = CooperationModel(None, config=model_config, control_mode=True)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_control += m.get_fitness_by_type()["Control-Merchant"]
    return tot_control / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing CONSENSUS PROTOCOL [MODE 7] Module...")
    control_fitness = get_control_baseline(MODEL_CONFIG)
    print(f"Control Baseline (Strict Merchants) established at: {control_fitness:.2f} avg payoff.")

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
            print(f" > Gen {gen:03d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def parse_sympy(gp_string):
    mapping = {
        'add': lambda x, y: x + y, 
        'sub': lambda x, y: x - y, 
        'mul': lambda x, y: x * y,
        'MyToken': sp.Symbol('MyToken'),
        'PartnerToken': sp.Symbol('PartnerToken'),
        'MyRole': sp.Symbol('MyRole')
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

def plot_tree(individual):
    fig, ax = plt.subplots(figsize=(10, 8))
    n, e, l = gp.graph(individual)
    g = nx.DiGraph(); g.add_nodes_from(n); g.add_edges_from(e)
    try: p = hierarchy_pos(g)
    except TypeError: p = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, p, ax=ax, node_size=1000, node_color="gold")
    nx.draw_networkx_edges(g, p, ax=ax, arrows=False)
    nx.draw_networkx_labels(g, p, l, ax=ax, font_size=9)
    ax.set_title("Evolved Consensus Rule (F&M Protocol)", fontsize=14); ax.axis("off")
    plt.tight_layout(); plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    raw_rule = str(best_ind)
    simp_rule = parse_sympy(raw_rule)
    fossils_str = "\n".join([f"  Gen {g:03d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*70}\n"
        f"      CONSENSUS PROTOCOL [MODE 7] REPORT\n"
        f"{'='*70}\n"
        f"--- BEST EVOLVED RULE ---\n"
        f"Raw:   {raw_rule}\n"
        f"SymPy: {simp_rule}\n\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline (Merchants): {control_fit:.1f}\n"
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
    ax_plot.axhline(y=control_fit, color='black', linestyle=':', label=f'Control Baseline: {control_fit:.1f}')
    
    ax_plot.set_title('Consensus Protocol [Mode 7]: Endogenous Fiat')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_tree(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)