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
    "NUM_UC": 10,    
    "NUM_D": 10,
    "NUM_ROUNDS": 100,
    "INITIAL_ENDOWMENT": 1,        
    "ENDOWMENT_FRACTION": 0.5,
    "LIQUIDITY_PREMIUM": 0.00 # Micro-fitness for holding tokens (The Breadcrumb Trail)
}

# Massive search space requires high population and generations
EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.05
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (DECOUPLED BID & ASK)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# --- Tree 1: THE BID RULE (Initiator/Buyer) ---
# Output is parsed as an INTEGER (Tokens to offer)
pset_bid = gp.PrimitiveSet("BidRule", 2) 
pset_bid.renameArguments(ARG0='MyToken', ARG1='PartnerToken')
pset_bid.addPrimitive(operator.add, 2)
pset_bid.addPrimitive(operator.sub, 2)
pset_bid.addPrimitive(operator.mul, 2)
pset_bid.addEphemeralConstant("rand_const_bid", random_constant)

# --- Tree 2: THE COOP RULE (Responder/Seller) ---
# Output is parsed as a BOOLEAN (>0 means Yes)
pset_coop = gp.PrimitiveSet("CoopRule", 3) 
pset_coop.renameArguments(ARG0='MyToken', ARG1='PartnerToken', ARG2='TokensOffered')
pset_coop.addPrimitive(operator.add, 2)
pset_coop.addPrimitive(operator.sub, 2)
pset_coop.addPrimitive(operator.mul, 2)
pset_coop.addEphemeralConstant("rand_const_coop", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_bid", gp.genHalfAndHalf, pset=pset_bid, min_=1, max_=3)
toolbox.register("expr_coop", gp.genHalfAndHalf, pset=pset_coop, min_=1, max_=3)

def init_individual(container, func_bid, func_coop):
    return container([gp.PrimitiveTree(func_bid()), gp.PrimitiveTree(func_coop())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_bid, toolbox.expr_coop)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile_bid", gp.compile, pset=pset_bid)
toolbox.register("compile_coop", gp.compile, pset=pset_coop)

# Custom Operators
def cxTwoTrees(ind1, ind2):
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutTwoTrees(individual):
    if random.random() < 0.5:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_bid, pset=pset_bid)
    else:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_coop, pset=pset_coop)
    return individual,

toolbox.register("mate", cxTwoTrees)
toolbox.register("mutate", mutTwoTrees)
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL (Escrow Physics)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.token = 0 
        
    def evaluate_bid(self, partner):
        """As Initiator: How many tokens am I willing to put in Escrow for your help?"""
        if self.agent_type == "Unconditional Cooperator": return 0 # Expects free help
        if self.agent_type == "Defector": return 0 # Never pays
        
        # Hardcoded Baseline (The Merchant): I will bid 1 token if I have it
        if self.agent_type == "Control-Agent":
            return 1 if self.token >= 1 else 0
            
        # EVOLVED iGSS RULE
        raw_bid = self.model.igss_bid_rule(self.token, partner.token)
        
        # Physics constraint: You cannot bid more than you have, or less than 0
        return max(0, min(self.token, int(raw_bid)))

    def evaluate_coop(self, partner, tokens_offered):
        """As Responder: Will I pay the cost to help you, given what is in Escrow?"""
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # Hardcoded Baseline (The Merchant): I will help if you pay me at least 1 token
        if self.agent_type == "Control-Agent":
            return tokens_offered >= 1
            
        # EVOLVED iGSS RULE
        decision_score = self.model.igss_coop_rule(self.token, partner.token, tokens_offered)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_bid_rule, igss_coop_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_bid_rule = igss_bid_rule
        self.igss_coop_rule = igss_coop_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-Agent" if control_mode else "iGSS-Agent"
        
        agents_list = []
        for _ in range(config["NUM_IGSS"]): agents_list.append(CoopAgent(self, target_agent))
        for _ in range(config["NUM_UC"]): agents_list.append(CoopAgent(self, "Unconditional Cooperator"))
        for _ in range(config["NUM_D"]): agents_list.append(CoopAgent(self, "Defector"))

        # Fiat Genesis
        economic_agents = [a for a in agents_list if a.agent_type in ["iGSS-Agent", "Control-Agent"]]
        random.shuffle(economic_agents)
        cutoff = int(len(economic_agents) * config["ENDOWMENT_FRACTION"])
        for i, agent in enumerate(economic_agents):
            if i < cutoff:
                agent.token = config["INITIAL_ENDOWMENT"]

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for initiator in agents:
            possible_responders = [a for a in agents if a.unique_id != initiator.unique_id]
            responder = random.choice(possible_responders)

            # THE ESCROW PROTOCOL
            # 1. Initiator sets terms
            tokens_in_escrow = initiator.evaluate_bid(responder)
            
            # 2. Responder evaluates terms
            responder_accepts = responder.evaluate_coop(initiator, tokens_in_escrow)
            
            # 3. Atomic Settlement
            if responder_accepts:
                # Economic Payload
                responder.payoff -= self.cost    # Responder pays the cost of helping
                initiator.payoff += self.benefit # Initiator receives the help
                
                # Fiat Transfer (Only if tokens were actually bid)
                if tokens_in_escrow > 0:
                    initiator.token -= tokens_in_escrow
                    responder.token += tokens_in_escrow

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Control-Agent", "Unconditional Cooperator", "Defector"]:
            type_agents = [a for a in self.agents if a.agent_type == a_type]
            if type_agents:
                # Calculate base payoff + Liquidity Premium
                total_fitness = sum(a.payoff + (a.token * self.config["LIQUIDITY_PREMIUM"]) for a in type_agents)
                fitness_data[a_type] = total_fitness / len(type_agents)
            else:
                fitness_data[a_type] = 0
        return fitness_data

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rules(individual, model_config, evo_config):
    func_bid = toolbox.compile_bid(expr=individual[0])
    func_coop = toolbox.compile_coop(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_bid_rule=func_bid, igss_coop_rule=func_coop, config=model_config)
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

def get_control_baseline(model_config):
    tot_control = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(None, None, config=model_config, control_mode=True)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_control += m.get_fitness_by_type()["Control-Agent"]
    return tot_control / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing ENDOGENOUS FIAT DISCOVERY [MODE 6] Module...")
    control_fitness = get_control_baseline(MODEL_CONFIG)
    print(f"Control Baseline (Merchant Agents) established at: {control_fitness:.2f} avg payoff.")

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
            history["fossil_record"][gen] = f"BID: {best_in_gen[0]}  |  COOP: {best_in_gen[1]}"
            print(f" > Gen {gen:03d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def parse_sympy(gp_string, rule_type):
    if rule_type == "bid":
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'MyToken': sp.Symbol('MyToken'), 'PartnerToken': sp.Symbol('PartnerToken')}
    else:
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'MyToken': sp.Symbol('MyToken'), 'PartnerToken': sp.Symbol('PartnerToken'), 'TokensOffered': sp.Symbol('TokensOffered')}
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
    
    # Bid Tree
    n1, e1, l1 = gp.graph(individual[0])
    g1 = nx.DiGraph(); g1.add_nodes_from(n1); g1.add_edges_from(e1)
    try: p1 = hierarchy_pos(g1)
    except TypeError: p1 = nx.spring_layout(g1)
    nx.draw_networkx_nodes(g1, p1, ax=ax1, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g1, p1, ax=ax1, arrows=False)
    nx.draw_networkx_labels(g1, p1, l1, ax=ax1, font_size=10)
    ax1.set_title("Evolved BID Rule (Initiator)", fontsize=14); ax1.axis("off")
    
    # Coop Tree
    n2, e2, l2 = gp.graph(individual[1])
    g2 = nx.DiGraph(); g2.add_nodes_from(n2); g2.add_edges_from(e2)
    try: p2 = hierarchy_pos(g2)
    except TypeError: p2 = nx.spring_layout(g2)
    nx.draw_networkx_nodes(g2, p2, ax=ax2, node_size=1000, node_color="lightgreen")
    nx.draw_networkx_edges(g2, p2, ax=ax2, arrows=False)
    nx.draw_networkx_labels(g2, p2, l2, ax=ax2, font_size=10)
    ax2.set_title("Evolved COOP Rule (Responder)", fontsize=14); ax2.axis("off")
    
    plt.tight_layout()
    plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    raw_bid = str(best_ind[0]); raw_coop = str(best_ind[1])
    simp_bid = parse_sympy(raw_bid, "bid"); simp_coop = parse_sympy(raw_coop, "coop")
    fossils_str = "\n".join([f"  Gen {g:03d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*70}\n"
        f"      ENDOGENOUS FIAT DISCOVERY [MODE 6] REPORT\n"
        f"{'='*70}\n"
        f"--- BEST EVOLVED RULES ---\n"
        f"[BID]  Raw: {raw_bid}\n       SymPy: {simp_bid}\n\n"
        f"[COOP] Raw: {raw_coop}\n       SymPy: {simp_coop}\n\n"
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
    
    ax_plot.set_title('Endogenous Fiat Discovery [Mode 6]')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_dual_trees(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)