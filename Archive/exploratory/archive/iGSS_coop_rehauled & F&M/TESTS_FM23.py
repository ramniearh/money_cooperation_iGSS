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
    "NUM_IGSS": 20,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 100,
    # --- CENTRAL BANK CONFIGS ---
    "INITIAL_ENDOWMENT": 1,        
    "ENDOWMENT_FRACTION": 0.5      # 50% Liquidity
}

# Boosted for 2-Tree Co-Evolution (The Handshake)
EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 200,         
    "PARSIMONY_TAX": 0.05
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (DUAL TREE HANDSHAKE)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# --- Tree 1: OFFER RULE (Evaluated by Donor) ---
pset_offer = gp.PrimitiveSet("OfferRule", 2) 
pset_offer.renameArguments(ARG0='MyToken', ARG1='PartnerToken')
pset_offer.addPrimitive(operator.add, 2)
pset_offer.addPrimitive(operator.sub, 2)
pset_offer.addPrimitive(operator.mul, 2)
pset_offer.addEphemeralConstant("rand_const_off", random_constant)

# --- Tree 2: CONSENT RULE (Evaluated by Recipient) ---
pset_consent = gp.PrimitiveSet("ConsentRule", 2) 
pset_consent.renameArguments(ARG0='MyToken', ARG1='PartnerToken')
pset_consent.addPrimitive(operator.add, 2)
pset_consent.addPrimitive(operator.sub, 2)
pset_consent.addPrimitive(operator.mul, 2)
pset_consent.addEphemeralConstant("rand_const_con", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_offer", gp.genHalfAndHalf, pset=pset_offer, min_=1, max_=3)
toolbox.register("expr_consent", gp.genHalfAndHalf, pset=pset_consent, min_=1, max_=3)

def init_individual(container, func1, func2):
    return container([gp.PrimitiveTree(func1()), gp.PrimitiveTree(func2())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_offer, toolbox.expr_consent)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compilers
toolbox.register("compile_offer", gp.compile, pset=pset_offer)
toolbox.register("compile_consent", gp.compile, pset=pset_consent)

# Custom Operators for 2-Tree DNA
def cxTwoTrees(ind1, ind2):
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutTwoTrees(individual):
    if random.random() < 0.5:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_offer, pset=pset_offer)
    else:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_consent, pset=pset_consent)
    return individual,

toolbox.register("mate", cxTwoTrees)
toolbox.register("mutate", mutTwoTrees)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. AGENT-BASED MODEL (Mutual Consent Physics)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.token = 0 
        
    def evaluate_offer(self, partner):
        """Called when this agent is the DONOR."""
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # Hardcoded Baseline: I will offer help if you have money to pay me.
        if self.agent_type == "Control-Agent":
            return partner.token >= 1
            
        decision_score = self.model.igss_offer_rule(self.token, partner.token)
        return decision_score > 0

    def evaluate_consent(self, partner):
        """Called when this agent is the RECIPIENT."""
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return True # Defectors always accept free help
        
        # Hardcoded Baseline: I will consent to receive help if I have money to pay.
        if self.agent_type == "Control-Agent":
            return self.token >= 1
            
        decision_score = self.model.igss_consent_rule(self.token, partner.token)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_offer_rule, igss_consent_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_offer_rule = igss_offer_rule
        self.igss_consent_rule = igss_consent_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-Agent" if control_mode else "iGSS-Agent"
        
        agents_list = []
        for _ in range(config["NUM_IGSS"]): agents_list.append(CoopAgent(self, target_agent))
        for _ in range(config["NUM_UC"]): agents_list.append(CoopAgent(self, "Unconditional Cooperator"))
        for _ in range(config["NUM_D"]): agents_list.append(CoopAgent(self, "Defector"))

        # --- THE CENTRAL BANK: PROPRIETARY FIAT GENESIS ---
        economic_agents = [a for a in agents_list if a.agent_type in ["iGSS-Agent", "Control-Agent"]]
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

            # THE HANDSHAKE PROTOCOL
            donor_offers = donor.evaluate_offer(recipient)
            recipient_consents = recipient.evaluate_consent(donor)
            
            # The transaction ONLY occurs if both parties agree
            if donor_offers and recipient_consents:
                self.resolve_transaction(donor, recipient)
            
    def resolve_transaction(self, donor, recipient):
        # 1. Economic Execution
        donor.payoff -= self.cost
        recipient.payoff += self.benefit
        
        # 2. Institutional Settlement (Mutual Consent Zero-Sum Transfer)
        cares_about_money = donor.agent_type in ["iGSS-Agent", "Control-Agent"] and recipient.agent_type in ["iGSS-Agent", "Control-Agent"]
        
        # Under mutual consent, the token exchange only processes if the recipient 
        # actually has funds to surrender (avoids negative balances).
        if cares_about_money and recipient.token >= 1:
            recipient.token -= 1
            donor.token += 1

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
    func_offer = toolbox.compile_offer(expr=individual[0])
    func_consent = toolbox.compile_consent(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_offer_rule=func_offer, igss_consent_rule=func_consent, config=model_config)
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
    
    print("Initializing MUTUAL CONSENT MONEY [MODE 4] Module...")
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
            history["fossil_record"][gen] = f"OFFER: {best_in_gen[0]}  |  CONSENT: {best_in_gen[1]}"
            print(f" > Gen {gen:02d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

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
        'PartnerToken': sp.Symbol('PartnerToken')
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
    
    # Offer Tree
    n1, e1, l1 = gp.graph(individual[0])
    g1 = nx.DiGraph(); g1.add_nodes_from(n1); g1.add_edges_from(e1)
    try: p1 = hierarchy_pos(g1)
    except TypeError: p1 = nx.spring_layout(g1)
    nx.draw_networkx_nodes(g1, p1, ax=ax1, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g1, p1, ax=ax1, arrows=False)
    nx.draw_networkx_labels(g1, p1, l1, ax=ax1, font_size=10)
    ax1.set_title("Evolved OFFER Rule (Donor)", fontsize=14); ax1.axis("off")
    
    # Consent Tree
    n2, e2, l2 = gp.graph(individual[1])
    g2 = nx.DiGraph(); g2.add_nodes_from(n2); g2.add_edges_from(e2)
    try: p2 = hierarchy_pos(g2)
    except TypeError: p2 = nx.spring_layout(g2)
    nx.draw_networkx_nodes(g2, p2, ax=ax2, node_size=1000, node_color="lightgreen")
    nx.draw_networkx_edges(g2, p2, ax=ax2, arrows=False)
    nx.draw_networkx_labels(g2, p2, l2, ax=ax2, font_size=10)
    ax2.set_title("Evolved CONSENT Rule (Recipient)", fontsize=14); ax2.axis("off")
    
    plt.tight_layout()
    plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    raw_off = str(best_ind[0]); raw_con = str(best_ind[1])
    simp_off = parse_sympy(raw_off); simp_con = parse_sympy(raw_con)
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*70}\n"
        f"      MUTUAL CONSENT MONEY [MODE 4] REPORT\n"
        f"{'='*70}\n"
        f"--- BEST EVOLVED RULES ---\n"
        f"[OFFER]   Raw: {raw_off}\n          SymPy: {simp_off}\n\n"
        f"[CONSENT] Raw: {raw_con}\n          SymPy: {simp_con}\n\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline (Handshake): {control_fit:.1f}\n"
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
    
    ax_plot.set_title('Mutual Consent Money [Mode 4]: Offer & Consent Co-Evolution')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_dual_trees(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)