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
    "NUM_ROUNDS": 50 # Adjust this to 50 if you want to match your previous run!
}

# Boosted to handle 2-Tree Co-Evolution with 4 Assessment Variables
EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 500,         
    "PARSIMONY_TAX": 0.05
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (2-TREE CO-EVOLUTION)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# --- Tree 1: Action Rule ---
pset_act = gp.PrimitiveSet("ActionRule_Combined", 2) 
pset_act.renameArguments(ARG0='PartnerStanding', ARG1='PartnerInMemory')
pset_act.addPrimitive(operator.add, 2)
pset_act.addPrimitive(operator.sub, 2)
pset_act.addPrimitive(operator.mul, 2)
pset_act.addEphemeralConstant("rand_const_act", random_constant)

# --- Tree 2: Universal Assessment Rule (Now with 4 Variables) ---
pset_ass = gp.PrimitiveSet("AssessmentRule_Combined", 4) 
pset_ass.renameArguments(ARG0='DonorAction', ARG1='DonorStanding', ARG2='RecipientStanding', ARG3='DonorInMemory')
pset_ass.addPrimitive(operator.add, 2)
pset_ass.addPrimitive(operator.sub, 2)
pset_ass.addPrimitive(operator.mul, 2)
pset_ass.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_act", gp.genHalfAndHalf, pset=pset_act, min_=1, max_=3)
toolbox.register("expr_ass", gp.genHalfAndHalf, pset=pset_ass, min_=1, max_=3)

def init_individual(container, func_act, func_ass):
    return container([gp.PrimitiveTree(func_act()), gp.PrimitiveTree(func_ass())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_act, toolbox.expr_ass)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compilers
toolbox.register("compile_act", gp.compile, pset=pset_act)
toolbox.register("compile_ass", gp.compile, pset=pset_ass)

# Custom Genetic Operators for 2-Tree Individuals
def cxTwoTrees(ind1, ind2):
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutTwoTrees(individual):
    if random.random() < 0.5:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_act, pset=pset_act)
    else:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_ass, pset=pset_ass)
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
        
        # Hardcoded Baselines
        if self.agent_type == "Control-TFT":
            return partner.unique_id not in self.memory
        if self.agent_type == "Control-Standing":
            return partner.standing == 1
            
        # iGSS EVOLVED ACTION RULE
        in_memory_val = 1 if partner.unique_id in self.memory else 0
        decision_score = self.model.rule_act(partner.standing, in_memory_val)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, rule_act, rule_ass, config=MODEL_CONFIG, control_type=None):
        super().__init__()
        self.rule_act = rule_act
        self.rule_ass = rule_ass
        self.config = config
        self.control_type = control_type
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = control_type if control_type else "iGSS-Agent"
        
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
            
        # 2. Institutional Assessments
        if self.control_type == "Control-TFT":
            if not cooperates:
                recipient.memory.add(donor.unique_id)
                if recipient.unique_id in donor.memory:
                    donor.memory.remove(recipient.unique_id)
                    
        elif self.control_type == "Control-Standing":
            if cooperates:
                donor.standing = 1
            else:
                if recipient.standing == 1:
                    donor.standing = 0
                    
        else:
            # iGSS EVOLVED UNIVERSAL ASSESSMENT RULE (Now with 4 Inputs)
            action_val = 1 if cooperates else 0
            donor_standing = donor.standing
            donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
            
            decision_score = self.rule_ass(action_val, donor_standing, recipient.standing, donor_in_mem)
            
            if decision_score > 0:
                donor.standing = 1
                recipient.memory.discard(donor.unique_id)
            else:
                donor.standing = 0
                recipient.memory.add(donor.unique_id)

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Control-TFT", "Control-Standing", "Unconditional Cooperator", "Defector"]:
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
    func_act = toolbox.compile_act(expr=individual[0])
    func_ass = toolbox.compile_ass(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(rule_act=func_act, rule_ass=func_ass, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    # Combined Parsimony Tax across both trees
    tax = (len(individual[0]) + len(individual[1])) * evo_config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

toolbox.register("evaluate", evaluate_rules, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def get_control_baseline(model_config, control_type):
    tot_fitness = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(rule_act=None, rule_ass=None, config=model_config, control_type=control_type)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_fitness += m.get_fitness_by_type()[control_type]
    return tot_fitness / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Combined [MODE 3] (Rigorous 2-Tree Co-Evolution + 4 Var Assessment)...")
    dr_control_fitness = get_control_baseline(MODEL_CONFIG, "Control-TFT")
    ir_control_fitness = get_control_baseline(MODEL_CONFIG, "Control-Standing")
    print(f"Control Baseline (DR - TFT): {dr_control_fitness:.1f}")
    print(f"Control Baseline (IR - Standing): {ir_control_fitness:.1f}")

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
            history["fossil_record"][gen] = f"ACT: {best_in_gen[0]} | ASSESS: {best_in_gen[1]}"
            print(f" > Gen {gen:03d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, dr_control_fitness, ir_control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def parse_sympy(gp_string, rule_type):
    if rule_type == "ACT":
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'PartnerStanding': sp.Symbol('PartnerStanding'), 'PartnerInMemory': sp.Symbol('PartnerInMemory')}
    else:
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'DonorAction': sp.Symbol('DonorAction'), 'DonorStanding': sp.Symbol('DonorStanding'), 'RecipientStanding': sp.Symbol('RecipientStanding'), 'DonorInMemory': sp.Symbol('DonorInMemory')}
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
    
    # Tree 1: Action
    n1, e1, l1 = gp.graph(individual[0])
    g1 = nx.DiGraph(); g1.add_nodes_from(n1); g1.add_edges_from(e1)
    try: p1 = hierarchy_pos(g1)
    except TypeError: p1 = nx.spring_layout(g1)
    nx.draw_networkx_nodes(g1, p1, ax=ax1, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(g1, p1, ax=ax1, arrows=False)
    nx.draw_networkx_labels(g1, p1, l1, ax=ax1, font_size=10)
    ax1.set_title("Evolved ACTION Rule", fontsize=14); ax1.axis("off")
    
    # Tree 2: Universal Assessment
    n2, e2, l2 = gp.graph(individual[1])
    g2 = nx.DiGraph(); g2.add_nodes_from(n2); g2.add_edges_from(e2)
    try: p2 = hierarchy_pos(g2)
    except TypeError: p2 = nx.spring_layout(g2)
    nx.draw_networkx_nodes(g2, p2, ax=ax2, node_size=1000, node_color="lightgreen")
    nx.draw_networkx_edges(g2, p2, ax=ax2, arrows=False)
    nx.draw_networkx_labels(g2, p2, l2, ax=ax2, font_size=10)
    ax2.set_title("Evolved UNIVERSAL ASSESSMENT Rule", fontsize=14); ax2.axis("off")

    plt.tight_layout(); plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, dr_fit, ir_fit):
    raw_act = str(best_ind[0]); raw_ass = str(best_ind[1])
    simp_act = parse_sympy(raw_act, "ACT"); simp_ass = parse_sympy(raw_ass, "ASSESS")
    fossils_str = "\n".join([f"  Gen {g:03d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*70}\n"
        f"      COMBINED [MODE 3] REPORT (Rigorous Co-Evolution)\n"
        f"{'='*70}\n"
        f"--- BEST EVOLVED RULES ---\n"
        f"[ACTION]\nRaw:   {raw_act}\nSymPy: {simp_act}\n\n"
        f"[ASSESSMENT]\nRaw:   {raw_ass}\nSymPy: {simp_ass}\n\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline (IR - Standing): {ir_fit:.1f}\n"
        f"Control Baseline (DR - TFT): {dr_fit:.1f}\n"
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
    
    ax_plot.axhline(y=ir_fit, color='black', linestyle=':', label=f'IR Baseline (Standing): {ir_fit:.1f}')
    ax_plot.axhline(y=dr_fit, color='grey', linestyle='-.', label=f'DR Baseline (TFT): {dr_fit:.1f}')
    
    ax_plot.set_title('Institutional Competition: Rigorous 2-Tree Co-Evolution')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_dual_trees(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history, dr_fit, ir_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, dr_fit, ir_fit)