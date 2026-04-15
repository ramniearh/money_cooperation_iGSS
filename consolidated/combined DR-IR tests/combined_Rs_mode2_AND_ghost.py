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

# Boosted for 2-Tree Co-Evolution
EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.05
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (DUAL ASSESSMENT RULES)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# --- Tree 1: IR Assessment Rule (Public Ledger) ---
pset_ir = gp.PrimitiveSet("AssessmentRule_IR", 2) 
pset_ir.renameArguments(ARG0='DonorAction', ARG1='RecipientStanding')
pset_ir.addPrimitive(operator.add, 2)
pset_ir.addPrimitive(operator.sub, 2)
pset_ir.addPrimitive(operator.mul, 2)
pset_ir.addEphemeralConstant("rand_const_ir", random_constant)

# --- Tree 2: DR Assessment Rule (Private Ledger) ---
pset_dr = gp.PrimitiveSet("AssessmentRule_DR", 2) 
pset_dr.renameArguments(ARG0='DonorAction', ARG1='DonorInMemory')
pset_dr.addPrimitive(operator.add, 2)
pset_dr.addPrimitive(operator.sub, 2)
pset_dr.addPrimitive(operator.mul, 2)
pset_dr.addEphemeralConstant("rand_const_dr", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_ir", gp.genHalfAndHalf, pset=pset_ir, min_=1, max_=3)
toolbox.register("expr_dr", gp.genHalfAndHalf, pset=pset_dr, min_=1, max_=3)

def init_individual(container, func_ir, func_dr):
    return container([gp.PrimitiveTree(func_ir()), gp.PrimitiveTree(func_dr())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_ir, toolbox.expr_dr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile_ir", gp.compile, pset=pset_ir)
toolbox.register("compile_dr", gp.compile, pset=pset_dr)

# Custom Genetic Operators for 2-Tree Individuals
def cxTwoTrees(ind1, ind2):
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutTwoTrees(individual):
    if random.random() < 0.5:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_ir, pset=pset_ir)
    else:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_dr, pset=pset_dr)
    return individual,

toolbox.register("mate", cxTwoTrees)
toolbox.register("mutate", mutTwoTrees)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. AGENT-BASED MODEL (Dual Ledgers)
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
            
        # iGSS FIXED ACTION RULE: The Paranoid Hybrid
        # Only cooperates if society says they are Good AND you hold no personal grudge
        return (partner.standing == 1) and (partner.unique_id not in self.memory)

class CooperationModel(mesa.Model):
    def __init__(self, rule_ir, rule_dr, config=MODEL_CONFIG, control_type=None):
        super().__init__()
        self.rule_ir = rule_ir
        self.rule_dr = rule_dr
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
            # iGSS EXECUTES BOTH EVOLVED ASSESSMENT TREES
            action_val = 1 if cooperates else 0
            
            # Tree 1: Evaluate Public Standing
            ir_score = self.rule_ir(action_val, recipient.standing)
            donor.standing = 1 if ir_score > 0 else 0
            
            # Tree 2: Evaluate Private Memory
            donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
            dr_score = self.rule_dr(action_val, donor_in_mem)
            if dr_score > 0:
                recipient.memory.add(donor.unique_id)
            else:
                recipient.memory.discard(donor.unique_id)

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
    func_ir = toolbox.compile_ir(expr=individual[0])
    func_dr = toolbox.compile_dr(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(rule_ir=func_ir, rule_dr=func_dr, config=model_config)
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

def get_control_baseline(model_config, control_type):
    tot_fitness = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(rule_ir=None, rule_dr=None, config=model_config, control_type=control_type)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_fitness += m.get_fitness_by_type()[control_type]
    return tot_fitness / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Combined [MODE 2] (Dual-Assessment Search)...")
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
            history["fossil_record"][gen] = f"IR: {best_in_gen[0]}  |  DR: {best_in_gen[1]}"
            print(f" > Gen {gen:02d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, dr_control_fitness, ir_control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def parse_sympy(gp_string, rule_type):
    if rule_type == "IR":
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'DonorAction': sp.Symbol('DonorAction'), 'RecipientStanding': sp.Symbol('RecipientStanding')}
    else:
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'DonorAction': sp.Symbol('DonorAction'), 'DonorInMemory': sp.Symbol('DonorInMemory')}
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
    nx.draw_networkx_nodes(g1, p1, ax=ax1, node_size=1000, node_color="lightgreen")
    nx.draw_networkx_edges(g1, p1, ax=ax1, arrows=False)
    nx.draw_networkx_labels(g1, p1, l1, ax=ax1, font_size=10)
    ax1.set_title("Evolved IR Assessment (Public Ledger)", fontsize=14); ax1.axis("off")
    
    n2, e2, l2 = gp.graph(individual[1])
    g2 = nx.DiGraph(); g2.add_nodes_from(n2); g2.add_edges_from(e2)
    try: p2 = hierarchy_pos(g2)
    except TypeError: p2 = nx.spring_layout(g2)
    nx.draw_networkx_nodes(g2, p2, ax=ax2, node_size=1000, node_color="orange")
    nx.draw_networkx_edges(g2, p2, ax=ax2, arrows=False)
    nx.draw_networkx_labels(g2, p2, l2, ax=ax2, font_size=10)
    ax2.set_title("Evolved DR Assessment (Private Ledger)", fontsize=14); ax2.axis("off")
    
    plt.tight_layout(); plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, dr_fit, ir_fit):
    raw_ir = str(best_ind[0]); raw_dr = str(best_ind[1])
    simp_ir = parse_sympy(raw_ir, "IR"); simp_dr = parse_sympy(raw_dr, "DR")
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*70}\n"
        f"      COMBINED [MODE 2] REPORT (Institutional Competition)\n"
        f"{'='*70}\n"
        f"--- BEST EVOLVED ASSESSMENTS ---\n"
        f"[IR - Public Ledger]\nRaw:   {raw_ir}\nSymPy: {simp_ir}\n\n"
        f"[DR - Private Ledger]\nRaw:   {raw_dr}\nSymPy: {simp_dr}\n\n"
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
    
    ax_plot.set_title('Institutional Competition: Evolving Dual Assessments')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_dual_trees(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history, dr_fit, ir_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, dr_fit, ir_fit)