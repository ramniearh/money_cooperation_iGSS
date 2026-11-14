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
    "NUM_IGSS": 30,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 100
}

EVO_CONFIG = {
    "POP_SIZE": 500,       # Massively increased for 3-tree search space
    "MAX_GENS": 1000,       # Massively increased runway
    "PARSIMONY_TAX": 0.01
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (TRIPLE TREES)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# --- Tree 1: Action Rule ---
pset_action = gp.PrimitiveSet("ActionRule_Combined", 2) 
pset_action.renameArguments(ARG0='PartnerInMemory', ARG1='PartnerStanding')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

# --- Trees 2 & 3: Assessment Rules ---
pset_assess = gp.PrimitiveSet("AssessmentRule_Combined", 4) 
pset_assess.renameArguments(ARG0='DonorAction', ARG1='DonorInMemory', ARG2='DonorStanding', ARG3='RecipientStanding')
pset_assess.addPrimitive(operator.add, 2)
pset_assess.addPrimitive(operator.sub, 2)
pset_assess.addPrimitive(operator.mul, 2)
pset_assess.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Individual is now a LIST of THREE trees: [Action, Memory, Standing]
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
toolbox.register("expr_assess", gp.genHalfAndHalf, pset=pset_assess, min_=1, max_=3)

def init_individual(container, func_act, func_ass):
    return container([gp.PrimitiveTree(func_act()), gp.PrimitiveTree(func_ass()), gp.PrimitiveTree(func_ass())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_action, toolbox.expr_assess)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compilers
toolbox.register("compile_action", gp.compile, pset=pset_action)
toolbox.register("compile_assess", gp.compile, pset=pset_assess)

# Custom Genetic Operators for 3-Tree DNA
def cxThreeTrees(ind1, ind2):
    roll = random.random()
    if roll < 0.33:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0]) # Crossover Action
    elif roll < 0.66:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1]) # Crossover Memory
    else:
        ind1[2], ind2[2] = gp.cxOnePoint(ind1[2], ind2[2]) # Crossover Standing
    return ind1, ind2

def mutThreeTrees(individual):
    roll = random.random()
    if roll < 0.33:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_action, pset=pset_action)
    elif roll < 0.66:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_assess, pset=pset_assess)
    else:
        individual[2], = gp.mutUniform(individual[2], expr=toolbox.expr_assess, pset=pset_assess)
    return individual,

toolbox.register("mate", cxThreeTrees)
toolbox.register("mutate", mutThreeTrees)
toolbox.register("select", tools.selTournament, tournsize=7)

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
        
        # Hardcoded Control Baseline: Strict Hybrid AND
        if self.agent_type == "Control-Agent":
            return (partner.unique_id not in self.memory) and (partner.standing == 1)
            
        # EVOLVED iGSS ACTION (Tree 0)
        in_memory_val = 1 if partner.unique_id in self.memory else 0
        decision_score = self.model.igss_action_rule(in_memory_val, partner.standing)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_action_rule, igss_mem_rule, igss_stand_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_action_rule = igss_action_rule
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
        if cooperates:
            donor.payoff -= self.cost
            recipient.payoff += self.benefit
            
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
            # EVOLVED iGSS ASSESSMENT (Trees 1 & 2)
            action_val = 1 if cooperates else 0
            donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
            
            # Tree 1: Memory Update
            mem_score = self.igss_mem_rule(action_val, donor_in_mem, donor.standing, recipient.standing)
            if mem_score > 0:
                recipient.memory.add(donor.unique_id)
            else:
                recipient.memory.discard(donor.unique_id)
                
            # Tree 2: Standing Update
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
    func_act = toolbox.compile_action(expr=individual[0])
    func_mem = toolbox.compile_assess(expr=individual[1])
    func_stand = toolbox.compile_assess(expr=individual[2])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(func_act, func_mem, func_stand, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    # Combined Parsimony Tax for all three trees
    tax = (len(individual[0]) + len(individual[1]) + len(individual[2])) * evo_config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

toolbox.register("evaluate", evaluate_rules, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def get_control_baseline(model_config):
    tot_control = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(None, None, None, config=model_config, control_mode=True)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_control += m.get_fitness_by_type()["Control-Agent"]
    return tot_control / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Combined DR & IR [MODE 3 - Total Co-Evolution]...")
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
            history["fossil_record"][gen] = f"ACT: {best_in_gen[0]} | MEM: {best_in_gen[1]} | STAND: {best_in_gen[2]}"
            print(f" > Gen {gen:03d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def parse_sympy(gp_string, rule_type="assess"):
    if rule_type == "action":
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'PartnerInMemory': sp.Symbol('PartnerInMemory'), 'PartnerStanding': sp.Symbol('PartnerStanding')}
    else:
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'DonorAction': sp.Symbol('DonorAction'), 'DonorInMemory': sp.Symbol('DonorInMemory'), 'DonorStanding': sp.Symbol('DonorStanding'), 'RecipientStanding': sp.Symbol('RecipientStanding')}
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

def plot_triple_trees(individual):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    trees = [
        (individual[0], ax1, "lightblue", "Evolved ACTION Rule"),
        (individual[1], ax2, "lightgreen", "Evolved MEMORY Rule"),
        (individual[2], ax3, "lightcoral", "Evolved STANDING Rule")
    ]
    
    for tree, ax, color, title in trees:
        n, e, l = gp.graph(tree)
        g = nx.DiGraph(); g.add_nodes_from(n); g.add_edges_from(e)
        try: p = hierarchy_pos(g)
        except TypeError: p = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, p, ax=ax, node_size=800, node_color=color)
        nx.draw_networkx_edges(g, p, ax=ax, arrows=False)
        nx.draw_networkx_labels(g, p, l, ax=ax, font_size=8)
        ax.set_title(title, fontsize=12); ax.axis("off")
        
    plt.tight_layout()
    plt.show(block=False)

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    simp_act = parse_sympy(str(best_ind[0]), "action")
    simp_mem = parse_sympy(str(best_ind[1]))
    simp_stand = parse_sympy(str(best_ind[2]))
    
    fossils_str = "\n".join([f"  Gen {g:03d}: {history['fossil_record'][g]}" for g in sorted(history["fossil_record"].keys())])
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*70}\n"
        f"      COMBINED DR & IR [MODE 3 - TOTAL CO-EVOLUTION] REPORT\n"
        f"{'='*70}\n"
        f"--- BEST EVOLVED RULES (SymPy) ---\n"
        f"[ACTION]:  {simp_act}\n"
        f"[MEMORY]:  {simp_mem}\n"
        f"[STANDING]: {simp_stand}\n\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline (Hybrid AND): {control_fit:.1f}\n"
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
    
    ax_plot.set_title('Combined DR & IR [Mode 3]: Total Co-Evolution')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show(block=False)
    plot_triple_trees(best_ind); plt.show() 

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)