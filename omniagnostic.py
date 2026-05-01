import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATION (The Omni-Environment)
# =============================================================================
MODEL_CONFIG = {
    "BENCHMARK_MODE": True,  # True = 10 Omni / 10 UC / 10 D. False = 3 Tribes of 10 Omni.
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "TRIBE_SIZE": 10,
    "NUM_TRIBES": 3,
    "NUM_ROUNDS": 250,
    "INITIAL_TOKENS": 1
}

EVO_CONFIG = {
    "MAX_GENS": 500,         
    "PARSIMONY_TAX": 0.1  
}

# =============================================================================
# 2. GP SETUP (Dual Tree - Action and Assessment)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# TREE 1: Action Rule (4 Inputs)
pset_act = gp.PrimitiveSet("ActionRule_Omni", 4) 
pset_act.renameArguments(ARG0='PartnerMemory', ARG1='PartnerRep', ARG2='MyTokens', ARG3='PartnerHasToken')
pset_act.addPrimitive(operator.add, 2); pset_act.addPrimitive(operator.sub, 2); pset_act.addPrimitive(operator.mul, 2)
pset_act.addEphemeralConstant("rand_const_act", random_constant)

# TREE 2: Assessment Rule (3 Inputs)
pset_ass = gp.PrimitiveSet("AssessmentRule_Omni", 3)
pset_ass.renameArguments(ARG0='ActionTaken', ARG1='RecipientRep', ARG2='HelperRep')
pset_ass.addPrimitive(operator.add, 2); pset_ass.addPrimitive(operator.sub, 2); pset_ass.addPrimitive(operator.mul, 2)
pset_ass.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_act", gp.genHalfAndHalf, pset=pset_act, min_=1, max_=3)
toolbox.register("expr_ass", gp.genHalfAndHalf, pset=pset_ass, min_=1, max_=3)
toolbox.register("tree_act", tools.initIterate, gp.PrimitiveTree, toolbox.expr_act)
toolbox.register("tree_ass", tools.initIterate, gp.PrimitiveTree, toolbox.expr_ass)

def init_dual(icls): return icls([toolbox.tree_act(), toolbox.tree_ass()])
toolbox.register("individual", init_dual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def cx_dual(ind1, ind2):
    if random.random() < 0.5: ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else: ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mut_dual(ind):
    if random.random() < 0.5: ind[0], = gp.mutUniform(ind[0], expr=toolbox.expr_act, pset=pset_act)
    else: ind[1], = gp.mutUniform(ind[1], expr=toolbox.expr_ass, pset=pset_ass)
    return ind,

toolbox.register("mate", cx_dual)
toolbox.register("mutate", mut_dual)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. MESA MODEL (The Unified Physics Engine)
# =============================================================================
class OmniAgent(mesa.Agent):
    def __init__(self, model, global_index, agent_type, rules=None):
        super().__init__(model) 
        self.global_index = global_index
        self.agent_type = agent_type
        
        if rules:
            self.action_rule = gp.compile(rules[0], pset_act)
            self.assessment_rule = gp.compile(rules[1], pset_ass)
            
        self.payoff = 0                             
        self.tokens = MODEL_CONFIG["INITIAL_TOKENS"]
        self.reputation = 1 # Public IR Ledger (1=Good, 0=Bad)
        self.memory = set() # Private DR Ledger (Set of unique_ids who helped me)
        
    def decide(self, partner):
        if self.agent_type == "UC": return True
        if self.agent_type == "D": return False
        
        in_mem = 1 if partner.unique_id in self.memory else 0
        has_tok = 1 if partner.tokens >= 1 else 0
        return self.action_rule(in_mem, partner.reputation, self.tokens, has_tok) > 0

class OmniModel(mesa.Model):
    def __init__(self, compiled_rules, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.benefit, self.cost = config["BENEFIT_TO_COST_RATIO"] * config["COST"], config["COST"]
        
        if config["BENCHMARK_MODE"]:
            for i, r in enumerate(compiled_rules): OmniAgent(self, i, "Omni", r)
            for i in range(config["TRIBE_SIZE"]): OmniAgent(self, 999, "UC"); OmniAgent(self, 999, "D")
            # Hardcode static rep
            for a in self.agents:
                if a.agent_type == "UC": a.reputation = 1
                elif a.agent_type == "D": a.reputation = 0
        else:
            for i, r in enumerate(compiled_rules): OmniAgent(self, i, "Omni", r)

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for i in range(0, len(agents) - 1, 2):
            self.interact(agents[i], agents[i+1]) # A asks B
            self.interact(agents[i+1], agents[i]) # B asks A
            
    def interact(self, recipient, donor):
        b_coops = donor.decide(recipient)
        
        if b_coops:
            donor.payoff -= self.cost
            recipient.payoff += self.benefit
            
            # The Token Exchange Protocol
            if recipient.tokens >= 1:
                recipient.tokens -= 1
                donor.tokens += 1
                
            # Update Recipient's Private Memory (DR)
            recipient.memory.add(donor.unique_id)
        else:
            if donor.unique_id in recipient.memory: recipient.memory.remove(donor.unique_id)
            
        # Update Donor's Public Reputation (IR)
        if donor.agent_type == "Omni":
            act_val = 1 if b_coops else 0
            new_rep = donor.assessment_rule(act_val, recipient.reputation, donor.reputation)
            donor.reputation = 1 if new_rep > 0 else 0

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def run_evolution():
    num_tribes = 1 if MODEL_CONFIG["BENCHMARK_MODE"] else MODEL_CONFIG["NUM_TRIBES"]
    tribes = [toolbox.population(n=MODEL_CONFIG["TRIBE_SIZE"]) for _ in range(num_tribes)]
    history = {"max": [], "avg": [], "uc": [], "d": [], "T0": [], "T1": [], "T2": [], "fossil": {}}
    
    mode_str = "BENCHMARK (10 Omni / 10 UC / 10 D)" if MODEL_CONFIG["BENCHMARK_MODE"] else "3 HETEROGENEOUS TRIBES"
    print(f"Initializing Mode 5: {mode_str}...")
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        rules_flat = [ind for pop in tribes for ind in pop]
        total_payoffs = [0] * len(rules_flat)
        uc_fits, d_fits = [], []
        
        for _ in range(3):
            m = OmniModel(rules_flat, MODEL_CONFIG)
            for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): m.step()
            for a in m.agents:
                if a.agent_type == "Omni": total_payoffs[a.global_index] += a.payoff
                elif a.agent_type == "UC": uc_fits.append(a.payoff)
                elif a.agent_type == "D": d_fits.append(a.payoff)
                
        global_idx = 0
        all_omni_fits = []
        for t_idx, pop in enumerate(tribes):
            for ind in pop:
                fit = (total_payoffs[global_idx]/3) - ((len(ind[0])+len(ind[1])) * EVO_CONFIG["PARSIMONY_TAX"])
                ind.fitness.values = (fit,)
                all_omni_fits.append(fit)
                global_idx += 1
                
        best_overall = tools.selBest(rules_flat, 1)[0]
        history["max"].append(best_overall.fitness.values[0])
        history["avg"].append(np.mean(all_omni_fits))
        
        if MODEL_CONFIG["BENCHMARK_MODE"]:
            history["uc"].append(np.mean(uc_fits)/3)
            history["d"].append(np.mean(d_fits)/3)
        else:
            history["T0"].append(max([ind.fitness.values[0] for ind in tribes[0]]))
            history["T1"].append(max([ind.fitness.values[0] for ind in tribes[1]]))
            history["T2"].append(max([ind.fitness.values[0] for ind in tribes[2]]))
            
        if gen % 10 == 0 or gen == 1:
            # Store the raw strings for the comprehensive report
            history["fossil"][gen] = {"act_raw": str(best_overall[0]), "ass_raw": str(best_overall[1])}
            print(f"> Gen {gen:03d} | Max Fit: {history['max'][-1]:.1f}")

        if gen < EVO_CONFIG["MAX_GENS"]:
            for i in range(len(tribes)):
                offspring = list(map(toolbox.clone, toolbox.select(tribes[i], len(tribes[i]))))
                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.5: toolbox.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
                for mutant in offspring:
                    if random.random() < 0.2: toolbox.mutate(mutant); del mutant.fitness.values
                tribes[i][:] = offspring

    final_m = OmniModel([best_overall for pop in tribes for best_overall in pop], MODEL_CONFIG)
    for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): final_m.step()
    w_dist = [a.tokens for a in final_m.agents]

    best_per_tribe = [tools.selBest(pop, 1)[0] for pop in tribes] if not MODEL_CONFIG["BENCHMARK_MODE"] else [best_overall]

    return best_overall, history, w_dist, best_per_tribe

# =============================================================================
# 5. VISUALIZATION & REPORTING
# =============================================================================
import networkx as nx

def simplify(gp_str, rule_type):
    if rule_type == "act":
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 
                   'PartnerMemory': sp.Symbol('P_Mem'), 'PartnerRep': sp.Symbol('P_Rep'), 'MyTokens': sp.Symbol('MyTok'), 'PartnerHasToken': sp.Symbol('P_HasTok')}
    else:
        mapping = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 
                   'ActionTaken': sp.Symbol('Act'), 'RecipientRep': sp.Symbol('RecRep'), 'HelperRep': sp.Symbol('HlpRep')}
    try: return sp.simplify(eval(gp_str, {"__builtins__": {}}, mapping))
    except Exception as e: return f"Error: {e}"

def generate_comprehensive_report(best_ind, history, cfg, evo_cfg, best_per_tribe):
    report = []
    report.append("=" * 100)
    report.append(f" MODE 5 OMNI-AGENT FINAL REPORT ".center(100))
    report.append("=" * 100)
    
    report.append("\n--- ECOSYSTEM CONFIGURATION ---")
    report.append(f"Mode:              {'Benchmark (10 Omni, 10 UC, 10 D)' if cfg['BENCHMARK_MODE'] else 'Heterogeneous Tribes'}")
    report.append(f"Tribes:            {cfg['NUM_TRIBES']} (Size: {cfg['TRIBE_SIZE']} agents/tribe)")
    report.append(f"Rounds per Gen:    {cfg['NUM_ROUNDS']}")
    report.append(f"Initial Tokens:    {cfg['INITIAL_TOKENS']}")
    report.append(f"Benefit / Cost:    {cfg['BENEFIT_TO_COST_RATIO']} / {cfg['COST']}")
    report.append(f"Max Generations:   {evo_cfg['MAX_GENS']}")
    report.append(f"Parsimony Tax:     {evo_cfg['PARSIMONY_TAX']}")
    
    if cfg["BENCHMARK_MODE"]:
        report.append("\n--- APEX OMNI STRATEGY ---")
        report.append(f"ACTION (Raw):  {str(best_ind[0])}")
        report.append(f"ACTION (Simp): {simplify(str(best_ind[0]), 'act')}")
        report.append(f"ASSESS (Raw):  {str(best_ind[1])}")
        report.append(f"ASSESS (Simp): {simplify(str(best_ind[1]), 'ass')}")
    else:
        report.append("\n--- TRIBAL STRATEGIES ---")
        for i, t_best in enumerate(best_per_tribe):
            report.append(f"TRIBE {chr(65+i)}:")
            report.append(f"  ACT (Raw):  {str(t_best[0])}")
            report.append(f"  ACT (Simp): {simplify(str(t_best[0]), 'act')}")
            report.append(f"  ASS (Raw):  {str(t_best[1])}")
            report.append(f"  ASS (Simp): {simplify(str(t_best[1]), 'ass')}")
            
    report.append("\n--- FULL FOSSIL RECORD ---")
    sorted_gens = sorted(list(history["fossil"].keys()))
    # Iterate through ALL generations stored in the fossil record
    for g in sorted_gens:
        act_raw = history['fossil'][g]['act_raw']
        ass_raw = history['fossil'][g]['ass_raw']
        report.append(f"\nGen {g:03d}:")
        report.append(f"  ACT: {act_raw}")
        report.append(f"       -> {simplify(act_raw, 'act')}")
        report.append(f"  ASS: {ass_raw}")
        report.append(f"       -> {simplify(ass_raw, 'ass')}")
        
        report.append("\n--- ECOSYSTEM CONFIGURATION ---")
        
    report.append(f"Mode:              {'Benchmark (10 Omni, 10 UC, 10 D)' if cfg['BENCHMARK_MODE'] else 'Heterogeneous Tribes'}")
    report.append(f"Tribes:            {cfg['NUM_TRIBES']} (Size: {cfg['TRIBE_SIZE']} agents/tribe)")
    report.append(f"Rounds per Gen:    {cfg['NUM_ROUNDS']}")
    report.append(f"Initial Tokens:    {cfg['INITIAL_TOKENS']}")
    report.append(f"Benefit / Cost:    {cfg['BENEFIT_TO_COST_RATIO']} / {cfg['COST']}")
    report.append(f"Max Generations:   {evo_cfg['MAX_GENS']}")
    report.append(f"Parsimony Tax:     {evo_cfg['PARSIMONY_TAX']}")
    
    if cfg["BENCHMARK_MODE"]:
        report.append("\n--- APEX OMNI STRATEGY ---")
        report.append(f"ACTION (Raw):  {str(best_ind[0])}")
        report.append(f"ACTION (Simp): {simplify(str(best_ind[0]), 'act')}")
        report.append(f"ASSESS (Raw):  {str(best_ind[1])}")
        report.append(f"ASSESS (Simp): {simplify(str(best_ind[1]), 'ass')}")
    else:
        report.append("\n--- TRIBAL STRATEGIES ---")
        for i, t_best in enumerate(best_per_tribe):
            report.append(f"TRIBE {chr(65+i)}:")
            report.append(f"  ACT (Raw):  {str(t_best[0])}")
            report.append(f"  ACT (Simp): {simplify(str(t_best[0]), 'act')}")
            report.append(f"  ASS (Raw):  {str(t_best[1])}")
            report.append(f"  ASS (Simp): {simplify(str(t_best[1]), 'ass')}")

    return "\n".join(report)

# NetworkX hierarchical layout generator to bypass pygraphviz dependencies
def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):
    if pos is None: pos = {root: (xcenter, vert_loc)}
    else: pos[root] = (xcenter, vert_loc)
    if parsed is None: parsed = set()
    parsed.add(root)
    neighbors = list(G.neighbors(root))
    if parent is not None and parent in neighbors: neighbors.remove(parent)
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    return pos

def draw_gp_tree(expr, ax, title):
    nodes, edges, labels = gp.graph(expr)
    if len(nodes) <= 1:
        ax.text(0.5, 0.5, labels.get(0, str(expr)), ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgoldenrodyellow"))
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        return

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = hierarchy_pos(g, root=0)
    
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color='gray')
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color='lightgoldenrodyellow', node_size=800, edgecolors='black')
    nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=9, font_family='sans-serif')
    ax.set_title(title, fontweight='bold')
    ax.axis('off')

def plot_dashboard(best_ind, history, w_dist, cfg, best_per_tribe):
    # Reduced figure height now that the text block is removed
    fig = plt.figure(figsize=(22, 11))
    # 2 rows, 2 columns layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.2])
    
    # --- ROW 1: ECOLOGY & WEALTH ---
    ax_dyn = fig.add_subplot(gs[0, 0])
    b_max = cfg["NUM_ROUNDS"] * (cfg["BENEFIT_TO_COST_RATIO"] * cfg["COST"] - cfg["COST"])
    if cfg["BENCHMARK_MODE"]:
        ax_dyn.plot([(s/b_max)*100 for s in history["max"]], label='Apex Omni', color='darkorange', lw=2.5)
        ax_dyn.plot([(s/b_max)*100 for s in history["uc"]], label='UCs', color='mediumseagreen', lw=2)
        ax_dyn.plot([(s/b_max)*100 for s in history["d"]], label='Defectors', color='crimson', lw=2)
    else:
        ax_dyn.plot([(s/b_max)*100 for s in history["T0"]], label='Tribe A', lw=2)
        ax_dyn.plot([(s/b_max)*100 for s in history["T1"]], label='Tribe B', lw=2)
        ax_dyn.plot([(s/b_max)*100 for s in history["T2"]], label='Tribe C', lw=2)
        ax_dyn.plot([(s/b_max)*100 for s in history["avg"]], label='Global Avg', color='gray', ls='--')
    ax_dyn.set_title('Ecological Dynamics', fontsize=14, fontweight='bold')
    ax_dyn.set_ylabel('Efficiency (% of Theoretical Utopia)', fontsize=12)
    ax_dyn.legend(loc='lower left'); ax_dyn.grid(True, alpha=0.3)

    ax_wealth = fig.add_subplot(gs[0, 1])
    ax_wealth.hist(w_dist, bins=np.arange(0, max(w_dist)+2)-0.5, color='gold', edgecolor='black', alpha=0.8)
    ax_wealth.set_title('Final Wealth Distribution', fontsize=14, fontweight='bold')
    ax_wealth.set_xlabel('Tokens Held', fontsize=12)
    ax_wealth.set_ylabel('Number of Agents', fontsize=12)
    ax_wealth.set_xticks(range(max(w_dist)+1))
    ax_wealth.grid(axis='y', alpha=0.3)

    # --- ROW 2: GP LOGIC TREES ---
    ax_act_tree = fig.add_subplot(gs[1, 0])
    draw_gp_tree(best_ind[0], ax_act_tree, "Apex Action Rule Tree (Decision to Donate)")
    
    ax_ass_tree = fig.add_subplot(gs[1, 1])
    draw_gp_tree(best_ind[1], ax_ass_tree, "Apex Assessment Rule Tree (Public Gossip Update)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best, history, w_dist, best_per_tribe = run_evolution()
    
    # Generate and print the exhaustive single source of truth report to the terminal
    final_report = generate_comprehensive_report(best, history, MODEL_CONFIG, EVO_CONFIG, best_per_tribe)
    print(f"\n{final_report}\n")
    
    # Plot the clean, visual-only dashboard
    plot_dashboard(best, history, w_dist, MODEL_CONFIG, best_per_tribe)