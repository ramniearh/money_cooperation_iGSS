import os
import csv
import itertools
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
from deap import gp

def simplify_rule(gp_string, variables):
    """Parses a DEAP primitive tree string into a simplified SymPy mathematical expression."""
    mapping = {
        'add': lambda x, y: x + y, 
        'sub': lambda x, y: x - y, 
        'mul': lambda x, y: x * y
    }
    # Dynamically inject the requested variables as SymPy symbols
    for var in variables:
        mapping[var] = sp.Symbol(var)
        
    try: 
        return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: 
        return f"SymPy Parsing Error: {e}"

def evaluate_truth_table(simp_rule_str, variables):
    """Evaluates the simplified rule across all binary combinations of its variables."""
    if "Error" in simp_rule_str:
        return None
    
    local_dict = {var: sp.Symbol(var) for var in variables}
    try:
        expr = sp.sympify(simp_rule_str, locals=local_dict)
    except Exception:
        return None
        
    # Generate all possible True/False combinations based on the number of variables
    combinations = list(itertools.product([True, False], repeat=len(variables)))
    results = []
    
    for combo in combinations:
        # Convert True/False to 1/0 for math evaluation
        sub_map = {sp.Symbol(var): (1 if val else 0) for var, val in zip(variables, combo)}
        try:
            val = expr.subs(sub_map)
            val = float(val)
            # Evaluate the boolean threshold (> 0 means action is taken / good standing)
            results.append(val > 0)
        except Exception:
            results.append(False)
            
    return tuple(results)

def log_results_to_csv(run_log, filename="results.csv"):
    """Appends the results of an experimental run to the CSV."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=run_log.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_log)

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Helper function to organize networkx nodes into a neat hierarchical tree layout."""
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

def plot_deap_trees(individual, save_path=None, block=False):
    """Dynamically plots either a single tree (Modes 1/2) or dual trees (Mode 3)."""
    # Detect if it's a Mode 3 individual (a list of two GP trees)
    is_dual = isinstance(individual, list) and len(individual) == 2
    
    if is_dual:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        axes = [ax1, ax2]
        titles = ["Evolved ACTION Rule", "Evolved ASSESSMENT Rule"]
        colors = ["lightblue", "lightgreen"]
        trees = individual
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        axes = [ax1]
        titles = ["GP Evolved Rule"]
        colors = ["lightgreen"]
        trees = [individual]
        
    for i, ax in enumerate(axes):
        nodes, edges, labels = gp.graph(trees[i])
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        try:
            pos = hierarchy_pos(g)
        except TypeError:
            pos = nx.spring_layout(g, seed=42)
            
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1000, node_color=colors[i])
        nx.draw_networkx_edges(g, pos, ax=ax, arrows=False)
        nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=10)
        ax.set_title(titles[i], fontsize=14)
        ax.axis("off")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if not block:
        plt.close(fig)
    else:
        plt.show(block=False)

def generate_dashboard(best_ind, history, run_log, save_only=True):
    """
    Generates the detailed terminal report, the fitness line charts, 
    and the tree visualizations.
    """
    # =========================================================================
    # DYNAMIC TERMINAL REPORTING
    # =========================================================================
    mode_name = run_log.get("Mode", "Experiment")
    
    report_text = f"\n{'='*65}\n"
    report_text += f"      {mode_name.upper()} REPORT\n"
    report_text += f"{'='*65}\n"
    report_text += f"Run ID: {run_log.get('Run_ID', 'N/A')} | Config: {run_log.get('CONFIG_GROUP', 'N/A')} | Rep: {run_log.get('REPETITION', 'N/A')} | Seed: {run_log.get('SEED', 'N/A')}\n\n"
    
    # Dual-tree (Mode 3 / Co-Evo) vs Single-tree logic
    if "Best_Act_SymPy" in run_log:
        report_text += f"--- BEST EVOLVED RULES ---\n"
        report_text += f"[ACTION RULE]\nRaw:   {run_log.get('Best_Act_Raw', 'N/A')}\nSymPy: {run_log.get('Best_Act_SymPy', 'N/A')}\nStrat: {run_log.get('Identified_Action', 'N/A')}\n\n"
        report_text += f"[ASSESSMENT RULE]\nRaw:   {run_log.get('Best_Ass_Raw', 'N/A')}\nSymPy: {run_log.get('Best_Ass_SymPy', 'N/A')}\nStrat: {run_log.get('Identified_Assessment', 'N/A')}\n\n"
    else:
        report_text += f"--- BEST EVOLVED RULE ---\n"
        report_text += f"Raw:   {run_log.get('Best_Rule_Raw', 'N/A')}\nSymPy: {run_log.get('Best_Rule_SymPy', 'N/A')}\nStrat: {run_log.get('Identified_Strategy', 'N/A')}\n\n"
    
    report_text += f"--- ABM CONFIGURATIONS ---\n"
    report_text += f"iGSS: {run_log.get('NUM_IGSS', 'N/A')} | UC: {run_log.get('NUM_UC', 'N/A')} | Defectors: {run_log.get('NUM_D', 'N/A')}\n"
    report_text += f"Benefit: {run_log.get('BENEFIT_TO_COST_RATIO', 'N/A')} | Cost: {run_log.get('COST', 'N/A')} | Rounds: {run_log.get('NUM_ROUNDS', 'N/A')}\n\n"
    
    report_text += f"--- EVO CONFIGURATIONS ---\n"
    report_text += f"Pop: {run_log.get('POP_SIZE', 'N/A')} | Gens: {run_log.get('MAX_GENS', 'N/A')} | Parsimony Tax: {run_log.get('PARSIMONY_TAX', 'N/A')}\n\n"
    
    report_text += f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
    
    # Calculate Max Theoretical Fitness dynamically
    b_ratio = run_log.get('BENEFIT_TO_COST_RATIO', 5)
    cost = run_log.get('COST', 1)
    rounds = run_log.get('NUM_ROUNDS', 100)
    theoretical_max = rounds * (b_ratio * cost - cost)
    
    report_text += f"Maximum Theoretical Fitness: {theoretical_max}\n"
    report_text += f"Control Baseline: {run_log.get('Control_Baseline', 'N/A')}\n"
    
    # Specific logic for the Mixed Ecology dual baselines
    if "Control_DR_Score" in run_log:
        report_text += f"  > Baseline DR (Tit-for-Tat): {run_log.get('Control_DR_Score', 'N/A')}\n"
        report_text += f"  > Baseline IR (Strict Disc): {run_log.get('Control_IR_Score', 'N/A')}\n"
        
    report_text += f"iGSS Max Payoff: {run_log.get('iGSS_Max_Fitness', 'N/A')}\n"
    report_text += f"iGSS Avg Payoff: {run_log.get('iGSS_Avg_Fitness', 'N/A')}\n"
    
    if "UC_Fitness" in run_log:
        report_text += f"Uncond. Cooperators Avg: {run_log.get('UC_Fitness', 'N/A')}\n"
    if "D_Fitness" in run_log:
        report_text += f"Defectors Avg: {run_log.get('D_Fitness', 'N/A')}\n\n"

    # --- DYNAMIC FOSSIL RECORD ---
    fossil_dict = history.get("fossil_record", {})
    if fossil_dict:
        report_text += f"--- FOSSIL RECORD (Sample Timeline) ---\n"
        sorted_gens = sorted(list(fossil_dict.keys()))
        
        # Extract ~5 evenly spaced milestones
        if len(sorted_gens) <= 5:
            display_gens = sorted_gens
        else:
            display_gens = [sorted_gens[int(i * (len(sorted_gens)-1) / 4)] for i in range(5)]
            display_gens = sorted(list(set(display_gens))) # Ensure unique, sorted list
            
        for g in display_gens:
            report_text += f"  > Gen {g:03d}: {fossil_dict[g]}\n"
            
    report_text += f"\n{'='*65}\n"
    print(report_text)

    # =========================================================================
    # VISUALIZATION GENERATION (Plots & Trees)
    # =========================================================================
    if not os.path.exists("figures"):
        os.makedirs("figures")
        
    run_id = run_log.get("Run_ID", "Unknown_Run")
    
    # 1. Line Chart: Evolutionary Fitness History
    fig, ax_plot = plt.subplots(figsize=(10, 6))
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    
    if "uc_scores" in history and len(history["uc_scores"]) > 0:
        ax_plot.plot(history["uc_scores"], label='Unconditional Cooperators', color='green', linewidth=1.5, linestyle='--')
    if "d_scores" in history and len(history["d_scores"]) > 0:
        ax_plot.plot(history["d_scores"], label='Defectors', color='red', linewidth=1.5, linestyle='-.')
        
    control_fit = run_log.get('Control_Baseline', 'N/A')
    if control_fit != 'N/A':
        ax_plot.axhline(y=float(control_fit), color='black', linestyle=':', label=f'Control Baseline: {control_fit}')
        
    # Dual baseline drawing for Mixed Models
    if "Control_DR_Score" in run_log:
        dr_fit = run_log.get('Control_DR_Score', 0)
        ir_fit = run_log.get('Control_IR_Score', 0)
        ax_plot.axhline(y=float(dr_fit), color='orange', alpha=0.5, linestyle='--', label=f'DR Baseline: {dr_fit}')
        ax_plot.axhline(y=float(ir_fit), color='purple', alpha=0.5, linestyle='--', label=f'IR Baseline: {ir_fit}')
        
    mode_str = run_log.get('Mode', 'Evolutionary Progress')
    ax_plot.set_title(f'{mode_str} [{run_id}]')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"figures/{run_id}_Fitness.png"
    plt.savefig(plot_path)
    
    if not save_only:
        plt.show(block=False)
    else:
        plt.close(fig)
        
    # 2. NetworkX: Evolved Genetic Trees
    tree_path = f"figures/{run_id}_Tree.png"
    plot_deap_trees(best_ind, save_path=tree_path, block=(not save_only))