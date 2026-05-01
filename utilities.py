import os
import csv
import itertools
from datetime import datetime
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
from deap import gp

# =============================================================================
# 1. MATHEMATICAL PARSING & TRUTH TABLES
# =============================================================================
def simplify_rule(gp_string, variable_names):
    """
    Dynamically parses a DEAP string into a simplified SymPy expression.
    """
    mapping = {
        'add': lambda x, y: x + y, 
        'sub': lambda x, y: x - y, 
        'mul': lambda x, y: x * y
    }
    # Dynamically inject the mode-specific variables into the SymPy environment
    for var in variable_names:
        mapping[var] = sp.Symbol(var)
        
    try: 
        return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: 
        return f"SymPy Parsing Error: {e}"

def evaluate_truth_table(sympy_expr_str, variable_names):
    """
    Evaluates a SymPy string across all binary states to generate a truth tuple.
    Returns format (True, False, False, True) for strategy identification.
    """
    try:
        expr = sp.sympify(sympy_expr_str)
        symbols = [sp.Symbol(name) for name in variable_names]
        eval_func = sp.lambdify(symbols, expr, "numpy")
        
        # Generates binary combinations (e.g., (1,1), (1,0), (0,1), (0,0))
        states = list(itertools.product([1, 0], repeat=len(variable_names)))
        
        truth_list = []
        for state in states:
            # Handle constants safely in lambdify
            result = eval_func(*state) if hasattr(eval_func, '__code__') else expr.evalf(subs=dict(zip(symbols, state)))
            truth_list.append(float(result) > 0)
            
        return tuple(truth_list)
        
    except Exception as e:
        return f"Parse Error: {e}"

# =============================================================================
# 2. DYNAMIC CSV LOGGER
# =============================================================================
def log_results_to_csv(run_data, filename="results_log.csv"):
    """
    Appends run data to a CSV. Dynamically generates headers based on the 
    dictionary keys passed to it. Forward-compatible with any config additions.
    """
    file_exists = os.path.isfile(filename)
    headers = list(run_data.keys())
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(run_data)
        run_id = run_data.get('Run_ID', 'UNKNOWN')
        print(f"\n[Logger] Run ID '{run_id}' successfully appended to {filename}.")

# =============================================================================
# 3. VISUALIZATION & BATCH DASHBOARD
# =============================================================================
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Pure Python hierarchical layout for networkx (bypasses pygraphviz)."""
    if not nx.is_tree(G): raise TypeError('Graph must be a tree to use hierarchy_pos')
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
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=node)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def _draw_single_tree(individual, ax, title, color="lightblue"):
    """Helper to cleanly draw a single DEAP abstract syntax tree."""
    nodes, edges, labels = gp.graph(individual)
    g = nx.DiGraph() 
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    
    try: pos = hierarchy_pos(g)
    except TypeError: pos = nx.spring_layout(g, seed=42) 
        
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1000, node_color=color)
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=False)
    nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=10)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

def generate_dashboard(best_ind, history, run_log, save_only=False):
    """
    Generates terminal reports and renders/saves the matplotlib charts.
    """
    mode = run_log.get("Mode", "Experiment")
    run_id = run_log.get("Run_ID", "output")
    
    # Calculate economics dynamically from the run_log
    cost = run_log.get("COST", 1)
    benefit = run_log.get("BENEFIT_TO_COST_RATIO", 5) * cost
    theoretical_max = run_log.get("NUM_ROUNDS", 100) * (benefit - cost)
    
    # 1. Terminal Report
    print(f"\n{'='*60}")
    print(f"      {mode.upper()} REPORT")
    print(f"{'='*60}")
    print(f"--- BEST EVOLVED RULE ---")
    print(f"Run ID: {run_id} | Config: {run_log.get('CONFIG_GROUP', 'N/A')} | Repetition: {run_log.get('REPETITION', 'N/A')} | Seed: {run_log.get('SEED', 'N/A')}\n")
    print(f"Raw:   {run_log.get('Best_Rule_Raw', 'N/A')}")
    print(f"SymPy: {run_log.get('Best_Rule_SymPy', 'N/A')}")
    print(f"Strat: {run_log.get('Identified_Strategy', 'N/A')}\n")
    print(f"--- FINAL PERFORMANCE ---")
    print(f"Theoretical Max: {theoretical_max}")
    print(f"Control Baseline: {run_log.get('Control_Baseline', 0)}")
    print(f"iGSS Max Payoff: {run_log.get('iGSS_Max_Fitness', 'N/A')}")
    print(f"iGSS Avg Payoff: {run_log.get('iGSS_Avg_Fitness', 'N/A')}")
    print(f"{'='*60}\n")

    # Setup directories if saving locally during a batch run
    if save_only:
        os.makedirs("figures", exist_ok=True)

    # 2. Main Fitness Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    ax.plot(history["uc_scores"], label='Unconditional Cooperators', color='green', linewidth=1.5, linestyle='--')
    ax.plot(history["d_scores"], label='Defectors', color='red', linewidth=1.5, linestyle='-.')
    
    ax.axhline(y=run_log.get("Control_Baseline", 0), color='black', linestyle=':', label='Control Baseline')
    ax.axhline(y=theoretical_max, color='purple', alpha=0.3, label='Maximum Theoretical Fitness')
    
    ax.set_title(f'{mode}: Evolutionary Dynamics')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cumulative Payoff (Per Agent)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_only:
        plt.savefig(f"figures/{run_id}_fitness.png", dpi=300)
        plt.close(fig) # Clear memory for next batch
    else:
        plt.show(block=False)
    
    # 3. AST Tree Plotting (Auto-detects single vs co-evolution)
    if isinstance(best_ind, gp.PrimitiveTree):
        # Single-tree format (Mode 1 & 2)
        fig_tree, ax_tree = plt.subplots(figsize=(8, 6))
        color = "lightgreen" if "Assessment" in mode or "Mode 2" in mode else "lightblue"
        _draw_single_tree(best_ind, ax_tree, "GP Rule Abstract Syntax Tree", color)
    else:
        # Dual-tree format (Mode 3 Co-evolution)
        fig_tree, axes = plt.subplots(1, 2, figsize=(14, 6))
        _draw_single_tree(best_ind[0], axes[0], "Evolved ACTION Rule", "lightblue")
        _draw_single_tree(best_ind[1], axes[1], "Evolved ASSESSMENT Rule", "lightgreen")
        
    plt.tight_layout()
    
    if save_only:
        plt.savefig(f"figures/{run_id}_tree.png", dpi=300)
        plt.close(fig_tree)
    else:
        plt.show() # Blocks execution until windows are manually closed