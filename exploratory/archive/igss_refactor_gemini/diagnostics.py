import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
from deap import gp
from datetime import datetime

# =============================================================================
# 1. SYMBOLIC MATH SIMPLIFIER
# =============================================================================
def simplify_logic(rule_str):
    """Parses and mathematically simplifies GP trees."""
    vars = ['Is_In_Memory', 'Partner_Standing', 'Partner_Tokens', 'Action', 
            'Helper_Standing', 'Recipient_Standing', 'Helper_Tokens', 'Recipient_Tokens']
    local_dict = {v: sp.Symbol(v) for v in vars}
    local_dict.update({
        'add': lambda x, y: x + y, 
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y, 
        'if_then': sp.Function('if_then')
    })
    try:
        # Converts DEAP string to SymPy and prunes algebraic bloat
        return str(sp.simplify(sp.sympify(rule_str, locals=local_dict)))
    except Exception:
        return rule_str

# =============================================================================
# 2. METADATA GENERATOR
# =============================================================================
def get_mode_context(config):
    """Generates descriptive metadata based on the active strategy and mode."""
    mode = config.get("EVO_MODE", 1)
    strategy = "N/A"
    
    if config["USE_STANDING"]: strategy = "Indirect Reciprocity (Standing)"
    elif config["USE_MEMORY"]: strategy = "Direct Reciprocity (Memory)"
    elif config["USE_TOKENS"]: strategy = "Monetary Exchange (Tokens)"

    if mode == 1:
        evolve_info = "EVOLVING: Action Rule | FIXED: Assessment (Physics)"
    elif mode == 2:
        evolve_info = "EVOLVING: Assessment Rule | FIXED: Action (Benchmark)"
    else:
        evolve_info = "CO-EVOLUTION: Action & Assessment Both Evolving"

    return strategy, evolve_info

# =============================================================================
# 3. HIGH-DENSITY DASHBOARD
# =============================================================================
def plot_unified_dashboard(experiment_name, history, model_config, best_rule, raw_rule_str, simplified_rule):
    """Dense 2-row dashboard: Learning/Tree (Top) and Rich Data Report (Bottom)."""
    
    # --- Performance Data ---
    net = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"]
    t_max = model_config["NUM_ROUNDS"] * (net + model_config.get("BASELINE_FITNESS", 1))
    
    max_eff = [(s / t_max) * 100 for s in history["max_fitness"]]
    avg_eff = [(s / t_max) * 100 for s in history["avg_fitness"]]
    strategy_label, evolve_label = get_mode_context(model_config)

    # --- Fossil Record Processing ---
    fossil_str = "\n".join([f"Gen {g:02}: {r}" for g, r in history["fossil_record"].items()])

    # --- Layout Setup ---
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    # PANEL 1: Learning Curve
    ax_plot.plot(max_eff, color='#1f77b4', linewidth=2, label='Max %')
    ax_plot.plot(avg_eff, color='#aec7e8', linestyle='--', label='Avg %')
    ax_plot.axhline(100, color='red', linestyle=':', alpha=0.5, label='Theoretical Max')
    ax_plot.set_title(f"Efficiency Growth: {experiment_name}", fontweight='bold')
    ax_plot.set_xlabel("Generation")
    ax_plot.set_ylabel("Efficiency (%)")
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.2)

    # PANEL 2: Tree Visual (Action Tree/Primary Tree)
    nodes, edges, labels = gp.graph(best_rule)
    g = nx.DiGraph()
    g.add_nodes_from(nodes); g.add_edges_from(edges)
    try: pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception: pos = nx.spring_layout(g, seed=42)

    nx.draw(g, pos, ax=ax_tree, labels=labels, node_size=800, node_color="#2ca02c", 
            edge_color="gray", font_size=7, alpha=0.8, with_labels=True)
    ax_tree.set_title("Evolved Decision Tree Structure", fontweight='bold')

    # PANEL 3: Data-Dense Metadata Report
    ax_text.axis('off')
    report = (
        f"--- DISCOVERED LOGIC ---\n"
        f"RAW      : {raw_rule_str}\n"
        f"PRUNED   : {simplified_rule}\n\n"
        f"--- RUN METADATA ---\n"
        f"ID       : {experiment_name} | MODE: {model_config.get('EVO_MODE', 1)} | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"TARGET   : {strategy_label}\n"
        f"PROCESS  : {evolve_label}\n"
        f"PERF     : Max Eff {max_eff[-1]:.1f}% | Avg Eff {avg_eff[-1]:.1f}%\n"
        f"AGENTS   : iGSS({model_config['NUM_IGSS']}) UC({model_config['NUM_UC']}) D({model_config['NUM_D']})\n"
        f"ECON     : Rounds({model_config['NUM_ROUNDS']}) B/C({model_config['BENEFIT_TO_COST_RATIO']}) Liq({model_config.get('INITIAL_LIQUIDITY','N/A')})\n"
        f"--- EVOLUTIONARY SNAPSHOTS ---\n"
        f"{fossil_str}\n\n"
    )
    ax_text.text(0, 1, report, family='monospace', fontsize=9, verticalalignment='top', wrap=True)

    plt.tight_layout()
    plt.show()