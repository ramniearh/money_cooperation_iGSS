"""
Visualization and reporting for IGSS experiments.

Provides plotting functions for evolutionary progress, GP tree visualization,
and formatted reports suitable for academic publication.
"""

from typing import Optional, List, Dict, Any
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
from deap import gp

from .config import ExperimentConfig, ModelConfig, EvoConfig
from .evolution import EvolutionHistory


def simplify_rule(gp_string: str, argument_names: List[str]) -> str:
    """
    Simplify a GP rule using SymPy.
    
    Args:
        gp_string: String representation of GP tree
        argument_names: List of argument names to treat as symbols
        
    Returns:
        Simplified mathematical expression
    """
    # Build mapping for SymPy
    mapping = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
    }
    
    for name in argument_names:
        mapping[name] = sp.Symbol(name)
    
    try:
        return str(sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping)))
    except Exception as e:
        return f"SymPy Parsing Error: {e}"


def hierarchy_pos(
    G: nx.DiGraph,
    root: Optional[Any] = None,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0.0,
    xcenter: float = 0.5
) -> Dict[Any, Tuple[float, float]]:
    """
    Compute hierarchical layout for tree visualization.
    
    Pure Python implementation to avoid pygraphviz dependency.
    
    Args:
        G: NetworkX directed graph (must be a tree)
        root: Root node (auto-detected if None)
        width: Width of layout
        vert_gap: Vertical gap between levels
        vert_loc: Vertical location of root
        xcenter: Horizontal center of root
        
    Returns:
        Dictionary mapping nodes to (x, y) positions
    """
    if not nx.is_tree(G):
        raise TypeError('Graph must be a tree to use hierarchy_pos')
    
    if root is None:
        root = next(iter([n for n, d in G.in_degree() if d == 0]))
    
    def _hierarchy_pos(
        G, node, width, vert_gap, vert_loc, xcenter,
        pos=None, parent=None
    ):
        if pos is None:
            pos = {node: (xcenter, vert_loc)}
        else:
            pos[node] = (xcenter, vert_loc)
        
        children = list(G.successors(node))
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G, child, width=dx, vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap, xcenter=nextx,
                    pos=pos, parent=node
                )
        return pos
    
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def plot_tree(
    individual: Any,
    ax: Optional[plt.Axes] = None,
    title: str = "GP Tree",
    node_color: str = "lightblue",
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot a single GP tree.
    
    Args:
        individual: GP individual (PrimitiveTree or list containing one)
        ax: Matplotlib axes (creates new figure if None)
        title: Plot title
        node_color: Color for nodes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Handle both single trees and lists
    if isinstance(individual, list):
        tree = individual[0]
    else:
        tree = individual
    
    nodes, edges, labels = gp.graph(tree)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    
    try:
        pos = hierarchy_pos(g)
    except TypeError:
        pos = nx.spring_layout(g, seed=42)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1000, node_color=node_color)
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=False)
    nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=10)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    
    return fig


def plot_dual_trees(
    individual: List,
    titles: tuple = ("Evolved ACTION Rule", "Evolved ASSESSMENT Rule"),
    colors: tuple = ("lightblue", "lightgreen"),
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot two GP trees side by side.
    
    Args:
        individual: List containing two GP trees
        titles: Titles for each tree
        colors: Colors for each tree
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Tree 1
    n1, e1, l1 = gp.graph(individual[0])
    g1 = nx.DiGraph()
    g1.add_nodes_from(n1)
    g1.add_edges_from(e1)
    try:
        p1 = hierarchy_pos(g1)
    except TypeError:
        p1 = nx.spring_layout(g1)
    
    nx.draw_networkx_nodes(g1, p1, ax=ax1, node_size=1000, node_color=colors[0])
    nx.draw_networkx_edges(g1, p1, ax=ax1, arrows=False)
    nx.draw_networkx_labels(g1, p1, l1, ax=ax1, font_size=10)
    ax1.set_title(titles[0], fontsize=14)
    ax1.axis("off")
    
    # Tree 2
    n2, e2, l2 = gp.graph(individual[1])
    g2 = nx.DiGraph()
    g2.add_nodes_from(n2)
    g2.add_edges_from(e2)
    try:
        p2 = hierarchy_pos(g2)
    except TypeError:
        p2 = nx.spring_layout(g2)
    
    nx.draw_networkx_nodes(g2, p2, ax=ax2, node_size=1000, node_color=colors[1])
    nx.draw_networkx_edges(g2, p2, ax=ax2, arrows=False)
    nx.draw_networkx_labels(g2, p2, l2, ax=ax2, font_size=10)
    ax2.set_title(titles[1], fontsize=14)
    ax2.axis("off")
    
    plt.tight_layout()
    return fig


def plot_triple_trees(
    individual: List,
    titles: tuple = ("ACTION", "MEMORY", "STANDING"),
    colors: tuple = ("lightblue", "lightgreen", "lightcoral"),
    figsize: tuple = (18, 5)
) -> plt.Figure:
    """
    Plot three GP trees side by side.
    
    Args:
        individual: List containing three GP trees
        titles: Titles for each tree
        colors: Colors for each tree
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    trees = [
        (individual[0], ax1, colors[0], titles[0]),
        (individual[1], ax2, colors[1], titles[1]),
        (individual[2], ax3, colors[2], titles[2]),
    ]
    
    for tree, ax, color, title in trees:
        n, e, l = gp.graph(tree)
        g = nx.DiGraph()
        g.add_nodes_from(n)
        g.add_edges_from(e)
        try:
            p = hierarchy_pos(g)
        except TypeError:
            p = nx.spring_layout(g)
        
        nx.draw_networkx_nodes(g, p, ax=ax, node_size=800, node_color=color)
        nx.draw_networkx_edges(g, p, ax=ax, arrows=False)
        nx.draw_networkx_labels(g, p, l, ax=ax, font_size=8)
        ax.set_title(f"Evolved {title} Rule", fontsize=12)
        ax.axis("off")
    
    plt.tight_layout()
    return fig


def generate_report(
    best_individual: Any,
    history: EvolutionHistory,
    config: ExperimentConfig,
    control_fitness: float
) -> str:
    """
    Generate formatted text report.
    
    Args:
        best_individual: Best evolved individual
        history: Evolution history
        config: Experiment configuration
        control_fitness: Control baseline fitness
        
    Returns:
        Formatted report string
    """
    model_config = config.model
    evo_config = config.evolution
    
    # Calculate theoretical maximum
    theoretical_max = (
        model_config.num_rounds * 
        (model_config.benefit_to_cost_ratio * model_config.cost - model_config.cost)
    )
    
    # Format rules
    if config.num_trees == 1:
        raw = str(best_individual)
        arg_names = (
            config.action_primitive.argument_names 
            if config.action_primitive 
            else config.assessment_primitive.argument_names
        )
        simp = simplify_rule(raw, arg_names)
        rules_section = f"--- BEST EVOLVED RULE ---\nRaw:   {raw}\nSymPy: {simp}\n"
    elif config.num_trees == 2:
        raw_act = str(best_individual[0])
        raw_ass = str(best_individual[1])
        simp_act = simplify_rule(raw_act, config.action_primitive.argument_names)
        simp_ass = simplify_rule(raw_ass, config.assessment_primitive.argument_names)
        rules_section = (
            f"--- BEST EVOLVED RULES ---\n"
            f"[ACTION]\nRaw:   {raw_act}\nSymPy: {simp_act}\n\n"
            f"[ASSESSMENT]\nRaw:   {raw_ass}\nSymPy: {simp_ass}\n"
        )
    else:  # 3 trees
        simp_act = simplify_rule(str(best_individual[0]), config.action_primitive.argument_names)
        simp_mem = simplify_rule(str(best_individual[1]), config.assessment_primitive.argument_names)
        simp_stand = simplify_rule(str(best_individual[2]), config.assessment_primitive.argument_names)
        rules_section = (
            f"--- BEST EVOLVED RULES (SymPy) ---\n"
            f"[ACTION]:   {simp_act}\n"
            f"[MEMORY]:   {simp_mem}\n"
            f"[STANDING]: {simp_stand}\n"
        )
    
    # Fossil record
    gen_width = 3 if evo_config.max_gens >= 1000 else 2
    fossils_str = "\n".join([
        f"  Gen {g:0{gen_width}d}: {history.fossil_record[g]}"
        for g in sorted(history.fossil_record.keys())
    ])
    
    # Mechanism-specific info
    mechanism_info = ""
    if config.mechanism == "joint":
        mechanism_info = f"Action Logic: {model_config.action_logic}\n"
    elif config.mechanism == "money":
        mechanism_info = (
            f"Liquidity (Fiat Genesis): {model_config.initial_endowment} tokens "
            f"to {int(model_config.endowment_fraction * 100)}% of money-users\n"
        )
    
    # Build report
    separator = "=" * (70 if config.num_trees == 3 else 60)
    
    report = f"""
{separator}
      {config.name.upper().replace('_', ' ')} REPORT
{separator}
{rules_section}
--- ABM CONFIGURATIONS ---
{mechanism_info}iGSS: {model_config.num_igss} | UC: {model_config.num_uc} | Defectors: {model_config.num_d}
Benefit: {model_config.benefit_to_cost_ratio} | Cost: {model_config.cost} | Rounds: {model_config.num_rounds}

--- EVO CONFIGURATIONS ---
Pop: {evo_config.pop_size} | Gens: {evo_config.max_gens} | Parsimony Tax: {evo_config.parsimony_tax}

--- FINAL PERFORMANCE (PER AGENT) ---
Maximum Theoretical Fitness: {theoretical_max}
Control Baseline ({config.control_agent_type}): {control_fitness:.1f}
iGSS Max Payoff: {history.max_igss[-1]:.1f}
Uncond. Cooperators Avg Payoff: {history.uc_scores[-1]:.1f}
Defectors Avg Payoff: {history.d_scores[-1]:.1f}

--- FOSSIL RECORD ---
{fossils_str}
{separator}
"""
    
    return report


def plot_dashboard(
    best_individual: Any,
    history: EvolutionHistory,
    config: ExperimentConfig,
    control_fitness: float,
    show_trees: bool = True,
    block: bool = True
):
    """
    Create comprehensive visualization dashboard.
    
    Args:
        best_individual: Best evolved individual
        history: Evolution history
        config: Experiment configuration
        control_fitness: Control baseline fitness
        show_trees: Whether to display GP trees
        block: Whether to block until plots are closed
    """
    model_config = config.model
    
    # Print report
    report = generate_report(best_individual, history, config, control_fitness)
    print(report)
    
    # Calculate theoretical maximum
    theoretical_max = (
        model_config.num_rounds * 
        (model_config.benefit_to_cost_ratio * model_config.cost - model_config.cost)
    )
    
    # Plot fitness evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history.max_igss, label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax.plot(history.avg_igss, label='iGSS Agents (Avg)', color='cornflowerblue', 
            linewidth=2, linestyle=':')
    ax.plot(history.uc_scores, label='Unconditional Cooperators', 
            color='green', linewidth=1.5, linestyle='--')
    ax.plot(history.d_scores, label='Defectors', 
            color='red', linewidth=1.5, linestyle='-.')
    
    # Baselines
    ax.axhline(y=control_fitness, color='black', linestyle=':',
               label=f'Control Baseline: {control_fitness:.1f}')
    ax.axhline(y=theoretical_max, color='purple', alpha=0.3,
               label=f'Maximum Theoretical Fitness: {theoretical_max}')
    
    # Title based on experiment
    title_parts = [config.mechanism.upper(), f"Mode {config.mode[-1]}"]
    if config.mechanism == "joint" and config.mode == "mode2":
        title_parts.append(f"(Logic: {model_config.action_logic})")
    
    ax.set_title(f"{' '.join(title_parts)}: Evolutionary Performance")
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cumulative Payoff (Per Agent)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Plot trees
    if show_trees:
        if config.num_trees == 1:
            tree_title = (
                "GP Rule Abstract Syntax Tree" 
                if config.mode == "mode1" 
                else "GP Assessment Rule"
            )
            plot_tree(best_individual, title=tree_title)
        elif config.num_trees == 2:
            plot_dual_trees(best_individual)
        else:  # 3 trees
            plot_triple_trees(best_individual)
        
        if block:
            plt.show()
