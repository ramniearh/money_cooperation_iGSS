"""
Mode 1: Evolved Action Rules

Evolve cooperation strategies using Direct Reciprocity (Memory), 
Indirect Reciprocity (Standing), and/or Monetary Exchange (Tokens).

The assessment rule is hardcoded:
- Cooperating -> Good standing
- Defecting against Good recipient -> Bad standing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import MODE1_BASE, EVO_DEFAULT, RUN_MODE
from evolution_base import (
    setup_creator, create_action_pset, create_toolbox_single,
    evaluate_single_tree, run_evolution_loop
)


# =============================================================================
# CONFIGURATION (Override defaults here)
# =============================================================================

# Choose which mechanism(s) to use
# Options: MODE1_DR_ONLY, MODE1_IR_ONLY, MODE1_MONEY_ONLY, or MODE1_BASE (all)
CURRENT_MODEL_CONFIG = MODE1_BASE.copy()
# CURRENT_MODEL_CONFIG = MODE1_DR_ONLY.copy()
# CURRENT_MODEL_CONFIG = MODE1_IR_ONLY.copy()
# CURRENT_MODEL_CONFIG = MODE1_MONEY_ONLY.copy()

# Uncomment to require tokens for cooperation (conditional money)
# CURRENT_MODEL_CONFIG["REQUIRE_TOKENS_TO_COOPERATE"] = True

CURRENT_EVO_CONFIG = EVO_DEFAULT.copy()

# =============================================================================
# EVOLUTION SETUP
# =============================================================================

def setup_mode1():
    """Set up DEAP for Mode 1 (single-tree action rule evolution)."""
    setup_creator()
    
    # Determine number of inputs based on config
    num_inputs = 0
    input_names = []
    if CURRENT_MODEL_CONFIG.get("USE_MEMORY", False):
        num_inputs += 1
        input_names.append("ARG0_Memory")
    if CURRENT_MODEL_CONFIG.get("USE_STANDING", False):
        num_inputs += 1
        input_names.append("ARG1_Standing")
    if CURRENT_MODEL_CONFIG.get("USE_TOKENS", False):
        num_inputs += 1
        input_names.append("ARG2_Tokens")
    
    # Default to 3 inputs if all enabled
    if num_inputs == 0:
        num_inputs = 3
        input_names = ["ARG0_Memory", "ARG1_Standing", "ARG2_Tokens"]
    
    # Create primitive set and toolbox
    pset = create_action_pset(num_inputs=num_inputs, input_names=input_names)
    toolbox = create_toolbox_single(pset)
    
    return toolbox, pset


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_mode1(individual, toolbox):
    """Evaluate an individual for Mode 1."""
    return evaluate_single_tree(
        individual, toolbox, 
        CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG,
        action_rule=None, assessment_rule=None
    )


# =============================================================================
# MAIN EVOLUTION
# =============================================================================

def run_evolution():
    """Run the evolutionary process for Mode 1."""
    toolbox, pset = setup_mode1()
    
    # Create partial evaluation function
    from functools import partial
    eval_fn = partial(evaluate_mode1, toolbox=toolbox)
    
    print("=" * 50)
    print("MODE 1: Evolving Action Rules")
    print("=" * 50)
    print(f"Mechanisms:")
    print(f"  - Direct Reciprocity (Memory): {CURRENT_MODEL_CONFIG.get('USE_MEMORY', False)}")
    print(f"  - Indirect Reciprocity (Standing): {CURRENT_MODEL_CONFIG.get('USE_STANDING', False)}")
    print(f"  - Monetary Exchange (Tokens): {CURRENT_MODEL_CONFIG.get('USE_TOKENS', False)}")
    print(f"  - Require Tokens to Cooperate: {CURRENT_MODEL_CONFIG.get('REQUIRE_TOKENS_TO_COOPERATE', False)}")
    print(f"\nEvolution: Pop={CURRENT_EVO_CONFIG['POP_SIZE']}, Gens={CURRENT_EVO_CONFIG['MAX_GENS']}")
    print("-" * 50)
    
    best_rule, history, final_pop = run_evolution_loop(
        toolbox, CURRENT_EVO_CONFIG, eval_fn, dual_tree=False
    )
    
    return best_rule, history


# =============================================================================
# VISUALIZATION & REPORTING
# =============================================================================

def save_batch_data(best_rule, history, model_config, evo_config):
    """Save experimental data to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mode1_experiment_{timestamp}.json"
    
    try:
        clean_history = {
            "max_fitness": [float(x) for x in history["max_fitness"]],
            "avg_fitness": [float(x) for x in history["avg_fitness"]],
            "fossil_record": history["fossil_record"]
        }
        
        data = {
            "mode": 1,
            "best_strategy": str(best_rule),
            "final_max_fitness": float(history["max_fitness"][-1]),
            "final_avg_fitness": float(history["avg_fitness"][-1]),
            "model_config": model_config,
            "evo_config": evo_config,
            "history": clean_history
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"\n[!] Data saved to {filename}")
        
    except Exception as e:
        print(f"\n[X] Error saving data: {e}")


def plot_dashboard(best_rule, history, model_config, evo_config):
    """Generate visual dashboard with chart and report."""
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, figsize=(14, 6), 
        gridspec_kw={'width_ratios': [2, 1]}
    )
    
    # Calculate efficiency metrics
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    net_profit = benefit - model_config["COST"]
    theoretical_max = model_config["NUM_ROUNDS"] * net_profit
    
    max_efficiency = [(score / theoretical_max) * 100 for score in history["max_fitness"]]
    avg_efficiency = [(score / theoretical_max) * 100 for score in history["avg_fitness"]]
    
    final_max_eff = max_efficiency[-1]
    final_avg_eff = avg_efficiency[-1]
    
    # Format fossil record
    fossil_gens = sorted(history["fossil_record"].keys())
    fossils_str = "\n".join([
        f"  Gen {g:02d}: {history['fossil_record'][g]}" 
        for g in fossil_gens
    ])
    
    # Build report text
    report_text = (
        f"\n{'='*40}\n"
        f"     MODE 1 FINAL LAB REPORT\n"
        f"{'='*40}\n\n"
        f"--- PERFORMANCE ---\n"
        f"Final Max Efficiency: {final_max_eff:.1f}%\n"
        f"Final Avg Efficiency: {final_avg_eff:.1f}%\n\n"
        f"--- CONFIGURATION ---\n"
        f"Memory (DR):    {model_config.get('USE_MEMORY', False)}\n"
        f"Standing (IR):  {model_config.get('USE_STANDING', False)}\n"
        f"Tokens (Money): {model_config.get('USE_TOKENS', False)}\n"
        f"Require Tokens: {model_config.get('REQUIRE_TOKENS_TO_COOPERATE', False)}\n\n"
        f"Populations: iGSS: {model_config['NUM_IGSS']} | "
        f"UC: {model_config['NUM_UC']} | D: {model_config['NUM_D']}\n"
        f"Economics:   Cost: {model_config['COST']} | "
        f"Benefit: {model_config['BENEFIT_TO_COST_RATIO']} | "
        f"Liq: {model_config['INITIAL_LIQUIDITY']}\n"
        f"Evolution:   Pop: {evo_config['POP_SIZE']} | "
        f"Gens: {evo_config['MAX_GENS']} | "
        f"Tax: {evo_config['PARSIMONY_TAX']}\n\n"
        f"--- FOSSIL RECORD ---\n"
        f"{fossils_str}\n"
        f"{'='*40}\n\n"
        f"--- BEST STRATEGY ---\n"
        f"{str(best_rule)}\n"
    )
    
    # Print to terminal
    print(report_text)
    
    # Plot chart
    ax_plot.plot(max_efficiency, label='Max Efficiency', color='blue', linewidth=2)
    ax_plot.plot(avg_efficiency, label='Avg Efficiency', color='lightblue', linestyle='--')
    ax_plot.set_title('Mode 1: Evolutionary Learning Curve')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cooperation Efficiency (% of Theoretical Max)')
    ax_plot.set_ylim(0, 105)
    ax_plot.legend()
    ax_plot.grid(True, alpha=0.3)
    
    # Plot text
    ax_text.axis('off')
    plot_text = report_text.replace('='*40 + '\n', '').replace('     MODE 1 FINAL LAB REPORT\n', '')
    ax_text.text(0.0, 0.95, plot_text, fontsize=9, family='monospace',
                 verticalalignment='top', transform=ax_text.transAxes, wrap=True)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for Mode 1 experiments."""
    mode = RUN_MODE.strip().upper()
    print(f"Starting Mode 1 in {mode} mode...\n")
    
    best_rule, history = run_evolution()
    
    if mode == "VISUAL":
        print("\n[!] Evolution complete. Launching visual dashboard...")
        plot_dashboard(best_rule, history, CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG)
    elif mode == "BATCH":
        save_batch_data(best_rule, history, CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG)
    else:
        print(f"\n[X] Error: RUN_MODE '{RUN_MODE}' is invalid. Use 'VISUAL' or 'BATCH'.")


if __name__ == "__main__":
    main()
