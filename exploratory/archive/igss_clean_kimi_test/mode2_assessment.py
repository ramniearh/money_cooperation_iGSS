"""
Mode 2: Evolved Assessment Rules

Evolve social norms (assessment rules) for assigning "good" vs "bad" standing.
The action rule is hardcoded: cooperate if partner has good standing.

Assessment rule inputs:
- Action: 1 if helper cooperated, 0 if defected
- HelperStanding: Helper's prior standing (1=Good, 0=Bad)
- RecipientStanding: Recipient's prior standing (1=Good, 0=Bad)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import MODE2_BASE, EVO_DEFAULT, RUN_MODE
from evolution_base import (
    setup_creator, create_assessment_pset, create_toolbox_single,
    evaluate_single_tree, run_evolution_loop
)


# =============================================================================
# CONFIGURATION (Override defaults here)
# =============================================================================

CURRENT_MODEL_CONFIG = MODE2_BASE.copy()
CURRENT_EVO_CONFIG = EVO_DEFAULT.copy()

# =============================================================================
# EVOLUTION SETUP
# =============================================================================

def setup_mode2():
    """Set up DEAP for Mode 2 (single-tree assessment rule evolution)."""
    setup_creator()
    
    # Create primitive set for assessment rules (always 3 inputs)
    pset = create_assessment_pset()
    toolbox = create_toolbox_single(pset)
    
    return toolbox, pset


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_mode2(individual, toolbox):
    """
    Evaluate an individual for Mode 2.
    
    The action rule is hardcoded (cooperate if standing > 0),
    so we pass a dummy action_rule and evolve the assessment_rule.
    """
    # For Mode 2, we need to pass a dummy action_rule since the hardcoded
    # logic is in the agent's evaluate_partner method
    return evaluate_single_tree(
        individual, toolbox,
        CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG,
        action_rule=True, assessment_rule=None  # action_rule=True signals hardcoded
    )


# =============================================================================
# MAIN EVOLUTION
# =============================================================================

def run_evolution():
    """Run the evolutionary process for Mode 2."""
    toolbox, pset = setup_mode2()
    
    # Create partial evaluation function
    from functools import partial
    eval_fn = partial(evaluate_mode2, toolbox=toolbox)
    
    print("=" * 50)
    print("MODE 2: Evolving Assessment Rules")
    print("=" * 50)
    print(f"Action Rule: HARDCODED (Cooperate if Standing > 0)")
    print(f"Assessment Inputs: Action, HelperStanding, RecipientStanding")
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
    filename = f"mode2_experiment_{timestamp}.json"
    
    try:
        clean_history = {
            "max_fitness": [float(x) for x in history["max_fitness"]],
            "avg_fitness": [float(x) for x in history["avg_fitness"]],
            "fossil_record": history["fossil_record"]
        }
        
        data = {
            "mode": 2,
            "best_assessment_rule": str(best_rule),
            "action_rule": "HARDCODED: Cooperate if Standing > 0",
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
        1, 2, figsize=(15, 6),
        gridspec_kw={'width_ratios': [1.8, 1.2]}
    )
    
    # Calculate efficiency metrics
    benefit = model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"]
    theoretical_max = model_config["NUM_ROUNDS"] * (benefit - model_config["COST"])
    
    # Calculate achievable ceiling based on population
    total_agents = model_config["NUM_IGSS"] + model_config["NUM_UC"] + model_config["NUM_D"]
    cooperative_agents = model_config["NUM_IGSS"] + model_config["NUM_UC"]
    achievable_ceiling = (cooperative_agents / total_agents) * 100
    
    max_efficiency = [(score / theoretical_max) * 100 for score in history["max_fitness"]]
    avg_efficiency = [(score / theoretical_max) * 100 for score in history["avg_fitness"]]
    
    # Format fossil record
    fossil_gens = sorted(history["fossil_record"].keys())
    fossils_str = "\n".join([
        f"  Gen {g:02d}: {history['fossil_record'][g]}"
        for g in fossil_gens
    ])
    
    # Build report text
    report_text = (
        f"--- BEST EVOLVED INSTITUTION ---\n"
        f"ACTION RULE:\n"
        f"  HARDCODED: Cooperate if Standing > 0\n\n"
        f"ASSESSMENT RULE:\n"
        f"  {str(best_rule)}\n\n"
        f"--- ENVIRONMENT SETTINGS ---\n"
        f"iGSS: {model_config['NUM_IGSS']} | "
        f"UC: {model_config['NUM_UC']} | "
        f"Defectors: {model_config['NUM_D']}\n"
        f"Benefit: {model_config['BENEFIT_TO_COST_RATIO']} | "
        f"Cost: {model_config['COST']} | "
        f"Rounds: {model_config['NUM_ROUNDS']}\n\n"
        f"--- FOSSIL RECORD ---\n"
        f"{fossils_str}\n"
    )
    
    # Print to terminal
    print("\n" + "=" * 50)
    print("      MODE 2 FINAL LAB REPORT")
    print("=" * 50)
    print(report_text)
    print("=" * 50)
    
    # Plot chart
    ax_plot.plot(max_efficiency, label='Max Efficiency', color='teal', linewidth=2)
    ax_plot.plot(avg_efficiency, label='Avg Efficiency', color='turquoise', linestyle='--')
    ax_plot.axhline(y=achievable_ceiling, color='red', linestyle=':', linewidth=2,
                    label=f'Achievable Max ({achievable_ceiling:.1f}%)')
    
    ax_plot.set_title('Mode 2: Evolving Assessment Rules')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Efficiency (% of Total Possible Points)')
    ax_plot.set_ylim(0, 105)
    ax_plot.legend()
    ax_plot.grid(True, alpha=0.3)
    
    # Plot text
    ax_text.axis('off')
    ax_text.text(0.0, 0.95, report_text, fontsize=10, family='monospace',
                 verticalalignment='top', transform=ax_text.transAxes, wrap=True)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for Mode 2 experiments."""
    mode = RUN_MODE.strip().upper()
    print(f"Starting Mode 2 in {mode} mode...\n")
    
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
