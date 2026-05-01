"""
Mode 3: Co-Evolution of Action and Assessment Rules

Simultaneously evolve both:
- Action Rule: How to behave (cooperate or defect)
- Assessment Rule: How to judge others' actions

Uses dual-tree GP where each individual contains two genetic programs.

Action rule input:
- PartnerStanding: Partner's reputation (1=Good, 0=Bad)

Assessment rule inputs:
- Action: 1 if helper cooperated, 0 if defected
- HelperStanding: Helper's prior standing
- RecipientStanding: Recipient's prior standing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import MODE3_BASE, EVO_DEFAULT, RUN_MODE
from evolution_base import (
    setup_creator_dual, create_action_pset, create_assessment_pset,
    create_toolbox_dual, evaluate_dual_tree, run_evolution_loop
)


# =============================================================================
# CONFIGURATION (Override defaults here)
# =============================================================================

CURRENT_MODEL_CONFIG = MODE3_BASE.copy()
CURRENT_EVO_CONFIG = EVO_DEFAULT.copy()

# =============================================================================
# EVOLUTION SETUP
# =============================================================================

def setup_mode3():
    """Set up DEAP for Mode 3 (dual-tree co-evolution)."""
    setup_creator_dual()
    
    # Create primitive sets
    # Action rule: 1 input (PartnerStanding)
    pset_action = create_action_pset(
        num_inputs=1, 
        input_names=["PartnerStanding"]
    )
    
    # Assessment rule: 3 inputs (Action, HelperStanding, RecipientStanding)
    pset_assess = create_assessment_pset()
    
    # Create toolbox with dual-tree operators
    toolbox = create_toolbox_dual(pset_action, pset_assess)
    
    return toolbox, pset_action, pset_assess


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_mode3(individual, toolbox):
    """Evaluate a dual-tree individual for Mode 3."""
    return evaluate_dual_tree(
        individual, toolbox,
        CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG
    )


# =============================================================================
# MAIN EVOLUTION
# =============================================================================

def run_evolution():
    """Run the co-evolutionary process for Mode 3."""
    toolbox, pset_action, pset_assess = setup_mode3()
    
    # Create partial evaluation function
    from functools import partial
    eval_fn = partial(evaluate_mode3, toolbox=toolbox)
    
    print("=" * 50)
    print("MODE 3: Co-Evolution of Action & Assessment Rules")
    print("=" * 50)
    print(f"Action Rule Input: PartnerStanding")
    print(f"Assessment Inputs: Action, HelperStanding, RecipientStanding")
    print(f"\nEvolution: Pop={CURRENT_EVO_CONFIG['POP_SIZE']}, Gens={CURRENT_EVO_CONFIG['MAX_GENS']}")
    print(f"Defector Pressure: {CURRENT_MODEL_CONFIG['NUM_D']} defectors")
    print("-" * 50)
    
    best_rules, history, final_pop = run_evolution_loop(
        toolbox, CURRENT_EVO_CONFIG, eval_fn, dual_tree=True
    )
    
    return best_rules, history


# =============================================================================
# VISUALIZATION & REPORTING
# =============================================================================

def save_batch_data(best_rules, history, model_config, evo_config):
    """Save experimental data to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mode3_experiment_{timestamp}.json"
    
    try:
        clean_history = {
            "max_fitness": [float(x) for x in history["max_fitness"]],
            "avg_fitness": [float(x) for x in history["avg_fitness"]],
            "fossil_record": history["fossil_record"]
        }
        
        data = {
            "mode": 3,
            "best_action_rule": str(best_rules[0]),
            "best_assessment_rule": str(best_rules[1]),
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


def plot_dashboard(best_rules, history, model_config, evo_config):
    """Generate visual dashboard with chart and report."""
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, figsize=(16, 6),
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
        f"  {str(best_rules[0])}\n\n"
        f"ASSESSMENT RULE:\n"
        f"  {str(best_rules[1])}\n\n"
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
    print("      MODE 3 FINAL LAB REPORT")
    print("=" * 50)
    print(f"\n--- CO-EVOLVED INSTITUTION ---")
    print(f"ACTION RULE (How to play):")
    print(f"  {str(best_rules[0])}")
    print("-" * 30)
    print(f"ASSESSMENT RULE (How to judge):")
    print(f"  {str(best_rules[1])}")
    print("\n[Action Legend: ARG0=PartnerStanding]")
    print("[Assess Legend: ARG0=Action | ARG1=HelperStanding | ARG2=RecipientStanding]")
    print(f"\n--- PERFORMANCE ---")
    print(f"Final Max Efficiency: {max_efficiency[-1]:.1f}%")
    print(f"Final Avg Efficiency: {avg_efficiency[-1]:.1f}%")
    print(f"\n--- FOSSIL RECORD ---")
    for gen in sorted(history["fossil_record"].keys()):
        fossil = history["fossil_record"][gen]
        print(f"  Gen {gen:02d}: {fossil}")
    print("=" * 50 + "\n")
    
    # Plot chart
    ax_plot.plot(max_efficiency, label='Max Efficiency', color='darkmagenta', linewidth=2)
    ax_plot.plot(avg_efficiency, label='Avg Efficiency', color='orchid', linestyle='--')
    ax_plot.axhline(y=achievable_ceiling, color='red', linestyle=':', linewidth=2,
                    label=f'Achievable Max ({achievable_ceiling:.1f}%)')
    
    ax_plot.set_title('Mode 3: Co-Evolutionary Institutional Discovery')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Efficiency (% of Total Possible Points)')
    ax_plot.set_ylim(0, 105)
    ax_plot.legend()
    ax_plot.grid(True, alpha=0.3)
    
    # Plot text
    ax_text.axis('off')
    ax_text.text(0.0, 0.95, report_text, fontsize=9, family='monospace',
                 verticalalignment='top', transform=ax_text.transAxes, wrap=True)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for Mode 3 experiments."""
    mode = RUN_MODE.strip().upper()
    print(f"Starting Mode 3 in {mode} mode...\n")
    
    best_rules, history = run_evolution()
    
    if mode == "VISUAL":
        print("\n[!] Evolution complete. Launching visual dashboard...")
        plot_dashboard(best_rules, history, CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG)
    elif mode == "BATCH":
        save_batch_data(best_rules, history, CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG)
    else:
        print(f"\n[X] Error: RUN_MODE '{RUN_MODE}' is invalid. Use 'VISUAL' or 'BATCH'.")


if __name__ == "__main__":
    main()
