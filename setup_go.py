import json
import numpy
from datetime import datetime
import matplotlib.pyplot as plt
from model import MODEL_CONFIG
from evolution import EVO_CONFIG, run_evolution

# =============================================================================
# EXPERIMENT PARAMETERS 
# Override default simulation configurations manually here prior to execution.
# =============================================================================

# MODE SWITCH: "VISUAL" (Displays chart with text report) or "BATCH" (Saves data silently)
RUN_MODE = "VISUAL" 

CURRENT_MODEL_CONFIG = MODEL_CONFIG.copy()
CURRENT_MODEL_CONFIG.update({
    "USE_MEMORY": False,   # ARG0: Direct Reciprocity
    "USE_STANDING": False, # ARG1: Indirect Reciprocity
    "USE_TOKENS": True,    # ARG2: Tokens (Money)
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 20
})

CURRENT_EVO_CONFIG = EVO_CONFIG.copy()
CURRENT_EVO_CONFIG.update({
    "POP_SIZE": 20,
    "MAX_GENS": 50,
    "PARSIMONY_TAX": 0.1
})
# =============================================================================

def save_batch_data(best_rule, history, final_pop, model_config, evo_config):
    """Saves all experimental data to a JSON file for large-scale analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_data_{timestamp}.json"
    
    data = {
        "best_strategy_discovered": str(best_rule),
        "final_max_fitness": history["max_fitness"][-1],
        "final_avg_fitness": history["avg_fitness"][-1],
        "model_configuration": model_config,
        "evolution_configuration": evo_config,
        "history": history
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\n[!] Batch run complete. Data saved to {filename}")

def plot_visual_dashboard(best_rule, history, model_config, evo_config):
    """Generates a combined visual chart and text report using Matplotlib subplots."""
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # --- Left Subplot: The Learning Curve ---
    ax_plot.plot(history["max_fitness"], label='Max Fitness (Best Strategy)', color='blue', linewidth=2)
    ax_plot.plot(history["avg_fitness"], label='Avg Fitness (Population)', color='lightblue', linestyle='--')
    ax_plot.set_title('Evolutionary Learning Curve: iGSS Agents')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Fitness Score (Average Payoff)')
    ax_plot.legend()
    ax_plot.grid(True, alpha=0.3)

    # --- Right Subplot: The Lab Report ---
    ax_text.axis('off') # Hide axes for text panel
    
    # Grab all fossil records to sample evolution
    fossil_gens = sorted(history["fossil_record"].keys())
    fossils_str = "\n".join([f"  Gen {g:02d}: {history['fossil_record'][g]}" for g in fossil_gens])
    
    report_text = (
        f"--- BEST STRATEGY DISCOVERED ---\n"
        f"{str(best_rule)}\n\n"
        
        f"--- CONFIGURATIONS ---\n"
        f"Memory (DR)    - Active: {model_config['USE_MEMORY']} (Arg0)\n"
        f"Standing (IR)  - Active: {model_config['USE_STANDING']} (Arg1)\n"
        f"Tokens (Money) - Active: {model_config['USE_TOKENS']} (Arg2)\n\n"
        f"Populations: iGSS: {model_config['NUM_IGSS']} | Uncond. Coop: {model_config['NUM_UC']} | Defectors: {model_config['NUM_D']}\n"
        f"Economics:   Cost: {model_config['COST']} | Benefit: {model_config['BENEFIT_TO_COST_RATIO']} | Liq: {model_config['INITIAL_LIQUIDITY']}\n"
        f"Evolution:   Pop: {evo_config['POP_SIZE']} | Gens: {evo_config['MAX_GENS']} | Tax: {evo_config['PARSIMONY_TAX']}\n\n"
        
        f"--- FOSSIL RECORD ---\n"
        f"{fossils_str}"
    )
    
    # Place text in the right subplot
    ax_text.text(0.0, 0.95, report_text, fontsize=10, family='monospace', 
                 verticalalignment='top', transform=ax_text.transAxes, wrap=True)
    
    plt.tight_layout()
    plt.show()

def main():
    print(f"Starting experiment in {RUN_MODE} mode...")
    
    best_rule, history, final_pop = run_evolution(
        model_config=CURRENT_MODEL_CONFIG, 
        evo_config=CURRENT_EVO_CONFIG
    )
    
    if RUN_MODE == "VISUAL":
        print("\n[!] Evolution complete. Launching visual dashboard...")
        plot_visual_dashboard(best_rule, history, CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG)
    elif RUN_MODE == "BATCH":
        save_batch_data(best_rule, history, final_pop, CURRENT_MODEL_CONFIG, CURRENT_EVO_CONFIG)
    else:
        print("Error: RUN_MODE must be 'VISUAL' or 'BATCH'.")

if __name__ == "__main__":
    main()