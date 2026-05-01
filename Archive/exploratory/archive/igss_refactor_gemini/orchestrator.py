import json
from datetime import datetime
from core import DEFAULT_CONFIG
from evolution_engine import EVO_CONFIG, run_universal_engine
from diagnostics import simplify_logic, plot_unified_dashboard

EXPERIMENTS = {
    # --- MODE 1: ACTION RULE ONLY ---
    "MODE1_IR": {"USE_MEMORY": False, "USE_STANDING": True, "USE_TOKENS": False, "EVO_MODE": 1},
    "MODE1_DR": {"USE_MEMORY": True, "USE_STANDING": False, "USE_TOKENS": False, "EVO_MODE": 1},
    "MODE1_MONEY": {"USE_MEMORY": False, "USE_STANDING": False, "USE_TOKENS": True, "EVO_MODE": 1, "INITIAL_LIQUIDITY": 1.0},
    
    # --- MODE 2: ASSESSMENT RULE ONLY ---
    "MODE2_IR": {"USE_MEMORY": False, "USE_STANDING": True, "USE_TOKENS": False, "EVO_MODE": 2},
    "MODE2_DR": {"USE_MEMORY": True, "USE_STANDING": False, "USE_TOKENS": False, "EVO_MODE": 2},
    "MODE2_MONEY": {"USE_MEMORY": False, "USE_STANDING": False, "USE_TOKENS": True, "EVO_MODE": 2, "INITIAL_LIQUIDITY": 1.0},
    
    # --- MODE 3: CO-EVOLUTION ---
    "MODE3_IR": {"USE_MEMORY": False, "USE_STANDING": True, "USE_TOKENS": False, "EVO_MODE": 3},
    "MODE3_DR": {"USE_MEMORY": True, "USE_STANDING": False, "USE_TOKENS": False, "EVO_MODE": 3},
    "MODE3_MONEY": {"USE_MEMORY": False, "USE_STANDING": False, "USE_TOKENS": True, "EVO_MODE": 3, "INITIAL_LIQUIDITY": 1.0},
}

# -------------------------------------------------------------
# MASTER SWITCH: Pick one of the 9 experiments above!
ACTIVE_EXPERIMENT = "MODE1_MONEY" 
RUN_MODE = "VISUAL" # "VISUAL" or "BATCH"
# -------------------------------------------------------------

CUSTOM_EVO_CONFIG = EVO_CONFIG.copy()
CUSTOM_EVO_CONFIG.update({"POP_SIZE": 40, "MAX_GENS": 50})

def main():
    if ACTIVE_EXPERIMENT not in EXPERIMENTS:
        print(f"[X] Error: '{ACTIVE_EXPERIMENT}' not found.")
        return

    print(f"Loading Profile: {ACTIVE_EXPERIMENT}...")
    current_model_config = DEFAULT_CONFIG.copy()
    current_model_config.update(EXPERIMENTS[ACTIVE_EXPERIMENT])
    mode = current_model_config["EVO_MODE"]
    
    best_rule, history, final_pop = run_universal_engine(
        model_config=current_model_config, 
        evo_config=CUSTOM_EVO_CONFIG
    )
    
    if RUN_MODE.strip().upper() == "VISUAL":
        # Handle Output Formatting based on Mode
        if mode in [1, 2]:
            raw_str = str(best_rule[0])
            pruned_str = simplify_logic(raw_str)
            graph_tree = best_rule[0]
        else:
            raw_str = f"ACT: {str(best_rule[0])}\nASSESS: {str(best_rule[1])}"
            pruned_str = f"ACT: {simplify_logic(str(best_rule[0]))}\nASSESS: {simplify_logic(str(best_rule[1]))}"
            graph_tree = best_rule[0] # Note: Dashboard only graphs the Action tree for visual clarity
            
        print("\n" + "="*50)
        print(f"   FINAL LAB REPORT: {ACTIVE_EXPERIMENT}")
        print("="*50)
        print(f"Raw Tree(s):\n{raw_str}")
        print(f"\nPruned Logic (SymPy):\n{pruned_str}")
        print("="*50 + "\n")
        
        plot_unified_dashboard(ACTIVE_EXPERIMENT, history, current_model_config, graph_tree, raw_str, pruned_str)

if __name__ == "__main__":
    main()