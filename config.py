import itertools

# =============================================================================
# 1. THE ABSOLUTE BASELINE (Fallbacks)
# =============================================================================
# These values are used UNLESS they are specifically overwritten by the Grid below.
# This serves as the "Single Source of Truth" for all models.
DEFAULTS = {
    # ABM Base Economics
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 100,
    
    # Central Bank / Scarcity (Used only in Money mode)
    "INITIAL_ENDOWMENT": 1,
    "ENDOWMENT_FRACTION": 0.5,
    
    # Future-Proofing: Noise & Trembling Hand
    "EXECUTION_ERROR_RATE": 0.0,
    "PERCEPTION_ERROR_RATE": 0.0,
    
    # DEAP Evolutionary Base
    "POP_SIZE": 40,        
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1,
    "CXPB": 0.5,
    "MUTPB": 0.2,
    "TOURNAMENT_SIZE": 3,
    "HOF_SIZE": 1,
    "INIT_METHOD": "genHalfAndHalf",
    "TREE_MIN_DEPTH": 1,
    "TREE_MAX_DEPTH": 3,
    "STATIC_DEPTH_LIMIT": 17
}

# =============================================================================
# 2. THE EXPERIMENT GRID (Parameters to Sweep)
# =============================================================================
# Add lists of values you want to test. 
# Total runs per script = the product of the lengths of these lists.
# E.g., testing 2 ratios * 2 population sizes = 4 total runs.

EXPERIMENT_GRID = {
    "BENEFIT_TO_COST_RATIO": [2, 5],            # Test low vs high temptation
    #"POP_SIZE": [40, 100],                      # Test small vs large search spaces
     "PARSIMONY_TAX": [0.1, 0.5],             # Uncomment to sweep penalty weights
    # "EXECUTION_ERROR_RATE": [0.0, 0.05]       # Uncomment to test noise resilience!
}

# =============================================================================
# 3. THE GENERATOR FUNCTION
# =============================================================================
# How many statistical repetitions do you want for EACH parameter combination?
RUNS_PER_CONFIG = 2  
BASE_SEED = 42

def get_experiment_batches():
    """
    Generates a flat list of configuration dictionaries for the entire experiment.
    If there are 4 combinations and 5 runs per config, this returns 20 dictionaries,
    each with a pre-calculated, deterministic SEED.
    """
    keys = EXPERIMENT_GRID.keys()
    values = EXPERIMENT_GRID.values()
    
    batches = []
    
    # 1. Loop through every unique parameter combination
    for combo_idx, combination in enumerate(itertools.product(*values)):
        
        # 2. For each combination, generate N statistical repetitions
        for run_idx in range(RUNS_PER_CONFIG):
            cfg = DEFAULTS.copy()
            
            # Apply the specific grid parameters (Cost, Pop_Size, etc.)
            overrides = dict(zip(keys, combination))
            cfg.update(overrides)
            
            # Embed the tracking data and seed directly into the config dictionary!
            cfg["SEED"] = BASE_SEED + (combo_idx * 1000) + run_idx
            cfg["CONFIG_GROUP"] = f"Config_{combo_idx+1}"
            cfg["REPETITION"] = run_idx + 1
            
            batches.append(cfg)
            
    # Fallback if grid is empty
    if not batches:
        cfg = DEFAULTS.copy()
        cfg["SEED"] = BASE_SEED
        cfg["CONFIG_GROUP"] = "Default"
        cfg["REPETITION"] = 1
        batches.append(cfg)
        
    return batches

# Quick Test fallback embeds the base seed
QUICK_TEST_CONFIG = DEFAULTS.copy()
QUICK_TEST_CONFIG["SEED"] = BASE_SEED