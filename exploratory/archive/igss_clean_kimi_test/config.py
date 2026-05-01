"""
Configuration Module for iGSS (Institutional Governing Social Systems)

All experiment configurations in one place - simple Python dictionaries.
No validation, no types - just dictionaries you can print and modify.
"""

# =============================================================================
# BASE MODEL CONFIGURATION
# =============================================================================

# Default configuration for Mode 1 (Action Rule Evolution with all signals)
MODE1_BASE = {
    "USE_MEMORY": True,      # ARG0: Direct Reciprocity
    "USE_STANDING": True,    # ARG1: Indirect Reciprocity  
    "USE_TOKENS": True,      # ARG2: Monetary Exchange
    "REQUIRE_TOKENS_TO_COOPERATE": False,  # If True, only cooperate if partner has tokens
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "INITIAL_LIQUIDITY": 2.0,  # Starting tokens (can be < 1.0 for probabilistic)
    "NUM_IGSS": 10,
    "NUM_UC": 10,            # Unconditional Cooperators
    "NUM_D": 10,             # Defectors
    "NUM_ROUNDS": 20
}

# Mode 1 variants - isolated mechanisms
MODE1_DR_ONLY = {
    **MODE1_BASE,
    "USE_MEMORY": True,
    "USE_STANDING": False,
    "USE_TOKENS": False,
}

MODE1_IR_ONLY = {
    **MODE1_BASE,
    "USE_MEMORY": False,
    "USE_STANDING": True,
    "USE_TOKENS": False,
}

MODE1_MONEY_ONLY = {
    **MODE1_BASE,
    "USE_MEMORY": False,
    "USE_STANDING": False,
    "USE_TOKENS": True,
}

# =============================================================================
# MODE 2 CONFIGURATION (Assessment Rule Evolution)
# =============================================================================

MODE2_BASE = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 20,
    # Assessment rule inputs (Action, HelperStanding, RecipientStanding)
    "ASSESS_INPUTS": ["Action", "HelperStanding", "RecipientStanding"]
}

# =============================================================================
# MODE 3 CONFIGURATION (Co-Evolution)
# =============================================================================

MODE3_BASE = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 20,             # Higher defector pressure
    "NUM_ROUNDS": 20,
    # Action rule only sees PartnerStanding (pure IR)
    "ACTION_INPUTS": ["PartnerStanding"],
    # Assessment rule sees full context
    "ASSESS_INPUTS": ["Action", "HelperStanding", "RecipientStanding"]
}

# =============================================================================
# EVOLUTION CONFIGURATION
# =============================================================================

EVO_DEFAULT = {
    "POP_SIZE": 40,
    "MAX_GENS": 50,
    "PARSIMONY_TAX": 0.1,    # Penalty for tree complexity
    "CX_PROB": 0.5,          # Crossover probability
    "MUT_PROB": 0.2          # Mutation probability
}

# Higher pressure evolution for more difficult problems
EVO_HIGH_PRESSURE = {
    **EVO_DEFAULT,
    "POP_SIZE": 60,
    "MAX_GENS": 75,
    "PARSIMONY_TAX": 0.05    # Less parsimony pressure for complex problems
}

# Quick test configuration for debugging
EVO_QUICK = {
    **EVO_DEFAULT,
    "POP_SIZE": 20,
    "MAX_GENS": 10,
}

# =============================================================================
# RUN MODE CONFIGURATION
# =============================================================================

# "VISUAL" - Shows chart with text report
# "BATCH" - Saves data silently to JSON
RUN_MODE = "VISUAL"
