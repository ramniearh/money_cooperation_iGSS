"""Every parameter of the model, and the sweeps (SPEC §6.3, §11, §13).

BASE holds one value for every parameter except MODE, MODEL and SCORING. Those
three have no base value: a sweep or the command line must give them, and a run
without them fails. No code anywhere supplies a fallback for a missing
parameter.

A sweep lists the values some parameters take. It runs every combination of
those lists, with everything else at its base value, SEEDS_PER_CELL times.
"""

import itertools

from igss import rules
from igss.scoring import SCORING_OPTIONS

BASE = {
    # --- Environment: who is in an episode ---------------------------------
    "N_IGSS": 32,              # N_I: iGSS agents per episode (must equal POP_SIZE under individual scoring)
    "N_C": 32,                 # unconditional cooperators
    "N_D": 32,                 # unconditional defectors

    # --- Game ---------------------------------------------------------------
    "B": 3,                    # benefit to the partner (5 before SPEC v1.10)
    "C": 1,                    # cost to the donor (1 by convention)
    "E_ACTION": 0.05,          # probability that an intended cooperation fails (0 before SPEC v1.10)
    "ROUNDS": 400,             # R: rounds per episode (about 4 meetings per ordered pair when N = 96)
    # The base assessment rule of each model, by the name of its predicate in rules.py.
    # Mode 1: the fixed rule. Modes 2 and 3: the rule background agents apply under
    # individual scoring, and the rule in force for strategies that never read their input.
    "DR_ASSESSMENT_RULE": "what_donor_did",
    "IR_ASSESSMENT_RULE": "did_not_refuse_a_one",

    # --- Search (DEAP) --------------------------------------------------------
    "POP_SIZE": 32,            # P: number of rules in the evolving population
    "GENERATIONS": 100,        # G
    "EPISODES_PER_SCORE": 1,   # E: episodes averaged per scoring
    "TOURNAMENT_SIZE": 3,
    "CXPB": 0.5,               # probability that a pair of parents is crossed
    "MUTPB": 0.2,              # probability that a child is mutated
    "INIT_DEPTH_MIN": 1,       # depth (DEAP height) of the initial expressions
    "INIT_DEPTH_MAX": 3,
    "MUT_DEPTH_MIN": 0,        # depth of the new piece a mutation inserts
    "MUT_DEPTH_MAX": 2,
    "DEPTH_LIMIT": 17,         # a deeper child is replaced by one of its parents
    "PARSIMONY_TAX": 0.0,      # per node, subtracted from the selection score (payoff per round)

    # --- Repetitions, seeds and logging --------------------------------------
    "SEEDS_PER_CELL": 50,      # runs per combination of a sweep
    "BASE_SEED": 1,            # repetition i of every combination has run seed BASE_SEED + i
    "BASELINE_SEEDS": 50,      # episodes per set of baseline runs
    "RULES_LOG_EVERY": 10,     # rules.csv gets generation 1, every k-th and the last
    "SIMPLIFY_MAX_SIZE": 50,   # expressions with more nodes get no SymPy form
}

# What the exploratory sweeps for IR in mode 2 have in common (SPEC §14.5).
_IR_MODE_2 = {"MODE": [2], "MODEL": ["IR"], "SCORING": ["aggregate"], "SEEDS_PER_CELL": [10]}

# What the sweeps of the broad exploratory sweep have in common (SPEC §14.7).
_BROAD = {"GENERATIONS": [200], "SEEDS_PER_CELL": [20]}
_BROAD_SEARCH = {"GENERATIONS": [200], "SEEDS_PER_CELL": [10]}
_ALL_MODELS_AND_MODES = {"MODE": [1, 2, 3], "MODEL": ["DR", "IR"]}
_BOTH_SCORING_OPTIONS = {"SCORING": ["individual", "aggregate"]}
_BACKGROUNDS_AND_ERROR = {"N_C": [0, 32], "N_D": [0, 32], "E_ACTION": [0.0, 0.05]}

SWEEPS = {
    # A first look: the base configuration in every mode and model, under
    # aggregate scoring only, a few seeds each.
    "base": {
        "MODE": [1, 2, 3],
        "MODEL": ["DR", "IR"],
        "SCORING": ["aggregate"],
        "SEEDS_PER_CELL": [10],
    },
    # The first sweep (SPEC §6.3), in mode 1: the action rule evolves.
    "first": {
        "MODE": [1],
        "MODEL": ["DR", "IR"],
        "SCORING": ["individual", "aggregate"],
        "N_C": [0, 32],
        "N_D": [0, 32],
        "B": [2, 5],
        "E_ACTION": [0.0, 0.05],
    },
    # The two below are examples of varying a game parameter and a search
    # parameter. Their values are placeholders for the authors to edit.
    "rounds": {
        "MODE": [1],
        "MODEL": ["DR", "IR"],
        "SCORING": ["individual", "aggregate"],
        "ROUNDS": [50, 200, 400, 800],
    },
    "parsimony": {
        "MODE": [1],
        "MODEL": ["DR", "IR"],
        "SCORING": ["individual", "aggregate"],
        "PARSIMONY_TAX": [0.0, 0.01, 0.05],
    },
    # The first sweep's environments in modes 2 and 3 (SPEC §14).
    "mode2": {
        "MODE": [2],
        "MODEL": ["DR", "IR"],
        "SCORING": ["individual", "aggregate"],
        "N_C": [0, 32],
        "N_D": [0, 32],
        "B": [2, 5],
        "E_ACTION": [0.0, 0.05],
    },
    "mode3": {
        "MODE": [3],
        "MODEL": ["DR", "IR"],
        "SCORING": ["individual", "aggregate"],
        "N_C": [0, 32],
        "N_D": [0, 32],
        "B": [2, 5],
        "E_ACTION": [0.0, 0.05],
    },
    # Exploratory sweeps for IR in mode 2 (SPEC §14.5): the environment as a full
    # grid, then length of play, generations and three search parameters, each
    # varied alone with everything else at its base value.
    "ir2_env": {
        **_IR_MODE_2,
        "N_C": [0, 32],
        "N_D": [0, 32],
        "B": [2, 5, 10],
    },
    # The same grid with execution error; a sweep of its own, so that the runs
    # of ir2_env are not repeated.
    "ir2_env_e005": {
        **_IR_MODE_2,
        "N_C": [0, 32],
        "N_D": [0, 32],
        "B": [2, 5, 10],
        "E_ACTION": [0.05],
    },
    "ir2_rounds": {**_IR_MODE_2, "ROUNDS": [100, 400, 1600]},
    # A shorter run is exactly the start of a longer one, so G is varied by
    # running 200 generations and reading the earlier ones.
    "ir2_generations": {**_IR_MODE_2, "GENERATIONS": [200]},
    "ir2_parsimony": {**_IR_MODE_2, "PARSIMONY_TAX": [0.0, 0.01, 0.05]},
    "ir2_depth": {**_IR_MODE_2, "DEPTH_LIMIT": [3, 5, 17]},
    "ir2_tournament": {**_IR_MODE_2, "TOURNAMENT_SIZE": [2, 3, 7]},
    # A first look at individual scoring (SPEC §14.6).
    "individual_look": {
        "MODE": [1, 2, 3],
        "MODEL": ["DR", "IR"],
        "SCORING": ["individual"],
        "N_C": [0, 32],
        "N_D": [0, 32],
        "B": [10],
        "GENERATIONS": [200],
        "SEEDS_PER_CELL": [20],
    },
    # The broad exploratory sweep (SPEC §14.7): eleven sweeps, each a full grid.
    "broad_individual": {
        **_ALL_MODELS_AND_MODES,
        "SCORING": ["individual"],
        "N_C": [0, 8, 32],
        "N_D": [0, 8, 32],
        "B": [2, 5, 10],
        "E_ACTION": [0.0, 0.05],
        **_BROAD,
    },
    "broad_aggregate": {
        **_ALL_MODELS_AND_MODES,
        "SCORING": ["aggregate"],
        **_BACKGROUNDS_AND_ERROR,
        "B": [2, 3, 5],
        **_BROAD,
    },
    # Mode 1 with each named assessment rule as the fixed rule.
    "broad_norms_ir": {
        "MODE": [1],
        "MODEL": ["IR"],
        **_BOTH_SCORING_OPTIONS,
        "IR_ASSESSMENT_RULE": ["did_not_refuse_a_one", "what_donor_did",
                               "action_matches_other_bit", "cooperated_with_a_one"],
        **_BACKGROUNDS_AND_ERROR,
        "B": [10],
        **_BROAD,
    },
    "broad_norms_dr": {
        "MODE": [1],
        "MODEL": ["DR"],
        **_BOTH_SCORING_OPTIONS,
        "DR_ASSESSMENT_RULE": ["what_donor_did", "cooperated_with_a_one"],
        **_BACKGROUNDS_AND_ERROR,
        "B": [10],
        **_BROAD,
    },
    "broad_rounds": {
        **_ALL_MODELS_AND_MODES,
        **_BOTH_SCORING_OPTIONS,
        "ROUNDS": [100, 400, 1600],
        **_BROAD,
    },
    "broad_error": {
        **_ALL_MODELS_AND_MODES,
        **_BOTH_SCORING_OPTIONS,
        "E_ACTION": [0.0, 0.05, 0.1, 0.2],
        **_BROAD,
    },
    "broad_episodes": {
        **_ALL_MODELS_AND_MODES,
        "SCORING": ["individual"],
        "EPISODES_PER_SCORE": [1, 4],
        **_BROAD,
    },
    "broad_parsimony": {**_ALL_MODELS_AND_MODES, **_BOTH_SCORING_OPTIONS,
                        "PARSIMONY_TAX": [0.0, 0.01, 0.05], **_BROAD_SEARCH},
    "broad_depth": {**_ALL_MODELS_AND_MODES, **_BOTH_SCORING_OPTIONS,
                    "DEPTH_LIMIT": [3, 5, 17], **_BROAD_SEARCH},
    "broad_tournament": {**_ALL_MODELS_AND_MODES, **_BOTH_SCORING_OPTIONS,
                         "TOURNAMENT_SIZE": [2, 3, 7], **_BROAD_SEARCH},
    "broad_long": {**_ALL_MODELS_AND_MODES, **_BOTH_SCORING_OPTIONS,
                   "GENERATIONS": [500], "SEEDS_PER_CELL": [10]},
    # A very small sweep, to check that everything runs. Its output means nothing.
    "smoke": {
        "MODE": [1, 2, 3],
        "MODEL": ["DR", "IR"],
        "SCORING": ["individual", "aggregate"],
        "N_C": [0, 4],
        "N_IGSS": [8],
        "N_D": [4],
        "POP_SIZE": [8],
        "ROUNDS": [20],
        "GENERATIONS": [5],
        "SEEDS_PER_CELL": [2],
        "BASELINE_SEEDS": [2],
        "RULES_LOG_EVERY": [2],
    },
}

# The sweeps above were defined and run while the base had b = 5 and e_action = 0
# (SPEC §14.8). They keep those values where they do not set b or e_action
# themselves, so that they still describe what was run. The values are added
# after the sweep's own factors: a factor with one value changes neither the
# order of the combinations nor their run ids.
_EARLIER_BASE = {"B": [5], "E_ACTION": [0.0]}
for _name, _factors in SWEEPS.items():
    SWEEPS[_name] = {**_factors, **{k: v for k, v in _EARLIER_BASE.items() if k not in _factors}}

# The main batch (SPEC §14.8), under aggregate scoring only. Grid A crosses the
# environment and game factors; grid B crosses the search parameters at the base
# environment and holds the long runs. Each pilot is its grid with one seed, so its
# runs are repetition 0 of the main runs.
_MAIN_GRID_A = {
    **_ALL_MODELS_AND_MODES,
    "SCORING": ["aggregate"],
    "N_C": [0, 32],
    "N_D": [0, 32],
    "B": [2, 3, 5, 10],
    "ROUNDS": [100, 400, 1600],
    "E_ACTION": [0.0, 0.05, 0.1, 0.2],
    "GENERATIONS": [200],
    "RULES_LOG_EVERY": [50],
}
_MAIN_GRID_B = {
    **_ALL_MODELS_AND_MODES,
    "SCORING": ["aggregate"],
    "TOURNAMENT_SIZE": [2, 3, 7],
    "DEPTH_LIMIT": [3, 5, 17],
    "PARSIMONY_TAX": [0.0, 0.01, 0.05],
    "GENERATIONS": [500],
    "RULES_LOG_EVERY": [10],
}
SWEEPS.update({
    "main": {**_MAIN_GRID_A, "SEEDS_PER_CELL": [100]},
    "main_pilot": {**_MAIN_GRID_A, "SEEDS_PER_CELL": [1]},
    "main_search": {**_MAIN_GRID_B, "SEEDS_PER_CELL": [20]},
    "main_search_pilot": {**_MAIN_GRID_B, "SEEDS_PER_CELL": [1]},
})

# Overrides used by `python -m igss.run quick`: one short run of the base configuration.
QUICK = {"GENERATIONS": 10, "SEEDS_PER_CELL": 1}

PARAMETERS = ("MODE", "MODEL", "SCORING") + tuple(BASE)


def validate(cfg):
    """Stop with an error if a configuration is incomplete or inconsistent."""
    missing = [name for name in PARAMETERS if name not in cfg]
    if missing:
        raise ValueError(f"configuration lacks {missing}; MODE, MODEL and SCORING have no default")
    if cfg["MODE"] not in rules.MODES:
        raise ValueError(f"MODE must be one of {rules.MODES}")
    if cfg["MODEL"] not in rules.MODELS:
        raise ValueError(f"MODEL must be one of {rules.MODELS}")
    if cfg["SCORING"] not in SCORING_OPTIONS:
        raise ValueError(f"SCORING must be one of {SCORING_OPTIONS}")
    for name in ("DR_ASSESSMENT_RULE", "IR_ASSESSMENT_RULE"):
        if cfg[name] not in rules.ASSESSMENT_RULES:
            raise ValueError(f"{name} must name a predicate in rules.ASSESSMENT_RULES")
    for name in ("POP_SIZE", "N_IGSS"):
        if cfg[name] < 2 or cfg[name] % 2:
            raise ValueError(f"{name} must be even and at least 2")
    if cfg["SCORING"] == "individual" and cfg["N_IGSS"] != cfg["POP_SIZE"]:
        raise ValueError("under individual scoring each rule has one agent: N_IGSS must equal POP_SIZE")
    if cfg["N_C"] < 0 or cfg["N_D"] < 0:
        raise ValueError("N_C and N_D cannot be negative")
    if not 0 <= cfg["E_ACTION"] <= 1:
        raise ValueError("E_ACTION is a probability")
    for name in ("ROUNDS", "GENERATIONS", "EPISODES_PER_SCORE", "TOURNAMENT_SIZE",
                 "SEEDS_PER_CELL", "BASELINE_SEEDS", "RULES_LOG_EVERY"):
        if cfg[name] < 1:
            raise ValueError(f"{name} must be at least 1")
    if not 0 <= cfg["INIT_DEPTH_MIN"] <= cfg["INIT_DEPTH_MAX"] <= cfg["DEPTH_LIMIT"]:
        raise ValueError("initial depths must satisfy 0 <= min <= max <= DEPTH_LIMIT")
    if not 0 <= cfg["MUT_DEPTH_MIN"] <= cfg["MUT_DEPTH_MAX"]:
        raise ValueError("mutation depths must satisfy 0 <= min <= max")
    return cfg


def assessment_rule(cfg):
    """The name of the assessment rule of the configuration's model."""
    return cfg[cfg["MODEL"] + "_ASSESSMENT_RULE"]


def sweep_cells(sweep_name, overrides=None):
    """One configuration per combination of a sweep's factors (no repetitions yet)."""
    factors = dict(SWEEPS[sweep_name])
    for name, value in (overrides or {}).items():
        factors[name] = [value]
    unknown = [name for name in factors if name not in PARAMETERS]
    if unknown:
        raise ValueError(f"sweep {sweep_name!r} names unknown parameters: {unknown}")
    # Each model reads only its own assessment rule. Varying one model's rule in a sweep
    # that also runs the other model would run that other model's combinations several
    # times, identically, under different run ids.
    for model in rules.MODELS:
        key = model + "_ASSESSMENT_RULE"
        if len(set(factors.get(key, []))) > 1 and set(factors.get("MODEL", [])) != {model}:
            raise ValueError(f"sweep {sweep_name!r} varies {key}, so its MODEL must be [{model!r}] only")
    cells = []
    for values in itertools.product(*factors.values()):
        cells.append(validate({**BASE, **dict(zip(factors, values))}))
    return cells


def expand_sweep(sweep_name, overrides=None):
    """One configuration per run: every cell of the sweep, SEEDS_PER_CELL times.

    Adds SWEEP, RUN_ID and RUN_SEED. Repetition i of every cell has the same
    run seed, BASE_SEED + i (SPEC §9).
    """
    runs = []
    for cell_index, cell in enumerate(sweep_cells(sweep_name, overrides)):
        for repetition in range(cell["SEEDS_PER_CELL"]):
            runs.append({
                **cell,
                "SWEEP": sweep_name,
                "RUN_ID": f"{sweep_name}-{cell_index:04d}-{repetition:03d}",
                "RUN_SEED": cell["BASE_SEED"] + repetition,
            })
    return runs
