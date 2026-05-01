# iGSS Clean Architecture

Refactored iGSS (Institutional Governing Social Systems) codebase with simplified, modular structure.

## Project Structure

```
igss_clean/
├── config.py              # All configuration dictionaries in one place
├── mechanisms.py          # Simple functions for DR/IR/Money logic
├── core_abm.py            # Shared Agent and Model classes (Mesa)
├── evolution_base.py      # Isolated DEAP setup - DON'T TOUCH
├── mode1_action.py        # Mode 1: Evolved action rules
├── mode2_assessment.py    # Mode 2: Evolved assessment rules
├── mode3_coev.py          # Mode 3: Co-evolution
└── tools/                 # Side-car diagnostics (separate from core)
    └── tree_drawer.py     # Graphviz tree visualization
```

## Quick Start

### Mode 1: Evolved Action Rules

```bash
python mode1_action.py
```

Edit `config.py` to choose which mechanisms to use:
- `MODE1_DR_ONLY` - Direct Reciprocity only (Memory)
- `MODE1_IR_ONLY` - Indirect Reciprocity only (Standing)
- `MODE1_MONEY_ONLY` - Monetary Exchange only (Tokens)
- `MODE1_BASE` - All three mechanisms combined

### Mode 2: Evolved Assessment Rules

```bash
python mode2_assessment.py
```

Action rule is hardcoded: "Cooperate if partner has good standing"
Assessment rule evolves: "How to judge actions and update standing"

### Mode 3: Co-Evolution

```bash
python mode3_coev.py
```

Simultaneously evolves both action and assessment rules using dual-tree GP.

## Configuration

Edit `config.py` to customize:

```python
# Model parameters
MODE1_BASE = {
    "USE_MEMORY": True,           # Enable Direct Reciprocity
    "USE_STANDING": True,         # Enable Indirect Reciprocity
    "USE_TOKENS": True,           # Enable Monetary Exchange
    "REQUIRE_TOKENS_TO_COOPERATE": False,  # Only cooperate if partner can pay
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "INITIAL_LIQUIDITY": 2.0,
    "NUM_IGSS": 10,               # Evolvable agents
    "NUM_UC": 10,                 # Unconditional Cooperators
    "NUM_D": 10,                  # Defectors
    "NUM_ROUNDS": 20
}

# Evolution parameters
EVO_DEFAULT = {
    "POP_SIZE": 40,
    "MAX_GENS": 50,
    "PARSIMONY_TAX": 0.1,         # Penalty for complex trees
    "CX_PROB": 0.5,               # Crossover probability
    "MUT_PROB": 0.2               # Mutation probability
}

# Run mode
RUN_MODE = "VISUAL"  # "VISUAL" for charts, "BATCH" for JSON output
```

## Key Design Decisions

### 1. DEAP Isolation

All DEAP code is in `evolution_base.py`. This file uses global state (creator, toolbox) which is messy but functional. **Don't refactor it unless absolutely necessary.**

### 2. Mechanisms as Functions

Instead of abstract classes, mechanisms are simple functions in `mechanisms.py`:

```python
def update_memory(helper, recipient, cooperates):
    """Direct Reciprocity logic"""
    if not cooperates:
        recipient.memory.add(helper.unique_id)
        if recipient.unique_id in helper.memory:
            helper.memory.remove(recipient.unique_id)  # Forgiveness

def update_standing_hardcoded(helper, recipient, cooperates):
    """Indirect Reciprocity logic (Mode 1)"""
    if cooperates:
        helper.standing = 1
    elif recipient.standing == 1:
        helper.standing = 0
```

### 3. Shared Core ABM

`core_abm.py` contains the `CoopAgent` and `CooperationModel` classes used by all modes. The model is configured via:
- `action_rule` - For Modes 1 & 3 (None = hardcoded for Mode 2)
- `assessment_rule` - For Modes 2 & 3 (None = hardcoded for Mode 1)

### 4. Diagnostics as Side-Project

The `tools/` folder contains standalone scripts that read output files. They never import core code.

## Visualizing Trees

After running an experiment:

```bash
# From experiment JSON
python tools/tree_drawer.py --input mode1_experiment_20240101_120000.json

# From direct string
python tools/tree_drawer.py --string "add(ARG0, mul(ARG1, 2))"
```

## Extending to DR+Money Co-Evolution

To add Mode 4 (DR+Money co-evolution):

1. Copy `mode3_coev.py` → `mode4_dr_money.py`
2. Modify the action primitive set to take 2 inputs: `["PartnerMemory", "PartnerTokens"]`
3. Keep assessment rule for IR, or add separate DR assessment
4. Update `evaluate_partner()` in `core_abm.py` if needed

## Dependencies

```bash
pip install mesa deap numpy matplotlib

# Optional (for tree visualization)
pip install graphviz
```

## Differences from Original Code

| Aspect | Original | New |
|--------|----------|-----|
| Structure | Monolithic scripts | Modular files |
| DEAP | Duplicated in each mode | Isolated in `evolution_base.py` |
| Config | Scattered dicts | Centralized in `config.py` |
| Mechanisms | Inline in model | Functions in `mechanisms.py` |
| Money | Unconditional cooperation | Optional `REQUIRE_TOKENS_TO_COOPERATE` |
| Visualization | Duplicated per mode | Shared patterns |

## Troubleshooting

### "creator already exists" error

DEAP's global state persists between runs. Restart Python or use:

```python
import deap.creator
deap.creator.FitnessMax = None
deap.creator.Individual = None
```

### Mode 2 evaluation issues

Mode 2 uses hardcoded action rules. The `evaluate_partner()` method in `core_abm.py` checks if `action_rule is None` to trigger hardcoded behavior.

### Token initialization

Set `INITIAL_LIQUIDITY < 1.0` for probabilistic token distribution (e.g., 0.5 = 50% chance of 1 token).
