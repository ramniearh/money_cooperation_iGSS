# Original File to Framework Mapping

This document provides a detailed mapping between the original 11 experimental scripts and the unified IGSS framework, demonstrating exact functional equivalence.

## File-by-File Mapping

### 1. DR_mode1.py → `run_experiment("DR_mode1")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `DR_MODE1_CONFIG.seed = 42` |
| `MODEL_CONFIG` | `DR_MODE1_CONFIG.model` |
| `EVO_CONFIG` | `DR_MODE1_CONFIG.evolution` |
| `pset_action` | `PrimitiveConfig.dr_action()` |
| `CoopAgent.evaluate_partner()` | `agents.py:CoopAgent._evaluate_igss()` |
| `CoopAgent.evaluate_partner()` (Control) | `agents.py:CoopAgent._evaluate_control()` |
| `CooperationModel.resolve()` | `models.py:CooperationModel._resolve_dr()` |
| `evaluate_rule()` | `evolution.py:EvolutionaryEngine.evaluate_individual()` |
| `get_control_baseline()` | `evolution.py:EvolutionaryEngine.get_control_baseline()` |
| `run_evolution()` | `evolution.py:EvolutionaryEngine.run()` |
| `simplify_rule()` | `visualization.py:simplify_rule()` |
| `hierarchy_pos()` | `visualization.py:hierarchy_pos()` |
| `plot_deap_tree()` | `visualization.py:plot_tree()` |
| `plot_dashboard()` | `visualization.py:plot_dashboard()` |

**Key Equivalence Notes:**
- Control agent type: `"Control-TFT"`
- Action rule input: `PartnerInMemory` (0 or 1)
- Resolution: Memory update on defection, forgiveness if mutual

---

### 2. DR_mode2.py → `run_experiment("DR_mode2")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `DR_MODE2_CONFIG.seed = 42` |
| `pset_assessment` | `PrimitiveConfig.dr_assessment()` |
| `CoopAgent.evaluate_partner()` | Hardcoded TFT: `partner.unique_id not in self.memory` |
| `CooperationModel.resolve()` (Control) | Hardcoded memory update |
| `CooperationModel.resolve()` (Evolved) | `igss_assessment_rule(DonorAction, DonorInMemory)` |

**Key Equivalence Notes:**
- Mode 2: Action is hardcoded TFT, only assessment evolves
- Assessment inputs: `(DonorAction, DonorInMemory)`
- Memory update: `> 0` adds to memory, `<= 0` removes

---

### 3. DR_mode3.py → `run_experiment("DR_mode3")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `DR_MODE3_CONFIG.seed = 42` |
| `EVO_CONFIG` (boosted) | `pop_size=100, max_gens=300, parsimony_tax=0.05` |
| `pset_action` + `pset_assessment` | Two primitive sets |
| `creator.Individual = list` | `Individual_2trees` class |
| `cxTwoTrees()` | `GeneticOperators.cx_two_trees()` |
| `mutTwoTrees()` | `GeneticOperators.mut_two_trees()` |
| `init_individual()` | `IndividualFactory.create_toolbox()` |

**Key Equivalence Notes:**
- Two-tree individual: `[action_tree, assessment_tree]`
- Custom crossover: 50% chance per tree
- Custom mutation: 50% chance per tree
- Combined parsimony tax: `len(tree1) + len(tree2)`

---

### 4. IR_mode1.py → `run_experiment("IR_mode1")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `IR_MODE1_CONFIG.seed = 42` |
| `pset_action` | `PrimitiveConfig.ir_action()` |
| `agent.standing = 1` | `CoopAgent.standing` initialized to 1 |
| `Control-Standing` | `"Control-Standing"` agent type |
| `resolve()` (cooperates) | `donor.standing = 1` |
| `resolve()` (defects) | `if recipient.standing == 1: donor.standing = 0` |

**Key Equivalence Notes:**
- Standing: 1 = Good, 0 = Bad
- Control: Cooperate if `partner.standing == 1`
- Standing update: Helping restores good standing
- Defecting against good agent destroys standing

---

### 5. IR_mode2_donor.py → `run_experiment("IR_mode2_donor")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `IR_MODE2_DONOR_CONFIG.seed = 42` |
| `pset_assessment` (3 args) | `PrimitiveConfig.ir_assessment_donor()` |
| `renameArguments` | `ARG0='DonorAction', ARG1='DonorStanding', ARG2='RecipientStanding'` |
| `resolve()` (Control) | Standing norm hardcoded |
| `resolve()` (Evolved) | `igss_assessment_rule(action_val, donor.standing, recipient.standing)` |

**Key Equivalence Notes:**
- 3-variable assessment includes donor's own standing
- Action is hardcoded: cooperate if `partner.standing == 1`
- Standing update: `> 0` sets to 1, `<= 0` sets to 0

---

### 6. IR_mode2_nodonor.py → `run_experiment("IR_mode2_nodonor")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `IR_MODE2_NODONOR_CONFIG.seed = 42` |
| `pset_assessment` (2 args) | `PrimitiveConfig.ir_assessment_strict()` |
| `renameArguments` | `ARG0='DonorAction', ARG1='RecipientStanding'` |

**Key Equivalence Notes:**
- Strict 2-variable: only action and recipient's standing
- Excludes donor's standing from assessment
- Same hardcoded action as IR_mode2_donor

---

### 7. IR_mode3.py → `run_experiment("IR_mode3")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `IR_MODE3_CONFIG.seed = 42` |
| `EVO_CONFIG` (boosted) | `pop_size=100, max_gens=200` |
| `pset_action` + `pset_assessment` (strict) | Action + strict 2-var assessment |
| `evaluate_rules()` | Compiles both trees |

**Key Equivalence Notes:**
- Co-evolution of action and assessment
- Assessment uses strict 2-variable form
- Same two-tree operators as DR_mode3

---

### 8. jointDRIR_mode1.py → `run_experiment("jointDRIR_mode1")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `JOINT_MODE1_CONFIG.seed = 42` |
| `pset_action` (2 args) | `PrimitiveConfig.joint_action()` |
| `renameArguments` | `ARG0='PartnerInMemory', ARG1='PartnerStanding'` |
| `Control-Hybrid` | `"Control-Hybrid"` agent type |
| Control logic | `(not in_memory) and (standing == 1)` |
| `resolve()` | Combined DR + IR updates |

**Key Equivalence Notes:**
- Hybrid action uses both memory and standing
- Control: AND logic (both conditions must pass)
- Resolution: Both memory and standing updates occur

---

### 9. jointDRIR_mode2_AND_OR_twotrees.py → `run_experiment("jointDRIR_mode2_AND_OR_twotrees")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 421` | `JOINT_MODE2_CONFIG.seed = 421` |
| `ACTION_LOGIC = "OR"` | `ModelConfig.action_logic = "OR"` |
| `EVO_CONFIG` (modified) | `pop_size=60, max_gens=150` |
| `pset_assess` (4 args) | `PrimitiveConfig.joint_assessment()` |
| `renameArguments` | `DonorAction, DonorInMemory, DonorStanding, RecipientStanding` |
| Two-tree individual | `[memory_tree, standing_tree]` |
| `cxTwoTrees()` | Crossover on one tree randomly |
| `resolve()` (Evolved) | Separate mem_score and stand_score |

**Key Equivalence Notes:**
- Action logic toggle: AND vs OR
- Two assessment trees evolved separately
- Memory update: Tree 0
- Standing update: Tree 1

---

### 10. jointDRIR_mode3_threetrees.py → `run_experiment("jointDRIR_mode3_threetrees")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `JOINT_MODE3_CONFIG.seed = 42` |
| `MODEL_CONFIG` (modified) | `num_igss=30` |
| `EVO_CONFIG` (massive) | `pop_size=500, max_gens=1000, parsimony_tax=0.01` |
| `pset_action` + `pset_assess` | Action + assessment primitive sets |
| Three-tree individual | `[action, memory, standing]` |
| `cxThreeTrees()` | `GeneticOperators.cx_three_trees()` |
| `mutThreeTrees()` | `GeneticOperators.mut_three_trees()` |
| `init_individual()` | Creates 3 trees |
| `tournament_size=7` | Increased selection pressure |

**Key Equivalence Notes:**
- Three-tree co-evolution
- Action tree: decides cooperation
- Memory tree: updates personal memory
- Standing tree: updates public standing
- Massive search space requires larger population

---

### 11. MONEY_mode1_integer_exclusive_defenced.py → `run_experiment("MONEY_mode1_integer_exclusive_defenced")`

| Original Element | Framework Equivalent |
|------------------|---------------------|
| `SEED = 42` | `MONEY_MODE1_CONFIG.seed = 42` |
| `MODEL_CONFIG` (modified) | `benefit_to_cost_ratio=2` |
| `INITIAL_ENDOWMENT = 1` | `ModelConfig.initial_endowment = 1` |
| `ENDOWMENT_FRACTION = 0.5` | `ModelConfig.endowment_fraction = 0.5` |
| `pset_action` (2 args) | `PrimitiveConfig.money_action()` |
| `renameArguments` | `ARG0='PartnerToken', ARG1='MyToken'` |
| `agent.token = 0` | `CoopAgent.token` initialized to 0 |
| Central Bank logic | `_initialize_money()` method |
| Control-Agent | `"Control-Agent"` with `partner.token >= 1` |
| `resolve()` (Token settlement) | Token moves if donor cares AND recipient has token |

**Key Equivalence Notes:**
- Closed economy: tokens conserved
- Central Bank: 50% of economic agents get 1 token
- Settlement: Token moves from recipient to donor
- Only economic agents (iGSS/Control) care about tokens

---

## Function Call Chain Equivalence

### Original Script Execution Flow

```
__main__
  └── run_evolution()
      ├── get_control_baseline()
      │   └── CooperationModel(control_mode=True)
      ├── toolbox.population()
      ├── toolbox.evaluate() [for each individual]
      │   └── evaluate_rule()
      │       └── CooperationModel()
      │           └── step() [NUM_ROUNDS times]
      ├── toolbox.select()
      ├── toolbox.mate() / toolbox.mutate()
      └── plot_dashboard()
```

### Framework Execution Flow

```
run_experiment()
  └── ExperimentRegistry.run_experiment()
      ├── ExperimentRegistry.create_experiment()
      │   ├── PrimitiveSetFactory.create()
      │   └── IndividualFactory.create_toolbox()
      └── EvolutionaryEngine.run()
          ├── EvolutionaryEngine.get_control_baseline()
          │   └── CooperationModel(control_mode=True)
          ├── toolbox.population()
          ├── toolbox.evaluate() [for each individual]
          │   └── EvolutionaryEngine.evaluate_individual()
          │       └── CooperationModel()
          │           └── step() [NUM_ROUNDS times]
          ├── toolbox.select()
          ├── toolbox.mate() / toolbox.mutate()
          └── plot_dashboard()
```

**The call chains are functionally identical.**

---

## State Variable Equivalence

| Original | Framework | Type | Purpose |
|----------|-----------|------|---------|
| `agent.payoff` | `CoopAgent.payoff` | float | Cumulative payoff |
| `agent.memory` | `CoopAgent.memory` | set | DR: defectors remembered |
| `agent.standing` | `CoopAgent.standing` | int | IR: 1=Good, 0=Bad |
| `agent.token` | `CoopAgent.token` | int | Money: token holdings |
| `model.benefit` | `CooperationModel.benefit` | float | Payoff benefit |
| `model.cost` | `CooperationModel.cost` | float | Payoff cost |
| `individual.fitness` | `individual.fitness` | FitnessMax | DEAP fitness |
| `individual.sim_igss` | `individual.sim_igss` | float | Avg iGSS payoff |

---

## Random Number Sequence Equivalence

For exact reproducibility, the framework maintains:

1. **Same seed setting**: `random.seed(SEED)` and `np.random.seed(SEED)`
2. **Same sequence**: All random operations occur in same order
3. **Same GP initialization**: `gp.genHalfAndHalf` with same parameters
4. **Same selection**: `tools.selTournament` with same tournament size
5. **Same crossover**: `gp.cxOnePoint` with same probability
6. **Same mutation**: `gp.mutUniform` with same probability

**Verification**: Running original and framework with same seed should produce identical:
- Initial population
- Selection outcomes
- Crossover points
- Mutation points
- Simulation outcomes

---

## Backward Compatibility Functions

For users transitioning from original scripts, the framework provides:

```python
from igss_framework.experiments import (
    run_DR_mode1, run_DR_mode2, run_DR_mode3,
    run_IR_mode1, run_IR_mode2_donor, run_IR_mode2_nodonor, run_IR_mode3,
    run_joint_mode1, run_joint_mode2, run_joint_mode3,
    run_money_mode1,
)

# Direct equivalents to original scripts
result = run_DR_mode1()
result = run_IR_mode3()
result = run_joint_mode2(logic="AND")
```

These functions provide the exact same interface as the original `if __name__ == "__main__"` blocks.
