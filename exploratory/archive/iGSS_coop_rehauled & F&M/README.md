# IGSS Framework: Integrated Genetic Programming for Social Simulation

A unified, publication-ready framework for evolutionary experiments on cooperation mechanisms, consolidating 11 separate experimental scripts into a modular, extensible architecture.

## Overview

This framework implements experiments studying cooperation through:
- **Direct Reciprocity (DR)**: Memory-based interactions
- **Indirect Reciprocity (IR)**: Reputation/standing-based interactions  
- **Joint DR+IR**: Combined memory and standing mechanisms
- **Money-based Cooperation**: Token accumulation in closed economies

Each mechanism supports three experimental modes:
- **Mode 1**: Evolve Action Rules only
- **Mode 2**: Evolve Assessment Rules only (hardcoded action)
- **Mode 3**: Co-evolve both Action and Assessment Rules

## Architecture

```
igss_framework/
├── __init__.py          # Package initialization
├── config.py            # Configuration classes and presets
├── primitives.py        # GP primitive set definitions
├── agents.py            # Agent behavior and strategies
├── models.py            # Agent-based simulation model
├── evolution.py         # Evolutionary algorithm engine
├── visualization.py     # Plotting and reporting
├── experiments.py       # Experiment registry and runners
└── main.py              # Command-line interface
```

## Installation

```bash
# Clone or download the framework
cd igss_framework

# Install dependencies
pip install numpy matplotlib networkx mesa sympy deap

# Optional: Install as package
pip install -e .
```

## Quick Start

### Command Line

```bash
# List all available experiments
python -m igss_framework --list

# Run a specific experiment
python -m igss_framework DR_mode1

# Run without plots (for batch processing)
python -m igss_framework IR_mode3 --no-plots

# Run quietly
python -m igss_framework jointDRIR_mode3_threetrees --quiet
```

### Python API

```python
from igss_framework import run_experiment, list_experiments

# List all experiments
print(list_experiments())

# Run an experiment
result = run_experiment("DR_mode1")

# Access results
print(f"Best fitness: {result.history.max_igss[-1]}")
print(f"Control baseline: {result.control_fitness}")
```

## Available Experiments

| Experiment | Mechanism | Mode | Description |
|------------|-----------|------|-------------|
| `DR_mode1` | DR | 1 | Evolve action rules (Tit-for-Tat baseline) |
| `DR_mode2` | DR | 2 | Evolve memory assessment rules |
| `DR_mode3` | DR | 3 | Co-evolve action + assessment |
| `IR_mode1` | IR | 1 | Evolve action rules (Standing baseline) |
| `IR_mode2_donor` | IR | 2 | Evolve assessment with donor standing |
| `IR_mode2_nodonor` | IR | 2 | Evolve assessment without donor standing |
| `IR_mode3` | IR | 3 | Co-evolve action + assessment |
| `jointDRIR_mode1` | Joint | 1 | Evolve hybrid action rules |
| `jointDRIR_mode2_AND_OR_twotrees` | Joint | 2 | Evolve dual assessment (AND/OR logic) |
| `jointDRIR_mode3_threetrees` | Joint | 3 | Total co-evolution (3 trees) |
| `MONEY_mode1_integer_exclusive_defenced` | Money | 1 | Token-based cooperation |

## Functional Equivalence

This framework maintains **exact functional equivalence** with the original 11 scripts:

- ✅ Identical random seeds (42, 421 where applicable)
- ✅ Identical configuration parameters
- ✅ Identical GP operators and probabilities
- ✅ Identical simulation resolution logic
- ✅ Identical agent behavior strategies
- ✅ Identical visualization output

## Customization

### Custom Experiment Configuration

```python
from igss_framework import (
    ExperimentConfig, ModelConfig, EvoConfig, 
    PrimitiveConfig, ExperimentRegistry, run_experiment
)

# Create custom configuration
config = ExperimentConfig(
    name="My_Experiment",
    mode="mode1",
    mechanism="DR",
    model=ModelConfig(
        benefit_to_cost_ratio=3,
        num_igss=20,
        num_rounds=50,
    ),
    evolution=EvoConfig(
        pop_size=50,
        max_gens=100,
    ),
    action_primitive=PrimitiveConfig.dr_action(),
    seed=123,
)

# Register and run
ExperimentRegistry.register(config.name, config)
result = run_experiment(config.name)
```

### Batch Processing

```python
import numpy as np
from igss_framework import run_experiment

# Run multiple seeds
results = []
for seed in range(42, 47):
    # Create config with different seed
    # ... (see example_usage.py)
    result = run_experiment(f"exp_seed_{seed}", show_plots=False)
    results.append(result.history.max_igss[-1])

print(f"Mean: {np.mean(results):.2f}, Std: {np.std(results):.2f}")
```

## Key Features

### 1. Modular Design
- Clear separation of concerns
- Easy to extend with new mechanisms
- Pluggable components

### 2. Type Safety
- Full type hints throughout
- Dataclass-based configurations
- Immutable config objects

### 3. Reproducibility
- Fixed random seeds
- Deterministic execution
- Version-controlled configurations

### 4. Academic Rigor
- Exact functional equivalence to original
- Comprehensive documentation
- Publication-ready code structure

## Configuration Reference

### ModelConfig
```python
ModelConfig(
    benefit_to_cost_ratio=5.0,  # Payoff multiplier
    cost=1.0,                    # Cooperation cost
    num_igss=10,                 # iGSS agent count
    num_uc=10,                   # Unconditional cooperator count
    num_d=10,                    # Defector count
    num_rounds=100,              # Simulation rounds
    action_logic="AND",          # Joint mode: AND/OR logic
    initial_endowment=1,         # Money: starting tokens
    endowment_fraction=0.5,      # Money: % with tokens
)
```

### EvoConfig
```python
EvoConfig(
    pop_size=40,           # Population size
    max_gens=100,          # Maximum generations
    parsimony_tax=0.1,     # Tree size penalty
    crossover_prob=0.5,    # Crossover probability
    mutation_prob=0.2,     # Mutation probability
    evaluation_runs=3,     # Simulation runs per eval
    baseline_runs=5,       # Runs for control baseline
    tournament_size=3,     # Selection tournament size
)
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{igss_framework,
  title={IGSS Framework: Integrated Genetic Programming for Social Simulation},
  author={[Your Name]},
  year={2024},
  url={[Your Repository]}
}
```

## License

[Your License]

## Contributing

[Your Contributing Guidelines]
