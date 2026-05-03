# Evolution of Cooperation: Direct and Indirect Reciprocity via Genetic Programming

## Overview
Framework to study the evolution of cooperation through Evolutionary Game Theory and inverse Generative Social Science, using `mesa` agent interactions and `deap` for evolutionary dynamics. Genetic Programming evolves agent decision rules for action (behavior) and assessment (normative judgment) under scenarios of Direct Reciprocity (DR) and Indirect Reciprocity (IR).

## Repository Structure

### 1. Core Configuration & Utilities
* **`config.py`**: The central configuration module. It defines the baseline economic parameters (e.g., benefit-to-cost ratios, game length), evolutionary hyperparameters (e.g., population size, mutation rates, parsimony pressure), and the experimental grid for batch execution.
* **`utilities.py`**: A suite of analytical tools. It includes functions for simplifying GP mathematical trees into symbolic expressions using `sympy`, evaluating truth tables to classify evolved strategies against known theoretical baselines (e.g., Tit-for-Tat, Image Scoring), logging experimental data, and visualizing evolutionary fitness and GP parse trees.

### 2. Direct Reciprocity (DR) Modules
These modules simulate pairwise interactions where agents rely on private memory ledgers of past encounters.
* **`DR_mode1.py` (Action Search)**: Fixes the memory assessment mechanism (agents remember defectors) and evolves the *action rule* (how an agent acts based on its memory of the partner).
* **`DR_mode2.py` (Assessment Search)**: Fixes the action rule to a strict conditional strategy (Tit-for-Tat) and evolves the *assessment rule* (the conditions under which a partner's actions warrant being added to or removed from the memory ledger; e.g., holding a grudge vs. forgiveness).
* **`DR_mode3.py` (Co-Evolution)**: Simultaneously evolves both the action rule and the assessment rule using a dual-tree genetic representation, allowing for the emergence of complex coupled strategies.

### 3. Indirect Reciprocity (IR) Modules
These modules simulate interactions where agents rely on a public reputation (standing), facilitating cooperation among strangers.
* **`IR_mode1.py` (Action Search)**: Fixes the normative assessment mechanism (e.g., the "Standing" norm) and evolves the *action rule* based exclusively on the recipient's public reputation.
* **`IR_mode2.py` (Assessment Search)**: Fixes the action rule to a strict discriminator strategy and evolves the *assessment rule* (the normative justice system that dictates how an agent's reputation is updated following a cooperative or defective action).
* **`IR_mode3.py` (Co-Evolution)**: Simultaneously co-evolves the action behavior and the normative assessment rule, allowing the system to discover stable reputation-based institutional norms without a priori assumptions.

## Dependencies
The framework requires Python 3.x and the following principal libraries:
* `mesa` (Agent-based modeling architecture)
* `deap` (Distributed Evolutionary Algorithms in Python)
* `numpy` (Numerical operations and random seed management)
* `sympy` (Symbolic mathematics for automated rule simplification)
* `networkx` & `matplotlib` (Graphing, hierarchy positioning, and data visualization)

## Usage

### Execution Modes
Each experimental script (e.g., `DR_mode1.py`) contains a main execution block with two distinct execution architectures:
1. **Single Visual Test**: Configured via `OPTION A`. Uncomment this block to run a single, rapid simulation instance. This mode provides detailed terminal reporting, translates the evolved GP trees into discrete strategic archetypes via Boolean truth tables, and displays structural visualizations of the rule trees and fitness trajectories.
2. **Batch Experiment**: Configured via `OPTION B`. This mode iterates over the `EXPERIMENT_GRID` defined in `config.py`, conducting multiple statistically independent runs for each parameter combination to ensure robust evolutionary findings.

### Outputs and Artifacts
Successful execution generates the following artifacts:
* **CSV Result Logs** (e.g., `results_DR_Mode1.csv`): Comprehensive datasets detailing hyperparameters, evolutionary performance metrics, baseline comparisons, and both the raw string and simplified symbolic representations of the dominant strategies.
* **Visualizations** (saved to the `/figures/` directory): Line charts tracking the evolutionary trajectory of agent fitness against theoretical control baselines, alongside NetworkX-generated topological maps of the evolved mathematical rule trees.
