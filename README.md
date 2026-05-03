# Evolution of Cooperation: Direct and Indirect Reciprocity via Genetic Programming

## Overview
Framework to study the evolution of cooperation through Evolutionary Game Theory and inverse Generative Social Science. An agent-based model with a random helping game is implemented in `mesa` and wrapped by `deap` genetic programming.Agent decision rule for action (behavior) and assessment (normative judgment) are evolved under scenarios of Direct Reciprocity (DR) and Indirect Reciprocity (IR). Further extensions include a monetary mechanism.

## Repository Structure

### 1. Core Configuration & Utilities
* **`config.py`**: defines the baseline parameters of the helping game (e.g., benefit-to-cost ratios, game length), evolutionary hyperparameters for the genetically-programmed rules (e.g., population size, mutation rates, parsimony pressure), and the experimental grid for batch execution.
* **`utilities.py`**: includes functions for simplifying GP mathematical trees into symbolic expressions using `sympy`, evaluating truth tables to classify evolved strategies against known theoretical baselines (e.g., Tit-for-Tat, Image Scoring), logging experimental data, and visualizing evolutionary fitness and GP parse trees.

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

### To-do list
Code & Architecture Fixes:
Truth Table Optimization: 
Replace eval() with toolbox.compile().
Performance Bottleneck: Fix $O(N^2)$ array generation in the ABM step() function.
Config Simplification: Split BENEFIT_TO_COST_RATIO into independent BENEFIT and COST.
Missing Configs: Move TOURNAMENT_SIZE, CXPB, MUTPB, and a future MEMORY_LENGTH into the EXPERIMENT_GRID.
Rename CoopAgent: Change to Actor or Player.
Rename DR Action Rule: Change "Tit-for-Tat" in the DR_NORMS_DICT to "Private Discriminator" (or "Conditional Cooperator") to maintain theoretical consistency with the IR modules.

Reporting & Paper Alignment:
Terminology Shift (Fossil): Replace "Fossil Record" with "Evolutionary Trajectory".
Terminology Shift (Populations): Rename POP_SIZE to RULE_GENEPOOL_SIZE or DEAP_POP_SIZE to prevent confusion with the ABM's physical agent population.
Enhanced Dashboard: Add total Compute Time, Standard Deviation of fitness, and a full dump of all active grid parameters.

Research Horizons:
Heterogeneous Evolution: Evolve separate, competing sub-populations of iGSS agents.10. Multi-Objective Optimization (MOO): (See below).