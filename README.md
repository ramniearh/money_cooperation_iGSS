# iGSS Model of Money and Cooperation Dynamics

This repository contains a prototype Inverse Generative Social Science (iGSS) model. Instead of pre-programming agents with canonical game-theoretic strategies (e.g., Tit-for-Tat or Image Scoring), we use Genetic Programming (GP) to evolve decision rules that map institutional signals into cooperation behavior.

The objective is to analyze how adaptive agents *use* different informational and economic features—such as memory, reputation, and token holdings—to sustain cooperation, and under what conditions these features become behaviorally relevant. Long-term, we plan to use this as scaffolding for a broader research effort focused on building virtual social institutions in a process of self-organization through iGSS.

---

## Conceptual Overview

Agents have access to a configurable set of **institutional signals**, each relevant to one well-studied mechanism in the evolution of human cooperation:

* **Direct Reciprocity (Memory)**
* **Indirect Reciprocity (Standing/Reputation)**
* **Token-Based Transfers (Proto-Monetary Mechanism)**

The GP engine evolves strategies that determine how these signals are combined into cooperation decisions. This allows us to study the *endogenous emergence of behavioral rules conditional on institutional environments* (rather than the emergence of the institutions themselves).

---

## 📁 Core Modules

* **`model.py`**
  Implements the Mesa Agent-Based Model (ABM). Defines agent types, interaction rules, payoff structure, and institutional signals (memory, standing, tokens).

* **`evolution.py`**
  Implements the DEAP Genetic Programming engine. Handles rule generation, fitness evaluation, selection, mutation, crossover, and parsimony pressure.

* **`setup_go.py`**
  Entry point for experiments. Defines configurations, execution mode (visual or batch), and handles output (plots or JSON data).

---

## ⚙️ Agent Decision Structure

Each evolving agent (iGSS agent) uses a GP-generated function:

f(ARG0, ARG1, ARG2) → ℝ

An agent cooperates if:

f(...) > 0

### 🔍 Input Signals

* **ARG0 — Memory (Direct Reciprocity)**
  Binary indicator: `1` if the partner has defected against the focal agent in the past, `0` otherwise.
  (Implements a punishment-trigger memory rather than full interaction history.)
  > ⚠️ Note: A value of `1` is a **danger signal** — a well-adapted rule should tend to *defect* when ARG0 is high, not cooperate.

* **ARG1 — Standing (Indirect Reciprocity)**
  Partner's current standing (`0` or `1` in the current implementation), passed as a numeric signal.

* **ARG2 — Tokens (Proto-Monetary Signal)**
  Partner's current token balance (integer). When cooperation occurs, the **recipient transfers one token to the helper** — tokens therefore accumulate with agents who give cooperation and are depleted in agents who receive it. A token-rich partner signals a history of cooperative giving.

> Note: All inputs are provided as numeric values, allowing the GP engine to construct flexible mathematical combinations.

---

## 🧬 Evolutionary Process

Strategies are represented as symbolic expression trees composed of:

* Arithmetic operators (`+`, `-`, `*`)
* Conditional operator (`if_then`)
* Random constants

### Fitness Evaluation

Each strategy is evaluated by:

1. Running the agent-based model multiple times
2. Measuring the **average payoff of iGSS agents**
3. Applying a **parsimony penalty** to discourage overly complex rules:

fitness = average_payoff − (tree_size × parsimony_tax)

This introduces a trade-off between performance and simplicity.

---

## 🏛️ Institutional Configuration

All experiments are configured in `setup_go.py`.

### Mechanism Toggles

Enable or disable institutional signals:

* `USE_MEMORY` → Direct reciprocity
* `USE_STANDING` → Indirect reciprocity
* `USE_TOKENS` → Token-based transfers

### Population Composition

* `NUM_IGSS`: Evolving agents
* `NUM_UC`: Unconditional cooperators
* `NUM_D`: Defectors

### Economic Parameters

* `COST`: Cost of cooperation
* `BENEFIT_TO_COST_RATIO`: Multiplier for cooperative benefit
* `INITIAL_LIQUIDITY`: Initial token distribution

---

## 📊 Output Modes

Set `RUN_MODE` in `setup_go.py`:

### "VISUAL"

* Displays evolutionary learning curves
* Reports best discovered strategy
* Shows "fossil record" of intermediate strategies

### "BATCH"

* Saves results to a timestamped `.json` file
* Includes:

  * Fitness history
  * Configurations
  * Best evolved rule

---

## 🚀 Execution

Run the model with:

python setup_go.py

---

## 🛠️ Advanced Institutional Discovery (Modes 2 & 3)

While the base configuration (Mode 1) focuses on evolving the *Action Rule* (how to respond to predefined institutional signals), the repository also includes advanced configurations designed to study the endogenous emergence of the institutions themselves:

* **Mode 2: Evolving Assessment Rules (Social Norms)**
  In this mode, the agents' economic response is hardcoded to act as "Strict Discriminators" (cooperate if partner is Good, defect if Bad). The GP engine is tasked with evolving the "justice system" (the Assessment Rule). The algorithm searches for the logical categorization required to assign a "Good" or "Bad" reputation based on observed variables (`Action`, `HelperStanding`, `RecipientStanding`).

* **Mode 3: Co-Evolutionary Institutional Discovery**
  The most complex configuration. Agents utilize a **dual-tree Genetic Program** to simultaneously evolve *both* the economic response (Action Rule) and the social norm (Assessment Rule). The Action Rule receives only the partner's standing as input; memory and token signals are not available in this mode, making it a pure IR institutional search. This mode tests whether an evolutionary engine can endogenously discover foundational IR institutions (such as the "Stern Judging" norm) from scratch when placed under sufficient environmental pressure.
  > ⚠️ Note: Mode 3 uses a higher defector pressure (`NUM_D = 20` vs. `10` in Modes 1 and 2) to prevent indiscriminate cooperation from being rewarded. Results across modes are therefore not directly comparable in terms of environmental difficulty.