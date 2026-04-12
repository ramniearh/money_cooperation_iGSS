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

* **ARG1 — Standing (Indirect Reciprocity)**
  Partner’s current standing (`0` or `1` in the current implementation), passed as a numeric signal.

* **ARG2 — Tokens (Proto-Monetary Signal)**
  Partner’s current token balance (integer). Tokens can be transferred during cooperative interactions.

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
* Shows “fossil record” of intermediate strategies

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
  The most complex configuration. Fully agnostic agents utilize a **dual-tree Genetic Program** to simultaneously invent *both* the economic response (Action Rule) and the social norm (Assessment Rule). This mode tests whether an evolutionary engine can endogenously discover foundational, mathematically flawless social institutions (such as the "Stern Judging" norm) from scratch when placed under sufficient environmental pressure.

---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------

## Model Pseudocode (main mode)
# (see DEAP documentation for evolutionary algorithms)

# ==========================================
# iGSS COOPERATION MODEL: CORE ABM LOGIC
# ==========================================

# --- 1. GLOBAL PARAMETERS & TOGGLES ---
# Economics: Cost = 1, Benefit = 3, Initial Tokens = 2
# Population: N = 30 (10 iGSS, 10 Unconditional Cooperators, 10 Defectors)
# Toggles: USE_MEMORY, USE_STANDING, USE_TOKENS

# --- 2. MAIN SIMULATION LOOP (THE SCHEDULER) ---
FOR EACH generation in the evolutionary engine:
    FOR EACH round (1 to 20):
        Randomly shuffle all 30 agents
        
        # Agents interact in simultaneous, independent pairs
        FOR EACH mutually exclusive pair (Agent A, Agent B):
            
            # Step 1: Simultaneous Assessment
            Decision_A = evaluate_partner(Focal: Agent A, Partner: Agent B)
            Decision_B = evaluate_partner(Focal: Agent B, Partner: Agent A)
            
            # Step 2: Simultaneous Resolution
            resolve_interaction(Helper: Agent A, Recipient: Agent B, Action: Decision_A)
            resolve_interaction(Helper: Agent B, Recipient: Agent A, Action: Decision_B)

# --- 3. CORE ASSESSMENT LOGIC ---
FUNCTION evaluate_partner(Focal_Agent, Partner_Agent):
    
    # Handle Fixed Strategies
    IF Focal_Agent is Unconditional Cooperator:
        RETURN Cooperate
    IF Focal_Agent is Defector:
        RETURN Defect
        
    # Handle Adaptive iGSS Strategy
    IF Focal_Agent is iGSS Agent:
        # Construct input signals based only on active institutional toggles
        Arg0_Memory = 0
        Arg1_Standing = 0
        Arg2_Tokens = 0
        
        IF USE_MEMORY is ACTIVE:
            IF Partner_Agent's ID is currently in Focal_Agent's Memory list:
                Arg0_Memory = 1  # Partner previously defected against Focal Agent
                
        IF USE_STANDING is ACTIVE:
            Arg1_Standing = Partner_Agent's current Standing  # 1 (Good) or 0 (Bad)
            
        IF USE_TOKENS is ACTIVE:
            Arg2_Tokens = Partner_Agent's current Token balance
            
        # Evaluate the GP-evolved mathematical syntax tree
        Output = execute_gp_rule(Arg0_Memory, Arg1_Standing, Arg2_Tokens)
        
        # Action Rule Translation
        IF Output > 0:
            RETURN Cooperate
        ELSE:
            RETURN Defect

# --- 4. CORE RESOLUTION LOGIC ---
FUNCTION resolve_interaction(Helper, Recipient, Action):
    
    IF Action is COOPERATE:
        # Standard Game Theory Payoffs
        Subtract Cost (1) from Helper's cumulative payoff
        Add Benefit (3) to Recipient's cumulative payoff
        
        # Institutional Update: Reputation
        Helper's Standing becomes 1 (Good)
        
        # Institutional Update: Monetary Exchange
        IF USE_TOKENS is ACTIVE AND Recipient has > 0 Tokens:
            Transfer 1 Token from Recipient to Helper
            
    IF Action is DEFECT:
        # Institutional Update: Reputation
        # Standing is only lost if defecting against a "Good" agent
        IF Recipient's Standing == 1 (Good):
            Helper's Standing becomes 0 (Bad)
            
        # Institutional Update: Memory (Direct Reciprocity Blacklisting)
        Add Helper's ID to Recipient's Memory list
        
        # Institutional Update: Forgiveness 
        # Retaliation clears the grudge
        IF Recipient's ID was previously in Helper's Memory list:
            Remove Recipient's ID from Helper's Memory list