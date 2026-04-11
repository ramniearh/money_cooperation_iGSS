# iGSS Model of Money and Cooperation Dynamics

This repository contains a prototype Inverse Generative Social Science (iGSS) model. Instead of pre-programming agents with traditional game-theory strategies (like Tit-for-Tat or Image Scoring), we use Genetic Programming (GP) to let agents *discover* optimal institutional rules for cooperation.

Our goal is to reverse-engineer classical cooperation mechanisms to see under what conditions an AI will endogenously discover mechanisms such as:
1. **Indirect Reciprocity** (Reputation/Standing)
2. **Direct Reciprocity** (Memory)
3. **Monetary Exchange** (Tokens)

## 📁 Core Modules

* **`model.py`:** The Mesa Agent-Based Model (ABM). Handles the environment, agent interactions, and payoff matrices.
* **`evolution.py`:** The DEAP Genetic Programming engine. Handles rule generation, fitness evaluation, and evolutionary progression.
* **`setup_go.py`:** The central execution module where all experimental parameters are configured and runs are initialized.

## ⚙️ Agent Sensors (Rule Arguments)

The GP engine constructs logic trees based on three available agent sensors. An agent cooperates if the evaluated equation outputs a value `> 0`.

* **`ARG0` (Memory / Direct Reciprocity):** Is the partner in the focal agent's defection memory? (1 = Yes, 0 = No)
* **`ARG1` (Standing / Indirect Reciprocity):** Does the partner have good standing? (1 = Yes, 0 = No)
* **`ARG2` (Tokens / Monetary Exchange):** What is the partner's current token balance? (Integer)

## 🎛️ Configuration & Usage

All experimental parameters must be configured manually inside `setup_go.py` prior to execution. Adjust the `CURRENT_MODEL_CONFIG` and `CURRENT_EVO_CONFIG` dictionaries to define the simulation.

**1. Mechanism Toggles:**
Enable or disable specific sensors by setting the following booleans to `True` or `False`:
* `USE_MEMORY`
* `USE_STANDING`
* `USE_TOKENS`

**2. Population Distribution:**
Define the environment's demographic makeup:
* `NUM_IGSS`: Number of evolving AI agents.
* `NUM_UC`: Number of Unconditional Cooperators.
* `NUM_D`: Number of Defectors.

**3. Output Modes:**
Set the `RUN_MODE` variable at the top of `setup_go.py` to dictate how data is handled:
* `"VISUAL"`: Renders a Matplotlib dashboard showing the evolutionary learning curve, final configurations, and a fossil record of intermediary strategies.
* `"BATCH"`: Runs headlessly. Exports all generational fitness data, configurations, and strategies to a timestamped `.json` file.

## 🚀 Execution

To initiate a run using the configured parameters, execute the following in your terminal:

```bash
python setup_go.py
```