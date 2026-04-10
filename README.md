# iGSS Cooperation Game: Institutional Discovery

This repository contains a prototype Inverse Generative Social Science (iGSS) model. Instead of pre-programming agents with traditional game-theory strategies (like Tit-for-Tat or Image Scoring), we use Genetic Programming (GP) to let agents *discover* optimal institutional rules for cooperation.

Our goal is to reverse-engineer classical cooperation mechanisms to see under what conditions an AI will organically invent:
1. **Indirect Reciprocity** (Reputation/Standing)
2. **Direct Reciprocity** (Memory)
3. **Monetary Exchange** (Tokens)

## 📁 File Structure

The architecture is strictly decoupled into two files to separate the simulation environment from the evolutionary search algorithm.

* **`model.py` (The Environment):** A Mesa-based Agent-Based Model. It handles random matching, the payoffs of the cooperation game, and the underlying physics of Standing, Memory, and Token transfers. 
* **`evolution.py` (The Discovery Engine):** A DEAP-based Genetic Programming loop. It generates mathematical rules, plugs them into the Mesa environment, measures how well the agents survived, and evolves better rules over generations.

## ⚙️ How it Works: The Sensors

Our `iGSS-Agents` are blank slates. When they evaluate a partner for cooperation, they are given up to three pieces of data (Sensors). The DEAP engine writes a mathematical logic gate using these inputs. If the equation outputs a number `> 0`, the agent cooperates.

* **`ARG0` (Standing):** Does the partner have a good reputation? (1 = Yes, 0 = No)
* **`ARG1` (Memory):** Has this partner defected against me before? (1 = Yes, 0 = No)
* **`ARG2` (Tokens):** How many tokens does this partner have? (Integer)

## 🎛️ Running Experiments (The Control Room)

All experiments are controlled via the `MODEL_CONFIG` dictionary located at the top of `model.py`. 

**1. Testing Mechanisms:**
Toggle the "Master Switches" (`USE_STANDING`, `USE_MEMORY`, `USE_TOKENS`) to `True` or `False`. If a switch is `False`, the AI is blindfolded to that data (it receives a `0`), forcing it to evolve rules using only the active mechanisms.

**2. The "Hive Mind" vs. The "Mixed Population":**
You can alter the population distribution in the config to test different evolutionary pressures:
* **The Hive Mind:** Set `NUM_IGSS = 50` and the others to `0`. This tests if the AI can discover the optimal "utopian law" for a homogenous society.
* **The Mixed Population:** Add Unconditional Defectors (`NUM_D`) and Unconditional Cooperators (`NUM_UC`) into the room alongside the `NUM_IGSS` agents. This tests if the AI can discover defensive institutions (like Money) to survive exploitation.

## 🚀 Quick Start
To run the model, simply execute the evolution script in your terminal:
```bash
python evolution.py
