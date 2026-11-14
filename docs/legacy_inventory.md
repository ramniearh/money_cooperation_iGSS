<!-- AI-generated draft (Claude, Anthropic) for checking and amendment by the authors. -->

# Inventory of the legacy code

*AI-generated draft (Claude, Anthropic), written for phase 1 of `docs/REBUILD_BRIEF.md`, for checking and amendment by the authors.*

This file maps what is in `legacy/`. It was written by reading the files. No legacy script was run, and nothing in `legacy/` was changed.

- It describes what the code does. It does not compare the code with `docs/SPEC.md`; that is phase 2.
- Values in the legacy CSV files are not summarised or interpreted. Only their columns and row layout are described.
- Legacy identifiers and labels (variable names, column names, dictionary strings) are quoted in `code font` exactly as they appear, including where they use terms that SPEC §3 does not.

**How the files were read.**

- Read in full: the six model scripts, `config.py`, `utilities.py`, `README.md`, both R scripts, and in `under_development/` the files `LINKED_DRIR_mode3.py`, `jointDRIR_mode1.py`, `money_mode1_integer_exclusive.py` and `FS23lasttry.py`.
- Read as a complete line-by-line difference against a file read in full: `IR_mode2.py`, `IR_mode3.py`, and the other thirteen scripts in `under_development/`.
- Read by header and structure only: the six CSV files.
- Read in part: the zip archive (see its entry).
- Sampled: `legacy/figures/` (two images opened; the rest counted by filename pattern).

## 1. Summary

| Group | Files | Core or peripheral |
| --- | --- | --- |
| Model scripts | `DR_mode1.py`, `DR_mode2.py`, `DR_mode3.py`, `IR_mode1.py`, `IR_mode2.py`, `IR_mode3.py` | model core (each also contains its own run-and-log block) |
| Configuration | `config.py` | model core |
| Shared helpers | `utilities.py` | mixed: rule classification is core; logging and plotting are peripheral |
| R plotting | `visualizations.R`, `mode3_visualizations.R` | peripheral |
| Output data | `results_{DR,IR}_Mode{1,2,3}.csv` | output data |
| Documentation | `README.md` | documentation |
| Work in progress | `under_development/` (17 scripts, 1 zip archive) | outside the scope of SPEC v1.x; see §5 |
| Per-run images | `figures/` (untracked, git-ignored) | output data |

Tracked files: 35. `figures/` holds 7,690 PNG files that git does not track.

## 2. Shared files

### `legacy/config.py`

- **Purpose.** Holds the default parameters, the grid of parameters to sweep, and a function that expands the grid into one configuration dictionary per run, each with its own seed.
- **Contents.**
  - `DEFAULTS`: 21 keys. Game: `BENEFIT_TO_COST_RATIO`, `COST`, `NUM_IGSS`, `NUM_UC`, `NUM_D`, `NUM_ROUNDS`. Money variants: `INITIAL_ENDOWMENT`, `ENDOWMENT_FRACTION`. Error: `EXECUTION_ERROR_RATE`, `PERCEPTION_ERROR_RATE` (commented in the file as placeholders, not implemented). Search: `POP_SIZE`, `MAX_GENS`, `PARSIMONY_TAX`, `CXPB`, `MUTPB`, `TOURNAMENT_SIZE`, `HOF_SIZE`, `INIT_METHOD`, `TREE_MIN_DEPTH`, `TREE_MAX_DEPTH`, `STATIC_DEPTH_LIMIT`.
  - `EXPERIMENT_GRID`: seven keys with lists of values (`BENEFIT_TO_COST_RATIO`, `NUM_UC`, `NUM_D`, `NUM_ROUNDS`, `POP_SIZE`, `PARSIMONY_TAX`, `TREE_MAX_DEPTH`). As committed, the lists multiply to 32 combinations.
  - `RUNS_PER_CONFIG = 5` and `BASE_SEED = 42`.
  - `get_experiment_batches()`: for each combination (index `combo_idx`) and each repetition (index `run_idx`) it copies `DEFAULTS`, applies the combination, and adds `SEED = BASE_SEED + combo_idx * 1000 + run_idx`, `CONFIG_GROUP = "Config_<combo_idx+1>"` and `REPETITION = run_idx + 1`. As committed this gives 160 dictionaries.
  - `QUICK_TEST_CONFIG`: `DEFAULTS` plus `SEED = BASE_SEED`.
- **Which keys the scripts read.** A search of every `.py` file in `legacy/` finds these keys in `config.py` only: `EXECUTION_ERROR_RATE`, `PERCEPTION_ERROR_RATE`, `TOURNAMENT_SIZE`, `INIT_METHOD`, `TREE_MIN_DEPTH`, `TREE_MAX_DEPTH`, `STATIC_DEPTH_LIMIT`. No script reads them. `INITIAL_ENDOWMENT` and `ENDOWMENT_FRACTION` are read only by `under_development/money_mode1.py`. Every key is nevertheless written to the CSV, because the scripts log the whole dictionary.
- **Inputs and outputs.** Reads and writes no files.
- **Dependencies.** `itertools` (standard library).
- **How it is run.** Not run directly. Imported by the six model scripts and by seven scripts in `under_development/`.
- **Class.** Model core (configuration).

### `legacy/utilities.py`

- **Purpose.** Shared helpers for the model scripts: turn an evolved expression into a simplified formula and a truth table, append a run to a CSV, print a text report, and save two images per run.
- **Functions.**
  - `simplify_rule(gp_string, variables)`: evaluates the expression's text with Python `eval` (built-ins removed; `add`, `sub`, `mul` mapped to arithmetic; variables mapped to SymPy symbols) and returns `sympy.simplify` of the result. On failure it returns a string beginning `SymPy Parsing Error:`.
  - `evaluate_truth_table(simp_rule_str, variables)`: parses the **simplified** string with `sympy.sympify`, substitutes every combination of input values, and returns a tuple of booleans (`value > 0`). Combinations come from `itertools.product([True, False], repeat=n)`: all-true first, first variable most significant. A row that fails to evaluate is recorded as `False`; a string containing `Error` gives `None`.
  - `log_results_to_csv(run_log, filename)`: appends one row. The header is written only if the file does not yet exist, and is taken from the keys of that first row.
  - `hierarchy_pos`, `plot_deap_trees`: draw one expression tree, or two side by side when the individual is a list of two trees, with NetworkX and Matplotlib.
  - `generate_dashboard(best_ind, history, run_log, save_only)`: prints the text report, then writes `figures/<Run_ID>_Fitness.png` and `figures/<Run_ID>_Tree.png`. It creates `figures/` if absent. With `save_only=False` the plots are also shown on screen.
- **Inputs.** Arguments only. `history` must hold `max_igss`, `avg_igss`, and optionally `uc_scores`, `d_scores`, `fossil_record`. `run_log` is the dictionary that becomes the CSV row.
- **Outputs.** The CSV named by the caller; two PNG files per run under `figures/`, relative to the working directory; text on the terminal.
- **Dependencies.** `sympy`, `networkx`, `matplotlib`, `deap` (`gp.graph`); `os`, `csv`, `itertools`. No versions are pinned.
- **How it is run.** Not run directly. Imported by the same scripts as `config.py`.
- **Class.** Mixed. `simplify_rule` and `evaluate_truth_table` decide how a rule is classified, which is model core. The rest is peripheral (logging, plotting).

### `legacy/README.md`

- **Purpose.** Describes the repository as it stood in May 2026: the six model scripts, `config.py`, `utilities.py`, the two ways of running a script, and the outputs. Ends with a to-do list.
- **Dependencies it names.** Python 3.x, `mesa`, `deap`, `numpy`, `sympy`, `networkx`, `matplotlib`. No versions.
- **Class.** Documentation.

## 3. The six model scripts

All six share one structure. It is described once here; each entry below gives only what differs.

### Structure common to all six

- **Dependencies.** `mesa`, `deap` (`base`, `creator`, `tools`, `gp`), `numpy`; `operator`, `random`, `datetime`. From legacy: `config` (`QUICK_TEST_CONFIG`, `get_experiment_batches`) and `utilities` (`simplify_rule`, `evaluate_truth_table`, `log_results_to_csv`, `generate_dashboard`). Mesa is used for its base classes only: `mesa.Agent`, `mesa.Model`, and the model's `agents` collection. The constructor call `super().__init__(model)` is the Mesa 3 form.
- **Expressions.** Built at import time, as module-level globals. Primitives `add`, `sub`, `mul`. Terminals: the script's input variables and a constant drawn from `{-10, -1, 0, 1, 10}`. Initial expressions use DEAP `genHalfAndHalf` with depths 1 to 3, typed into the script. Selection is `selTournament` with size 3, typed into the script. Crossover is `cxOnePoint` and mutation is `mutUniform`. No depth limit is registered.
- **Agents.** One class, `CoopAgent`, with an `agent_type` string: `"iGSS-Agent"`, `"Unconditional Cooperator"`, `"Defector"`, or a control type whose name differs by script.
- **One play-through.** `CooperationModel` creates `NUM_IGSS` agents of the evolved (or control) type, `NUM_UC` unconditional cooperators and `NUM_D` unconditional defectors. All iGSS agents in a model carry the same rule. `step()` shuffles the agents, and each in turn acts as donor towards one other agent drawn at random. `resolve()` applies payoffs (donor loses `COST`; the other gains `BENEFIT_TO_COST_RATIO × COST`) and the stored-value update. `step()` is called `NUM_ROUNDS` times.
- **Scoring.** `evaluate_rule` (or `evaluate_rules` in mode 3) plays three play-throughs for one rule. The fitness is the mean payoff of the iGSS agents across the three, minus `PARSIMONY_TAX ×` the number of nodes in the expression (both expressions in mode 3). The untaxed mean and the mean payoffs of the two background types are kept on the individual as `sim_igss`, `sim_uc`, `sim_d`.
- **Control value.** `get_control_baseline` plays five play-throughs in which the iGSS agents are replaced by a hard-coded control type, and returns its mean payoff.
- **Search loop.** `run_evolution(config)`:
  1. seeds Python's `random` and `numpy.random` with `SEED`. Play-throughs and DEAP operators both draw from Python's global `random`;
  2. computes the control value, then scores the initial population;
  3. for each of `MAX_GENS` generations: tournament selection, crossover with probability `CXPB`, mutation with probability `MUTPB`, scoring of changed individuals only (unchanged ones keep their earlier score);
  4. copies the best individual found so far (`HallOfFame`, size `HOF_SIZE`) into the last place of the population;
  5. records, per generation, `sim_igss`, `sim_uc` and `sim_d` of the best individual and the population mean of `sim_igss`. The best expression's text is stored for generation 1 and every tenth generation (`fossil_record`).

  It returns the hall-of-fame individual, the history and the control value.
- **Run block.** Under `if __name__ == "__main__":` each script defines dictionaries mapping truth-table tuples to label strings, then holds two blocks:
  - "Option A": one run with `QUICK_TEST_CONFIG`, plots shown on screen;
  - "Option B": every configuration from `get_experiment_batches()`, plots saved only.

  One block is switched off by enclosing it in a triple-quoted string. For each run the script simplifies the returned expression, builds its truth table, looks up a label, appends one CSV row and calls `generate_dashboard`.
- **How they are run.** `python <script>.py`, with no arguments. The imports of `config` and `utilities` and the relative output paths mean the working directory must be the one that holds those two files: `legacy/` since the phase 0 move. (`__pycache__/` at the repository root holds compiled `config` and `utilities` for CPython 3.14, from runs before the move.)
- **Outputs.** One CSV, appended to, in the working directory; two PNG files per run in `figures/`; text on the terminal.
- **CSV columns shared by all six**, in order:
  - `Timestamp`, `Machine_ID` (always the literal `Local_VSCode`), `Run_ID`, `Mode`;
  - the 21 `DEFAULTS` keys, in the order listed under `config.py`;
  - `SEED`, and in batch runs `CONFIG_GROUP`, `REPETITION`.
- **Meaning of the score columns** (all taken from the final generation):
  - `iGSS_Max_Fitness`: untaxed mean iGSS payoff of the best individual in the final population;
  - `iGSS_Avg_Fitness`: mean of that quantity over the final population;
  - `UC_Fitness`, `D_Fitness`: mean payoff of each background type in the best individual's play-throughs;
  - `Control_Baseline`: the control value.

  The expression columns describe the hall-of-fame individual; the score columns describe the best individual of the final population.
- **Class.** Model core: each script contains rules, play-through, scoring and search. Its run block is peripheral.

### `legacy/DR_mode1.py`

- **Purpose.** Searches for an action rule with one input, `PartnerInMemory` (1 if the other agent's id is in the donor's `memory` set). The stored-value update is fixed.
- **Fixed update.** When a donor defects, its id is added to the other agent's `memory`. In the same step, if the other agent's id is in the donor's own `memory`, it is removed from it. Nothing is removed on cooperation.
- **Control type.** `"Control-TFT"`: cooperates when the other agent is not in `memory`.
- **Label dictionary.** `DR_NORMS_DICT`, four entries keyed by two-value tuples.
- **Run block as committed.** The "Option A" block is live: its opening triple quote is absent, so the string literal encloses "Option B" instead. As committed the script performs one run with `QUICK_TEST_CONFIG` and shows the plots on screen.
- **Output.** `results_DR_Mode1.csv`. `Run_ID` is `DR1_TEST_<HHMMSS>` (single run) or `DR1_<CONFIG_GROUP>_R<REPETITION>` (batch).
- **Columns after the shared ones.** `Best_Rule_Raw`, `Best_Rule_SymPy`, `Identified_Strategy`, `iGSS_Max_Fitness`, `iGSS_Avg_Fitness`, `UC_Fitness`, `D_Fitness`, `Control_Baseline`, `Manual_Validation` (always written empty).

### `legacy/DR_mode2.py`

- **Purpose.** Searches for an assessment rule with two inputs, `DonorAction` and `DonorInMemory`. The action rule is fixed: cooperate when the other agent is not in `memory`.
- **Evolved update.** An output above 0 adds the donor's id to the other agent's `memory`; otherwise it is removed.
- **Control type.** `"Control-Agent"`, with a hard-coded update: add the donor on defection, remove it on cooperation.
- **Label dictionary.** `DR_MEMORY_DICT`, seven entries keyed by four-value tuples.
- **Run block as committed.** "Option B" (batch) is live.
- **Output.** `results_DR_Mode2.csv`; `Run_ID` prefix `DR2_`. Columns as for `DR_mode1.py`.

### `legacy/DR_mode3.py`

- **Purpose.** Searches for an action rule and an assessment rule together. An individual is a list of two expressions, with the inputs of the two scripts above.
- **Operators.** `cxTwoTrees` and `mutTwoTrees` choose one of the two expressions with probability 0.5 (typed into the script) and apply `cxOnePoint` or `mutUniform` to it.
- **Control type.** `"Control-Agent"`, as in `DR_mode2.py`.
- **Label dictionaries.** `DR_ACTION_DICT`, `DR_MEMORY_DICT`, and a function `classify_joint_dr` that maps a pair of tuples to one label.
- **Run block as committed.** "Option B" is live.
- **Output.** `results_DR_Mode3.csv`; `Run_ID` prefix `DR3_`.
- **Columns after the shared ones.** `Best_Act_Raw`, `Best_Act_SymPy`, `Identified_Action`, `Best_Ass_Raw`, `Best_Ass_SymPy`, `Identified_Assessment`, `Joint_Strategy`, `iGSS_Max_Fitness`, `iGSS_Avg_Fitness`, `UC_Fitness`, `D_Fitness`, `Control_Baseline`. There is no `Manual_Validation` column.

### `legacy/IR_mode1.py`

- **Purpose.** Searches for an action rule with one input, `PartnerStanding` (the other agent's `standing`, 0 or 1, starting at 1). The update is fixed.
- **Fixed update.** A donor that cooperates gets standing 1. A donor that defects against an agent with standing 1 gets standing 0. A donor that defects against an agent with standing 0 keeps its standing.
- **Control type.** `"Control-Standing"`: cooperates when the other agent's standing is 1.
- **Label dictionary.** `IR_ACTION_DICT`, four entries.
- **Run block as committed.** "Option B" is live.
- **Output.** `results_IR_Mode1.csv`; `Run_ID` prefix `IR1_`. Columns as for `DR_mode1.py`.

### `legacy/IR_mode2.py`

- **Purpose.** Searches for an assessment rule with two inputs, `DonorAction` and `RecipientStanding`. The action rule is fixed: cooperate when the other agent's standing is 1.
- **Evolved update.** The donor's standing becomes 1 if the output is above 0, otherwise 0.
- **Control type.** `"Control-Agent"`, with the fixed update of `IR_mode1.py`.
- **Label dictionary.** `IR_ASSESSMENT_DICT`, ten entries.
- **Run block as committed.** "Option B" is live.
- **Output.** `results_IR_Mode2.csv`. The `Run_ID` prefix is `IR2NoD_`, and `Mode` is `Indirect Reciprocity Mode 2 (Assessment w/ No Donor Var)`. Columns as for `DR_mode1.py`.

### `legacy/IR_mode3.py`

- **Purpose.** Searches for an action rule and an assessment rule together, with the inputs of the two IR scripts above. Structure as `DR_mode3.py`.
- **Label dictionaries.** `IR_ACTION_DICT`, `IR_ASSESSMENT_DICT`, and `classify_joint_ir`.
- **Run block as committed.** "Option B" is live.
- **Output.** `results_IR_Mode3.csv`; `Run_ID` prefix `IR3_`. Columns as for `DR_mode3.py`.

## 4. R scripts and data files

### `legacy/visualizations.R`

- **Purpose.** A sequence of `ggplot2` charts of run-level outcomes, written to be run piece by piece. Each chart object is printed; none is saved to a file.
- **Inputs.** `read_csv("results_DR_Mode1.csv")` at line 5, and `read_csv("results_DR_Mode3.csv")` at lines 282, 330 and 378. The paths are relative to the R working directory.
- **Columns used.** `Identified_Strategy`, `Joint_Strategy`, `NUM_ROUNDS`, `BENEFIT_TO_COST_RATIO`, `NUM_UC`, `NUM_D`, `NUM_IGSS`, `POP_SIZE`, `TREE_MAX_DEPTH`, `PARSIMONY_TAX`.
- **Outputs.** Plots on the active graphics device only. The charts are listed in §7.
- **Dependencies.** `tidyverse`. No version is pinned.
- **How it is run.** Interactively. No `source()` call or command-line entry was found. An RStudio project file (`money_cooperation_iGSS.Rproj`, untracked) sits at the repository root, so the project's working directory is the root, while the CSV files have been in `legacy/` since phase 0.
- **Class.** Peripheral (plotting).

### `legacy/mode3_visualizations.R`

- **Purpose.** Four further charts of run-level outcomes by the `Joint_Strategy` label.
- **Inputs.** `read_csv("results_IR_Mode2.csv")` at line 7. The comment on the line above says "Updated to Mode 3".
- **Columns used.** `Joint_Strategy`, `iGSS_Max_Fitness`, `NUM_ROUNDS`, `BENEFIT_TO_COST_RATIO`, `NUM_UC`, `NUM_D`, `NUM_IGSS`, `POP_SIZE`, `TREE_MAX_DEPTH`, `PARSIMONY_TAX`.
- **Outputs.** Plots on the active graphics device only.
- **Dependencies.** `tidyverse`.
- **How it is run.** As `visualizations.R`.
- **Class.** Peripheral (plotting).

### `legacy/results_*.csv` (six files)

One row per run. Each was written by the model script of the same name through `log_results_to_csv`.

| File | Data rows | Columns | `Run_ID` form |
| --- | --- | --- | --- |
| `results_DR_Mode1.csv` | 1 | 35 | `DR1_TEST_<HHMMSS>` |
| `results_DR_Mode2.csv` | 160 | 37 | `DR2_Config_<n>_R<r>` |
| `results_DR_Mode3.csv` | 160 | 40 | `DR3_Config_<n>_R<r>` |
| `results_IR_Mode1.csv` | 160 | 37 | `IR1_Config_<n>_R<r>` |
| `results_IR_Mode2.csv` | 160 | 37 | `IR2NoD_Config_<n>_R<r>` |
| `results_IR_Mode3.csv` | 160 | 40 | `IR3_Config_<n>_R<r>` |

- The 160-row files hold 32 values of `CONFIG_GROUP` with 5 repetitions each.
- `results_DR_Mode1.csv` holds one single-run row and so has no `CONFIG_GROUP` or `REPETITION` column.
- Column names are those listed in §3: 37 columns for modes 1 and 2, 40 for mode 3.
- `Manual_Validation` is empty in every row of the four files that have it.
- `Timestamp` values fall between 26 August and 1 September 2026.
- **Class.** Output data. SPEC §17 says the legacy "Test summary" entries are not established, and CLAUDE.md rule 8 applies: nothing in these files is treated as a result here.

### `legacy/figures/` (untracked)

Git-ignored, so not part of the repository history. 7,690 PNG files, written by `generate_dashboard`.

| Filename pattern | Files |
| --- | --- |
| `DR{1,2,3}_Config_<n>_R<r>_Fitness.png` and `…_Tree.png` | 3,840 |
| `IR{1,3}_Config_<n>_R<r>_Fitness.png` and `…_Tree.png` | 2,560 |
| `IR2NoD_Config_<n>_R<r>_Fitness.png` and `…_Tree.png` | 1,280 |
| `DR1_TEST_<HHMMSS>_Fitness.png` and `…_Tree.png` | 10 |

- For every script prefix the configuration numbers run from 1 to 128, with repetitions 1 to 5.
- File dates fall in two groups: 1–4 May 2026 (about 6,070 files) and 23 August – 1 September 2026 (about 1,610 files), plus 10 files from 19 May.
- What the two kinds of image show is described in §7.

## 5. `legacy/under_development/`

None of these files is imported by the six model scripts. They concern models outside SPEC v1.x: both stored values available at once, and token-based variants. All are classed as work in progress; each contains its own rules, play-through, scoring and search.

They come in two forms.

- **Form A (uses the shared files).** Same structure as §3: imports `config` and `utilities`, has the two run blocks, appends to a CSV and saves images. As located, `config.py` and `utilities.py` are one directory up, so these imports resolve only if `legacy/` is on the Python path.
- **Form B (self-contained).** No legacy imports. Module-level `SEED`, `MODEL_CONFIG` and `EVO_CONFIG` dictionaries; seeding happens at import. Crossover and mutation probabilities (0.5, 0.2) are typed into the loop. Contains its own copies of the simplification and plotting functions. Performs one run, prints a report and shows plots on screen. Reads and writes no files. Dependencies: `mesa`, `deap`, `numpy`, `sympy`, `networkx`, `matplotlib`.

Both forms are run as `python <script>.py` with no arguments. No output file from any of these scripts is present in the repository.

### Both stored values, linked update (`LINKED_*`)

| File | Form | What is evolved | Inputs | Fixed part | CSV written |
| --- | --- | --- | --- | --- | --- |
| `LINKED_DRIR_mode1.py` | A | action rule | `PartnerInMemory`, `PartnerStanding` | one hard-coded update that changes `standing` and `memory` together | `results_Combined_Mode1.csv` |
| `LINKED_DRIR_mode2.py` | A | assessment rule | `DonorAction`, `DonorInMemory`, `RecipientStanding` | action: cooperate if standing is 1 and the agent is not in `memory` | `results_Combined_Mode2.csv` |
| `LINKED_DRIR_mode3.py` | A | both, as two expressions | as the two rows above | — | `results_Combined_Mode3.csv` |

- In modes 2 and 3 a single evolved output sets both stored values: above 0 gives standing 1 and removal from `memory`; otherwise standing 0 and addition to `memory`.
- Control types: `"Control-Skeptic"` (mode 1), `"Control-Agent"` (modes 2 and 3).
- `Run_ID` prefixes `Combined1_`, `Combined2_`, `Combined3_`.
- Modes 2 and 3 have the batch block only. Mode 3 writes the mode-3 columns of §3 without `Joint_Strategy`.

### `LINKED_UPDATED_UTILITIES.py`

- **Purpose.** A variant of `utilities.py`. It adds three optional control values to the report and the fitness chart (`Baseline_Skeptic`, `Baseline_TFT`, `Baseline_Standing`) and drops the `Joint_Strategy` line from the report.
- **Note.** Its opening docstring gives instructions for a script named `Combined_Mode3.py`, which is not in the repository. No file imports this module.
- **Class.** Peripheral, with the same core functions as `utilities.py`.

### Both stored values, separate updates (`mixed_*`)

| File | Form | What is evolved | Inputs | Fixed part | CSV written |
| --- | --- | --- | --- | --- | --- |
| `mixed_DRIR_mode1.py` | A | action rule | `PartnerStanding`, `PartnerInMemory` | `memory` and `standing` each updated by its own hard-coded rule | `results_Mixed_Mode1.csv` |
| `mixed_DRIR_mode2.py` | A | assessment rule | `DonorAction`, `RecipientStanding`, `DonorInMemory` | action: cooperate if standing is 1 **or** the agent is not in `memory` | `results_Mixed_Mode2.csv` |
| `mixed_DRIR_mode3.py` | A | both, as two expressions | as the two rows above | — | `results_Mixed_Mode3.csv` |

- The input order differs from the `LINKED_*` scripts (standing first).
- Two control types, `"Control-DR"` and `"Control-IR"`, each played five times. `Control_Baseline` is the larger of the two values.
- Columns differ from §3: `Control_DR_Score` and `Control_IR_Score` are added; `UC_Fitness`, `D_Fitness` and `Manual_Validation` are absent.
- `Run_ID` prefixes `MIXED1_`, `MIXED2_`, `MIXED3_`.

### Both stored values, self-contained (`jointDRIR_*`)

| File | Form | What is evolved | Notes |
| --- | --- | --- | --- |
| `jointDRIR_mode1.py` | B | action rule on `PartnerInMemory`, `PartnerStanding` | fixed update combines the `DR_mode1.py` and `IR_mode1.py` updates; control `"Control-Hybrid"` |
| `jointDRIR_mode2_AND_OR_twotrees.py` | B | two assessment expressions, one for `memory` and one for `standing` | four inputs: `DonorAction`, `DonorInMemory`, `DonorStanding`, `RecipientStanding`; fixed action chosen by `ACTION_LOGIC` (`"AND"` or `"OR"`) |
| `jointDRIR_mode3_threetrees.py` | B | three expressions: action, `memory` update, `standing` update | tournament size 7; its own crossover and mutation choose one of the three expressions |

Each sets its own population size, number of generations and size penalty in `EVO_CONFIG`; the three files use different values.

### `IR_mode2_reverse_action_nodonor.py`

- **Form.** B.
- **Purpose.** Searches for an assessment rule on `DonorAction`, `RecipientStanding`, as `IR_mode2.py` does, with two changes: the iGSS agents' fixed action is to cooperate only when the other agent's standing is 0, and every standing starts at 0. The control type keeps the unreversed action.

### Token-based variants

Each agent holds a `token` count. A cooperating donor may receive a token from the agent it helps. Background agents ignore tokens when deciding.

| File | Form | Inputs to the evolved rule | What differs |
| --- | --- | --- | --- |
| `money_mode1.py` | A | `PartnerToken`, `MyToken` | reads `INITIAL_ENDOWMENT` and `ENDOWMENT_FRACTION` from the configuration; tokens given at the start to a fraction of iGSS (or control) agents only; a token moves only if the donor is an iGSS or control agent. Writes `results_Money_Mode1.csv`, `Run_ID` prefix `MONEY1_`; columns as §3 modes 1–2 without `UC_Fitness`, `D_Fitness`, `Manual_Validation` |
| `MONEY_mode1_boolean.py` | B | `PartnerToken`, `MyToken` | token is 0 or 1; half of all agents start with one; a token moves only from a holder to a non-holder |
| `MONEY_mode1_integer.py` | B | `PartnerToken`, `MyToken` | token counts are unbounded; tokens given to a fraction of all agents; a token moves whenever the helped agent holds one |
| `money_mode1_integer_exclusive.py` | B | `PartnerToken`, `MyToken` | as the row above, but only iGSS (or control) agents receive tokens at the start |
| `MONEY_mode1_integer_exclusive_defenced.py` | B | `PartnerToken`, `MyToken` | differs from the row above in 12 lines: a token moves only if the donor is an iGSS or control agent; a different `BENEFIT_TO_COST_RATIO`; an opening comment saying it "SHOULD match the original r&s paper. but needs validation" |
| `FS23lasttry.py` | B | `MyToken`, `PartnerToken`, `MyRole` | one expression returns a signed whole number, clipped to ±10, read as a token amount asked for (as donor, `MyRole = 1`) or offered (when helped, `MyRole = -1`). Help happens only when the two agents' amounts match and the tokens are available. Tournament size 5 |

### `Kimi_Agent_refactor重构与命名.zip`

- **Purpose.** An archive of a refactored copy of the project, dated 1–2 May 2026.
- **Contents.** 25 members under `project/`: 13 text files and 12 compiled `.pyc` files for CPython 3.12.
  - Scripts with the names of the six model scripts, plus `mixed_DRIR_mode3.py` and `money_mode1.py`.
  - `config.py`, `utilities.py`, `README.md`.
  - Two modules that exist nowhere else in the repository: `igss_engine.py` (shared search loops for one, two and three expressions per individual) and `abm_base.py` (a `BaseAgent` class holding payoff and type).
- **How it was read.** Listed, and read through the archive without extracting anything. `README.md` and `abm_base.py` were read in full. For the other text members only the imports, function names and output filenames were read. The `.pyc` files were not opened.
- **Differences noted.** The scripts import `get_scenario_batches` (not `get_experiment_batches`) from `config`; the README refers to `SCENARIO_GRID`, `NUM_UD` and a `GENERATE_FIGURES` switch; output files include `results_mixed_DRIR_Mode3.csv` and `results_MONEY_Mode1.csv`.
- **Class.** Archive of an alternative version. Not used by any other file.

## 6. Environment

**What the legacy files state.**

- No dependency file exists in the repository (`requirements.txt`, `pyproject.toml`, `renv.lock` or similar). No package version is pinned anywhere.
- `legacy/README.md` names Python 3.x; the README inside the zip names Python 3.9+.
- Compiled files show which interpreters were used: CPython 3.14 (repository root `__pycache__/`) and CPython 3.12 (inside the zip).

**Packages the legacy code imports.**

| Language | Packages | Used by |
| --- | --- | --- |
| Python | `mesa`, `deap`, `numpy` | all model scripts |
| Python | `sympy`, `networkx`, `matplotlib` | `utilities.py` and the form B scripts |
| R | `tidyverse` (`readr`, `dplyr`, `ggplot2`), with the Brewer and viridis colour scales that `ggplot2` provides | both R scripts |

**Installed on the machine where this inventory was written** (recorded as "installed here", not as a requirement): Python 3.14.4; Mesa 3.5.1; deap 1.4.3; numpy 2.4.4; sympy 1.14.0; networkx 3.6.1; matplotlib 3.10.8. `pytest` is not installed. An R 4.4.1 directory exists, but `Rscript` is not on the shell's path, so the installed `tidyverse` version was not read.

**Proposed `requirements.txt` for the new code** (not created; shown for confirmation):

```
deap
sympy
pytest
```

- These are the three packages SPEC §11 and §13 call for: DEAP for the search, SymPy for the human-readable simplification, pytest for the tests.
- SPEC §13 proposes that dependencies be pinned. Which versions to pin is not stated in SPEC.md, so no version numbers are proposed here; it is listed as an open point in §8.
- Not proposed, with reasons:
  - `mesa`: excluded by SPEC §13 ("no Mesa").
  - `numpy`: the legacy model scripts use it only to seed its generator and to take one mean. Python's standard library covers both.
  - `networkx`, `matplotlib`: used only for the per-run images. They would be needed only if SPEC §10 comes to specify figures drawn in Python.
- The R scripts need `tidyverse` only. Whether R stays the plotting language is a phase 4 matter under the brief.

## 7. What the outputs look like

This section describes form only: what is plotted, from which columns, and how it is grouped.

### Written by the Python code, once per run

| Output | Produced by | Shows |
| --- | --- | --- |
| `<Run_ID>_Fitness.png` | `generate_dashboard` | A line chart. Horizontal axis: generation. Vertical axis: "Cumulative Payoff (Per Agent)". Four lines: the best individual's untaxed mean iGSS payoff; the population mean of the same; the mean payoff of unconditional cooperators and of defectors in the best individual's play-throughs. One horizontal line at the control value. The title is the `Mode` string and the `Run_ID`. |
| `<Run_ID>_Tree.png` | `plot_deap_trees` | A node-and-edge drawing of the returned expression: primitives and terminals as labelled circles, laid out as a tree. One panel for modes 1 and 2; two panels (action, assessment) for mode 3. |
| Terminal report | `generate_dashboard` | Run identifiers; the returned expression as raw text, simplified formula and label; the main parameters; a "maximum theoretical" payoff computed as `NUM_ROUNDS × (benefit − cost)`; the control value; final-generation payoffs; about five stored expressions spread across the generations. |
| CSV row | `log_results_to_csv` | One row per run, columns as in §3. There is no per-generation file: the per-generation history exists only in memory and in the fitness chart. |

### Drawn by the R scripts (not saved to files)

All take one row per run, and almost all first keep only rows with `NUM_UC == 10`, `NUM_D == 10`, `NUM_IGSS == 10`.

| Chart (object name) | File | Type | Shows | Grouping |
| --- | --- | --- | --- | --- |
| `chart1_overview` | `visualizations.R` | stacked bars, as shares | share of runs with each `Identified_Strategy` label | horizontal axis: `NUM_ROUNDS` × `BENEFIT_TO_COST_RATIO`; panels: `NUM_D` by `NUM_UC`. No filter |
| `chart1A_baseline` | `visualizations.R` | bars, as counts | number of runs with each `Identified_Strategy` label | panels: `BENEFIT_TO_COST_RATIO` by `NUM_ROUNDS` |
| `chart1B_simple` | `visualizations.R` | one stacked bar per panel | share of runs with each `Identified_Strategy` label | panels: `NUM_D` by `NUM_UC`. No filter |
| `chart2_baseline_one_strategy` | `visualizations.R` | grouped bars | share of runs whose `Identified_Strategy` equals the string in `TARGET_STRATEGY` | horizontal axis: `PARSIMONY_TAX`; colour: `TREE_MAX_DEPTH`; panels: `POP_SIZE` |
| `chart2B_baseline_one_strategy_by_ABM` | `visualizations.R` | grouped bars | as above | horizontal axis: `PARSIMONY_TAX`; colour: `TREE_MAX_DEPTH`; panels: `BENEFIT_TO_COST_RATIO` by `NUM_ROUNDS` |
| `chart_abm_faceted` | both | pie charts | share of runs with each `Joint_Strategy` label | panels: `BENEFIT_TO_COST_RATIO` by `NUM_ROUNDS` |
| `chart_igss_faceted` | `visualizations.R` | pie charts | as above | panels: `TREE_MAX_DEPTH` by `POP_SIZE` |
| `chart_parsimony_vs_pop` | both | pie charts | as above | panels: `PARSIMONY_TAX` by `POP_SIZE` |
| `chart_parsimony_vs_depth` | both | pie charts | as above | panels: `PARSIMONY_TAX` by `TREE_MAX_DEPTH` |
| `fitness_by_strategy_plot` | `mode3_visualizations.R` | horizontal box plots with points | `iGSS_Max_Fitness` per run | one box per `Joint_Strategy` label, ordered by median |

Every R chart counts runs by a **label** column (`Identified_Strategy` or `Joint_Strategy`). None uses the expression columns, a truth table, or anything per generation.

## 8. Unclear items

Points that could not be settled by reading the files.

1. **Which runs the images in `figures/` belong to.** Configuration numbers there run to 128; the committed grid gives 32, and the CSV files hold 32. An earlier, larger grid appears to have produced the May images, but no copy of it was found (the zip's `config.py` was not compared for this). Because `Run_ID` repeats across sweeps, later images may have replaced earlier ones of the same name.
2. **Where the batch output of `DR_mode1.py` is.** `results_DR_Mode1.csv` holds one single-run row, while `figures/` holds 1,280 `DR1_Config_…` images.
3. **Which version of each script wrote each CSV.** The CSV rows record no code version, and the files are appended to. Whether the committed scripts are the ones that produced the committed rows cannot be confirmed from the repository.
4. **How the R scripts were run, and on which file.** They hold several `read_csv` lines for different files and are evidently run in pieces. Some charts use a column that the file read just above them does not have (`Joint_Strategy` with a mode 1 or mode 2 file). The untracked `.Rhistory` at the repository root shows other files being loaded interactively. That file and `.RData` lie outside `legacy/` and were not inventoried.
5. **Whether two parts of the R scripts run as written.** In `mode3_visualizations.R`, lines 94–95 use `=` inside `filter()`. In `visualizations.R`, `TARGET_STRATEGY` is `"Tit-for-tat"`, while the label written by the Python scripts is spelt with a capital T in "Tat". Neither was run.
6. **Package versions used for the legacy runs.** Only the interpreter versions can be inferred (§6). The Mesa version in particular is unknown; the code uses the Mesa 3 constructor form.
7. **The R and `tidyverse` versions.** Not readable from the shell used.
8. **Origin and status of the zip archive.** Whether it was ever run, and how it relates to the present `legacy/` files, is not stated anywhere. Most of its members were read only in outline.
9. **`Combined_Mode3.py`.** Named in the docstring of `LINKED_UPDATED_UTILITIES.py`; not in the repository. It may be an earlier name of `LINKED_DRIR_mode3.py`.
10. **References in comments.** "the original r&s paper", "R&S model", "F&M", "FS23" and "Figure 1 of your paper" are not identified in any file.
11. **The "Test summary" mentioned in SPEC §17.** No file in `legacy/` contains that phrase. The nearest thing found is the `Manual_Validation` column, which is empty throughout.
12. **Which versions to pin in `requirements.txt`.** SPEC §13 proposes pinning but gives no versions (§6).
