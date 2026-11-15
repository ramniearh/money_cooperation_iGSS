<!-- AI-generated draft (Claude, Anthropic) for checking and amendment by the authors. -->

# Rebuild plan: legacy code compared with the specification

*AI-generated draft (Claude, Anthropic), written for phase 2 of `docs/REBUILD_BRIEF.md`, for checking and amendment by the authors.*

This file compares the code in `legacy/` with `docs/SPEC.md` (v1.1) and lists the choices mode 1 still needs. It decides nothing. Every proposal in it is for the authors to accept, change or reject.

**How it was written.**

- The six model scripts, `config.py`, `utilities.py` and both R scripts were read in full for this file. Every file-and-line citation was taken from the source, not from `docs/legacy_inventory.md`.
- No legacy script was run. No file in `legacy/` was changed, and neither were `docs/SPEC.md` or `CLAUDE.md`.
- DEAP's behaviour is described from the source of the version installed on the machine used (DEAP 1.4): `selTournament`, `selRandom`, `staticLimit`, `mutUniform`, `cxOnePoint`, `genHalfAndHalf` and `HallOfFame.update`.
- `git diff main SSC26` was read for the history of the label dictionaries.
- No value from a legacy CSV file is cited or interpreted.

**Conventions.**

- Legacy identifiers and label strings are quoted in `code font` or in quotation marks exactly as they appear, including where they use terms that SPEC §3 does not.
- Bit strings follow SPEC §5: all-ones row first, first input most significant. Where a legacy tuple is translated into a SPEC bit string, the translation was done by hand and is shown, so that it can be checked. In the rebuild the same strings are derived by code (CLAUDE.md rule 3).
- The six model scripts share one structure. `DR_mode1.py` is cited as the representative; the other scripts are cited where they differ. The matching line numbers in `IR_mode1.py` are five lower than in `DR_mode1.py` from the scoring function onwards, and are given where the point matters.
- `legacy/under_development/` and the zip archive are outside this comparison: SPEC §17 names the six model scripts, and SPEC v1.x covers DR and IR only.
- **SPEC status** in the tables uses three values:
  - **same**: SPEC.md decides the point (section cited) and the legacy code agrees;
  - **contradicts**: SPEC.md decides the point and the legacy code does something else;
  - **silent**: SPEC.md does not decide the point. Each silent point that mode 1 needs has a row in §3.

---

## 1. Silent choices in the legacy code

### 1.1 Expressions and operators

| # | File and line | What the legacy code does | SPEC status |
| --- | --- | --- | --- |
| 1 | `DR_mode1.py:15-16`, `:24` (same lines in the other five) | Constants are drawn from `[-10, -1, 0, 1, 10]` through one DEAP ephemeral constant. | **contradicts** §9 and §13 (constants {0, 1}). **Silent** on whether the constants are two fixed terminals or one ephemeral constant (§3, row N1). |
| 2 | `DR_mode1.py:21-23` | Primitives `add`, `sub`, `mul`. | **same** (§9). |
| 3 | `DR_mode1.py:62`; `utilities.py:47` | An output above 0 counts as 1. | **same** (§5, §9). |
| 4 | `DR_mode1.py:32`; `IR_mode1.py:31`; `DR_mode2.py:31`; `IR_mode2.py:31`; `DR_mode3.py:39-40`; `IR_mode3.py:40-41` | Initial expressions from `genHalfAndHalf` with `min_=1, max_=3`, typed into the script. The numbers are DEAP tree heights (a lone terminal has height 0). | Values **same** (§9, §13). Typed in rather than read from the config: **contradicts** §11 and CLAUDE.md rule 7. **Silent** on whether "depth" means DEAP height (row N3). |
| 5 | `DR_mode1.py:38`; `IR_mode1.py:37` | Mutation (`mutUniform`) builds the new sub-expression with the same generator as initialisation: `genHalfAndHalf`, height 1–3. | **contradicts** §9 step 5 (depth 0–2). **Silent** on which generator builds it (row N2). |
| 6 | `DR_mode1.py:39`; `IR_mode1.py:38`; `DR_mode2.py:38`; `IR_mode2.py:38`; `DR_mode3.py:69`; `IR_mode3.py:70` | `selTournament` with `tournsize=3`, typed into the script. DEAP draws the three with replacement and keeps the first of the best in draw order. | Size **same** (§9, §13); typed in: **contradicts** rule 7. **Silent** on drawing with or without replacement (row N4). |
| 7 | `DR_mode1.py:37` | Crossover is `cxOnePoint`. DEAP leaves a pair unchanged when either expression is a single node. | **same** (§9 step 4). |
| 8 | all six scripts | No depth limit is registered: `staticLimit` appears nowhere in `legacy/`. | **contradicts** §9 step 6 (limit 17). |
| 9 | `DR_mode3.py:54`, `:61`; `IR_mode3.py:55`, `:62` | In mode 3, crossover and mutation act on one of the two expressions, chosen with probability 0.5 typed into the script. | **silent**. Mode 3 only (§14); not needed for mode 1. |
| 10 | `DR_mode1.py:26-28`, `:30-39` | The DEAP classes and toolbox are built at import time as module-level globals, guarded by `hasattr`. | **silent** (an implementation matter; listed so that the rebuild's choice is stated when `search.py` is planned). |

### 1.2 The play-through

| # | File and line | What the legacy code does | SPEC status |
| --- | --- | --- | --- |
| 11 | `DR_mode1.py:74-76` | Agents are created in the order iGSS, unconditional cooperators, unconditional defectors. Mesa assigns their ids in that order. | **silent** on how `agent_id` is assigned and on the order of the agent list before the first shuffle (row N5). |
| 12 | `DR_mode1.py:79-80` | Each round, all agents are shuffled with the global `random.shuffle`. | Shuffle: **same** (§4 step 1). Generator: **contradicts** §9 (episodes use their own generator). |
| 13 | `DR_mode1.py:83-85` | Each donor's partner is drawn uniformly from all other agents. Draws are independent across donors, so in one round an agent may be the partner of several donors or of none. A donor is never matched with itself. | **same** (§4 step 2). |
| 14 | `DR_mode1.py:49`, `:60` | DR: each agent holds a set, `memory`, of ids. The input `PartnerInMemory` is 1 when the partner **is** in the set, which is the event that follows a defection. | Polarity **contradicts** §3 and §13 (1 = the event that would normally favour cooperating). |
| 15 | `DR_mode1.py:49` | DR: `memory` starts empty, so a partner never met is treated as not having defected. | **same** in effect (§4: records start at 1). |
| 16 | `DR_mode1.py:90-101` | DR mode 1 fixed update. On defection, the donor's id is added to the partner's `memory`; and if the partner's id is in the donor's own `memory`, it is removed. On cooperation nothing changes. | **contradicts** §6.1 (`1100`). See §2.2, mismatch M2. |
| 17 | `IR_mode1.py:48` | IR: every `standing` starts at 1. | **same** (§4). |
| 18 | `IR_mode1.py:88-96` | IR mode 1 fixed update. Cooperation sets the donor's standing to 1. Defection against a partner with standing 1 sets it to 0. Defection against a partner with standing 0 leaves the donor's standing as it was. | **contradicts** §6.2 (`1101` sets it to 1 in the last case). See §2.2, mismatch M1. |
| 19 | `DR_mode1.py:87-88` (`resolve` called inside the donor loop) | Stored values change at once, so later donors in the same round see them. | **same** (§4 step 7; §13 "Update timing", proposed). |
| 20 | `DR_mode1.py:90-101`; `IR_mode1.py:88-96` | The update is applied whatever the donor's type, background agents included. | **same** (§4 step 7). |
| 21 | `DR_mode1.py:69-70`, `:91-93` | The benefit is computed as `BENEFIT_TO_COST_RATIO × COST`. A cooperating donor loses the cost and the partner gains the benefit. | Payoffs **same** (§4 step 6). **Silent** on whether the config holds b or b/c (row N17). |
| 22 | `config.py:21-23` | Two error rates exist as keys, commented "Placeholders (not implemented)". No script reads them. | Execution error is **open** in §13. A key that nothing reads **contradicts** rule 7. |
| 23 | `DR_mode1.py:51-53` | Background agents decide by type alone and do not read the input. | **same** (§4 step 4). |

### 1.3 Scoring

| # | File and line | What the legacy code does | SPEC status |
| --- | --- | --- | --- |
| 24 | `DR_mode1.py:74`; `config.py:12`, `:26` | The number of iGSS agents in a play-through (`NUM_IGSS`) is a separate parameter from the size of the rule population (`POP_SIZE`). All iGSS agents in a play-through carry the same rule. | **contradicts** §3 (one symbol, P) and §7 (P copies). §7 records this difference. |
| 25 | `DR_mode1.py:120-132`; `IR_mode1.py:115-127` | Each rule is scored on three play-throughs, `runs = 3` typed into the script, and the three payoffs are averaged. | **contradicts** §7 (one episode per rule). §7 records this difference. |
| 26 | `DR_mode1.py:108`, `:130` | The score is the mean **total** payoff of the iGSS agents over the play-through. | **contradicts** §13 "Score unit" (payoff per round, proposed). |
| 27 | `DR_mode1.py:134-135`; `DR_mode3.py:178`; `IR_mode3.py:174` | A size penalty, `PARSIMONY_TAX ×` the number of nodes, is subtracted from the score used for selection. In mode 3 the nodes of both expressions are counted. | **contradicts** §13 "Size control" (no size penalty, proposed). |
| 28 | `DR_mode1.py:107-110` | A type with no agents is reported with mean payoff 0. | **silent** on what is logged for a background type with N_C = 0 or N_D = 0, both of which occur in the first sweep (row N10). |
| 29 | `DR_mode1.py:138-146`, `:159`; `IR_mode1.py:133-141`, `:154` | A control value is computed from five play-throughs (`runs = 5`, typed in) in which a hard-coded rule replaces the evolved one. It is computed after seeding and before the first scoring, so it consumes random draws from the shared stream. | **contradicts** §8 (the baseline is a separate set of runs, for all four rules, alone and in pairs). |
| 30 | `DR_mode1.py:121-123` | Every play-through of every rule continues the same global random stream. Two rules with the same bit string therefore get different scores. | **contradicts** §7 and §13 "Aggregate scoring seed" (same seed within a generation, proposed). |

### 1.4 The search loop

| # | File and line | What the legacy code does | SPEC status |
| --- | --- | --- | --- |
| 31 | `DR_mode1.py:150-152` | `config.get("SEED", 42)`: a missing seed is replaced by 42. The seed is given to Python's global `random` and to `numpy.random`. Play-throughs and DEAP operators both draw from the global `random`; nothing draws from NumPy's generator. | Fallback value: **contradicts** rule 1 (no silent defaults). Shared generator: **contradicts** §9 and §13 "Seed scheme". |
| 32 | `DR_mode1.py:163-165`, `:167` | The initial population is scored once before the loop; the loop then runs `MAX_GENS` times. Nothing is logged for the initial population. | **silent** on how generations are numbered, whether G counts scorings or breeding steps, and which population is "the final generation" (row N11). |
| 33 | `DR_mode1.py:174-177` | Selected rules are paired in order, (1st, 2nd), (3rd, 4th), and so on; with an odd population the last is unpaired. Both scores of a pair are deleted whenever the crossover draw succeeds, even if DEAP then leaves the pair unchanged. | Pairing **same** (§9 step 4). **Silent** on an odd P (row N18). |
| 34 | `DR_mode1.py:179-182` | Each rule is then mutated with probability `MUTPB`, independently of whether it was crossed. | **same** (§9 step 5). |
| 35 | `DR_mode1.py:184-187` | Only rules changed by crossover or mutation are scored again. An unchanged rule keeps the score it drew in an earlier generation. | **contradicts** §9 step 2 (every rule is scored in every generation). |
| 36 | `DR_mode1.py:155`, `:190`, `:192-193` | The best rule found so far (`HallOfFame`) is copied over the last member of the new population, with its old score. `config.get("HOF_SIZE", 1)` supplies a fallback of 1. | **contradicts** §9 step 7 and §13 "Elitism" (none, proposed). |
| 37 | `DR_mode1.py:171-172` | `config.get("CXPB", 0.5)` and `config.get("MUTPB", 0.2)`: fallback values typed into the script. | Values **same** (§9); fallbacks **contradict** rule 1. |
| 38 | `DR_mode1.py:195-199` | Per generation, four numbers are kept: the untaxed payoff of the rule with the best taxed score, the population mean of the untaxed payoff, and the two background means from that best rule's play-throughs. | **contradicts** §9 "Every generation logs" (every rule's expression, bit string and score; counts per bit string; and more). |
| 39 | `DR_mode1.py:201-202` | The best rule's expression is stored for generation 1 and every tenth generation only. | **contradicts** §9 (every expression is kept). |
| 40 | `DR_mode1.py:205`, `:244-251` | The function returns the hall-of-fame rule, and the CSV's expression columns describe it. The CSV's score columns come from the last generation's best rule. The two can be different rules. | **silent** as such; under §9 and §10 a run reports a population, not one returned rule. |

### 1.5 Classification and labels

| # | File and line | What the legacy code does | SPEC status |
| --- | --- | --- | --- |
| 41 | `DR_mode1.py:234-235`; `utilities.py:9-23`, `:25-51` | The returned expression is turned into text, simplified with SymPy, turned into text again, parsed again, and the truth table is read from the parsed formula. The compiled expression is not used. | **contradicts** §5 and §13 "Classification". |
| 42 | `utilities.py:37` | Rows come from `itertools.product([True, False], repeat=n)`: all-true first, first variable most significant. | **same** (§5). |
| 43 | `utilities.py:27-28`, `:33-34`, `:48-49` | A simplified string containing "Error", or one that fails to parse, gives no truth table at all. A single row that fails to evaluate is recorded as 0. | **contradicts** §5 (classification is exact and involves no parsing). |
| 44 | `DR_mode1.py:216-221`; `IR_mode1.py:211-216`; `DR_mode2.py:222-230`; `IR_mode2.py:212-223`; `DR_mode3.py:254-269`; `IR_mode3.py:251-269` | Labels are attached to truth-table tuples typed by hand. The labels are literature names or invented names. A tuple not in the dictionary gets a label beginning "Novel". | **contradicts** §5 and rule 3 (bit strings derived from predicates) and rule 5 (literature names only as labels derived from §16). |
| 45 | `DR_mode1.py:216-221` | Only one rule, the one returned at the end of the run, is classified. | **contradicts** §9 step 1 and §10 (every rule, every generation). |

### 1.6 Configuration and output

| # | File and line | What the legacy code does | SPEC status |
| --- | --- | --- | --- |
| 46 | `config.py:8-37` | One dictionary of fallback values, `DEFAULTS`, used "UNLESS they are specifically overwritten by the Grid" (`config.py:6`). | **contradicts** §7 for scoring (`SCORING` has no default). **Silent** on the shape of `config.py` otherwise (row N17). |
| 47 | `config.py:46-56` | The sweep grid varies `BENEFIT_TO_COST_RATIO`, `NUM_ROUNDS`, `POP_SIZE`, `PARSIMONY_TAX` and `TREE_MAX_DEPTH`. `TREE_MAX_DEPTH` is read by no script (§2.1, item 4). | **contradicts** §6.3 (different factors) and rule 7. |
| 48 | `config.py:62-63`, `:88` | Five repetitions per combination. `SEED = 42 + combo_idx × 1000 + run_idx`. The same seeds are used by every script that imports the config. | **silent** on how run seeds are assigned, and on whether they are shared across environments and scoring options (row N7). |
| 49 | `utilities.py:53-61` | Each run appends one row to a CSV. The header is written only when the file does not yet exist, from the keys of that first row. Rows from different sweeps and different code versions accumulate in one file. | **silent** on append or overwrite, file names and column names (row N13). |
| 50 | `DR_mode1.py:247-251` | Scores are rounded to two decimals before being written. | **silent** on number format (row N13). |
| 51 | `DR_mode1.py:239-242`, `:282` | `Run_ID` is built from the configuration group and repetition, so it repeats from one sweep to the next. `Machine_ID` is the literal `Local_VSCode`. No code version is recorded. | **silent** on run identifiers and on recording the code version (row N13). |
| 52 | `DR_mode1.py:243`, `:284` | The whole configuration dictionary is written to the row, including keys that nothing reads. | **contradicts** rule 7 in effect: the file records parameters that had no effect. |
| 53 | `DR_mode1.py:225-258` | The choice between a single run and a batch is made by moving a triple-quoted string in the source. | **contradicts** §11 (`run.py` is a command-line entry point). |
| 54 | `utilities.py:161-164` | The report prints a "Maximum Theoretical Fitness" of `NUM_ROUNDS × (benefit − cost)`. | **silent**. No such quantity is defined in SPEC.md. |

---

## 2. Known problems re-checked

### 2.1 The four problems listed in SPEC §17

**1. "The DR memory dictionaries use the wrong row order, so 'Grim Ledger' and 'Sticky Memory' are mislabelled." — Confirmed.**

- The dictionary is `DR_MEMORY_DICT` at `DR_mode2.py:222-230`, repeated at `DR_mode3.py:261-269`.
- Its inputs are (`DonorAction`, `DonorInMemory`) (`DR_mode2.py:232`). `evaluate_truth_table` produces rows in the order (1,1), (1,0), (0,1), (0,0) (`utilities.py:37`): cooperation by a donor **in** `memory`, cooperation by one not in it, defection by one in it, defection by one not in it.
- The comment at `DR_mode2.py:221` gives the order as "(C from Str, C from Foe, D from Str, D from Foe)", that is, not-in-memory first. This is the opposite of the order the code produces.
- An output above 0 puts or keeps the donor in `memory` (`DR_mode2.py:105-110`).

Translation into SPEC §14.1. The SPEC record is 1 when the donor is **not** in `memory`, and `previous_record` is 1 when `DonorInMemory` is 0. So SPEC row (1,1) is legacy position 2, (1,0) is position 1, (0,1) is position 4 and (0,0) is position 3, and each output is negated.

| Legacy tuple | Legacy label | SPEC bit string (by hand) | What SPEC §14.1 says that string does |
| --- | --- | --- | --- |
| `(False, False, True, True)` | "Tit-for-tat Ledger" | `1100` | what the donor did |
| `(True, True, True, True)` | "Always Remember" | `0000` | always 0 |
| `(False, False, False, False)` | "Never Remember" | `1111` | always 1 |
| `(False, True, False, False)` | "Grim Ledger" | `0111` | 0 only if the donor cooperates and the record was 1 |
| `(False, True, False, True)` | "Sticky Memory (Ignore actions)" | `0101` | flips at every interaction |
| `(True, True, False, False)` | "Inverted Tit-for-tat Ledger" | `0011` | the opposite of what the donor did |
| `(True, False, True, True)` | "Inverted Grim Ledger" | `1000` | 1 only if the donor cooperates and the record was 1 |

- "Sticky Memory" is attached to the rule that flips the record at every interaction. The rule that never changes the record, `1010`, is the legacy tuple `(True, False, True, False)`, which has no entry.
- "Grim Ledger" is attached to `0111`. The rule in which a record that has dropped to 0 never returns to 1, `1000` (SPEC §6.1), is the tuple labelled "Inverted Grim Ledger".
- The four labels whose tuples are unaffected by the order of the second input (`1100`, `0000`, `1111` and `0011`) are attached to the rules their names describe.

History, from `git diff main SSC26`:

- On `main`, `DR_mode2.py` attached its grim label to `(False, True, True, True)` and its sticky label to `(False, True, False, True)`. Both match the comment's row order, not the code's.
- On `main`, `DR_mode3.py` attached the same two labels to `(True, False, True, True)` and `(True, False, True, False)`. By the translation above these are `1000` and `1010`: the rules the names describe.
- The `SSC26` commit replaced the `DR_mode3.py` dictionary with the one now in `DR_mode2.py`, and changed the grim tuple in both to `(False, True, False, False)`.

**2. "The DR control rule in modes 2–3 departs from the original model." — Confirmed that the rules differ; which is the original could not be determined.**

- Mode 1 fixed update (`DR_mode1.py:90-101`): on defection the donor is added to the partner's `memory`, and the partner is removed from the donor's own `memory` if present. On cooperation nothing changes.
- Modes 2–3 control update (`DR_mode2.py:90-97`; `DR_mode3.py:127-133`): on defection the donor is added to the partner's `memory`; on cooperation it is removed.
- In SPEC terms the modes 2–3 control is `1100`. The mode 1 update is `1000` on the partner's record of the donor, plus a second change in the same interaction: a donor that defects has its own record of that partner set back to 1.
- The comment at `DR_mode2.py:91` calls the modes 2–3 control "The strict pseudocode implementation". No pseudocode is in the repository, so which of the two rules is "the original model" cannot be read from it. SPEC §6.1's base rule, `1100`, is the modes 2–3 control and not the mode 1 update.
- The mode 1 update cannot be written as any of the 16 rules of §14.1, because it changes two records at once. It was therefore outside the space the legacy mode 2 and mode 3 searches could reach.

**3. "The mode-3 mirror labels are wrong." — Confirmed, on the reading given here.**

SPEC §17 does not say in what way they are wrong. The reading used: a "mirror" is the reversed action rule (`01`) paired with an assessment rule such that the pair behaves as the pair (`10`, f) does with the stored value's meaning flipped. For that, the mirror g must satisfy g(action, value) = NOT f(action, NOT value), and stored values must start at 0, not 1.

IR (`classify_joint_ir`, `IR_mode3.py:274-294`; "Inverted" entries at `IR_mode3.py:265-268` and `IR_mode2.py:219-222`). The legacy inputs and row order here are the same as SPEC §14.2, so tuples read directly as bit strings.

| Rule f | Mirror g, derived by hand | Tuple the legacy code uses as its mirror | Match |
| --- | --- | --- | --- |
| `1100` | `0011` | `0011` (`IR_mode3.py:290`) | yes |
| `1101` | `0001` | `0010` (`IR_mode3.py:291`) | no |
| `1001` | `1001` (its own mirror) | `0110` (`IR_mode3.py:292`) | no |
| `1000` | `1011` | `0111` ("Inverted Shunning", `IR_mode3.py:268`) | no |

The legacy "Inverted" tuples negate the output only. They do not also flip the meaning of the standing the rule reads, so they are correct only for `1100`, which ignores that input.

DR (`classify_joint_dr`, `DR_mode3.py:274-292`), using the translation of item 1. The legacy action tuple `(False, True)` is `10` and `(True, False)` is `01`.

| Legacy line | Action rule | Assessment rule (SPEC string) | Legacy joint label | Check |
| --- | --- | --- | --- | --- |
| `DR_mode3.py:284` | `10` | `1100` | "Tit-for-Tat" | agrees with SPEC §16 |
| `DR_mode3.py:285` | `10` | `0111` | "Grim Trigger" | SPEC §16 gives that name to `10` with `1000` |
| `DR_mode3.py:289` | `01` | `0011` | "Tit-for-Tat" | `0011` is the mirror of `1100` |
| `DR_mode3.py:290` | `01` | `1000` | "Grim Trigger" | the mirror of `1000` is `1011` |

- The pair (`10`, `1000`) falls through to "Unclassified Joint Strategy" (`DR_mode3.py:292`).
- In both models every stored value starts in the state that favours cooperating (`DR_mode3.py:79`; `IR_mode3.py:80`). A mirrored pair therefore starts by defecting where its counterpart starts by cooperating, so even a correctly derived mirror tuple does not behave identically from the first round.

**4. "Tree depth and tournament size were typed into the scripts rather than read from the config." — Confirmed.**

- Depth: `DR_mode1.py:32`, `IR_mode1.py:31`, `DR_mode2.py:31`, `IR_mode2.py:31`, `DR_mode3.py:39-40`, `IR_mode3.py:40-41` (`min_=1, max_=3`).
- Tournament size: `DR_mode1.py:39`, `IR_mode1.py:38`, `DR_mode2.py:38`, `IR_mode2.py:38`, `DR_mode3.py:69`, `IR_mode3.py:70` (`tournsize=3`).
- The comment at `DR_mode1.py:31` says the depths "could also be pulled from config".
- A search of every `.py` file in `legacy/` finds these keys in `config.py` only (`config.py:22-23`, `:31`, `:33-36`): `EXECUTION_ERROR_RATE`, `PERCEPTION_ERROR_RATE`, `TOURNAMENT_SIZE`, `INIT_METHOD`, `TREE_MIN_DEPTH`, `TREE_MAX_DEPTH`, `STATIC_DEPTH_LIMIT`. The search confirms the count of seven reported in phase 1.
- `INITIAL_ENDOWMENT` and `ENDOWMENT_FRACTION` (`config.py:18-19`) are read only under `under_development/`, not by the six model scripts.

### 2.2 Further mismatches, not listed in SPEC §17

These were found in this phase. They are kept apart from the confirmed list above. Each is a difference between the legacy code and SPEC.md, or within the legacy code; none is a statement about what the legacy runs showed.

| # | Mismatch | File and line | Why it matters |
| --- | --- | --- | --- |
| M1 | The legacy IR update is not `1101`. When a donor defects against a partner with standing 0, its standing is left as it was; `1101` sets it to 1. The legacy rule depends on the donor's own previous standing, which is the third input of §14.4 item 4, and is none of the 16 rules of §14.2. | `IR_mode1.py:88-96`; control in `IR_mode2.py:90-96`, `IR_mode3.py:128-134` | SPEC §6.2's base rule differs from what the legacy code ran. The legacy label "Simple Standing" (`IR_mode2.py:215`) is attached to `1101`, which is not the legacy control. The control was outside the space the mode 2 and 3 searches could reach. |
| M2 | The legacy DR mode 1 update is not `1100` (§2.1, item 2). | `DR_mode1.py:90-101` | SPEC §6.1's base rule differs from what `DR_mode1.py` ran. |
| M3 | The DR input has the opposite polarity to SPEC's: 1 means the partner is in `memory`. | `DR_mode1.py:60` | Legacy DR tuples and SPEC bit strings are not comparable without the translation in §2.1. |
| M4 | The sweep grid varies `TREE_MAX_DEPTH` over two values, and nothing reads it. Runs that differ only in that key differ in their seed and in nothing else. The R scripts group and colour by it. | `config.py:55`; `visualizations.R:146`, `:156`, `:196`, `:294`, `:390`; `mode3_visualizations.R:141` | A legacy chart that compares the two values compares seeds. |
| M5 | Classification runs on the SymPy-simplified text, and a row that fails to evaluate counts as 0. | `utilities.py:25-51` | §5 requires the compiled expression. |
| M6 | The constants are five values, not {0, 1}. | `DR_mode1.py:16` | §9. The set of expressions searched differs. |
| M7 | A size penalty and elitism are both applied; no depth limit is. | `DR_mode1.py:134-135`, `:192-193` | §9 steps 6–7; §13 "Size control", "Elitism". |
| M8 | A rule that is not changed keeps a score drawn in an earlier generation, and the copy of the best-so-far rule carries its old score. Scores in one generation therefore come from different generations' random draws. | `DR_mode1.py:184-187`, `:193` | §9 step 2. |
| M9 | The expression columns and the score columns of a CSV row can describe different rules. | `DR_mode1.py:205`, `:244-251` | The row cannot be read as "this rule earned this score". |
| M10 | The column named `iGSS_Max_Fitness` is not a maximum. It is the untaxed payoff of the rule whose **taxed** score is highest in the last generation. | `DR_mode1.py:195-196`, `:247` | The R box plot uses this column (`mode3_visualizations.R:63-64`). |
| M11 | Play-throughs, the control value and DEAP's operators share one random stream. | `DR_mode1.py:151`, `:159`, `:80`, `:85` | §9 "Random numbers". Any change to the play-through shifts the evolutionary stream. |
| M12 | The number of iGSS agents and the size of the rule population are separate parameters, and the grid varies only the second. | `config.py:12`, `:26`, `:53` | §3 uses one symbol, P, for both. |
| M13 | `DR_mode1.py` as committed performs one test run, not the batch. | `DR_mode1.py:225-258` | Explains why its CSV differs in shape from the other five (inventory §8, item 2). |
| M14 | In `IR_mode3.py`, a rule pair with action `10` and the always-0 assessment rule is labelled "Unconditional Defector", and in `DR_mode3.py` the corresponding pair carries a "(?)". With stored values starting at 1, such a pair cooperates with each partner until that partner's stored value is first written. | `IR_mode3.py:282`; `DR_mode3.py:283` | The joint labels describe the long-run behaviour only. |
| M15 | The R scripts read a different file from the one their charts need. | `mode3_visualizations.R:7` reads `results_IR_Mode2.csv` and then uses `Joint_Strategy` (`:20`); `visualizations.R:5` reads `results_DR_Mode1.csv` and uses `Joint_Strategy` at `:246` | Peripheral. Recorded for §4. |
| M16 | `filter()` is given `=` where `==` is needed, and `TARGET_STRATEGY` is spelt differently from the label the Python scripts write. | `mode3_visualizations.R:94-95`; `visualizations.R:132` against `DR_mode1.py:217` | Peripheral. Neither script was run, so the effect is not confirmed. |

**Finding for SPEC §17.** SPEC §17 says the findings of the inventory are added there. Proposed text, for the authors:

> *Proposed addition to SPEC §17 (not applied):*
>
> Further differences found in phase 2 of the rebuild brief (details in `docs/rebuild_plan.md` §2.2):
>
> - the legacy IR update leaves a donor's standing unchanged when it refuses a partner with standing 0; it is not `1101`;
> - the legacy DR mode 1 update is `1000` plus a reset of the donor's own record when it defects; it is not `1100`;
> - the legacy DR input is coded with 1 = the partner defected, the opposite of this file's coding;
> - the legacy sweep grid varied `TREE_MAX_DEPTH`, which no script read;
> - legacy scores within one generation were drawn in different generations, and the expression and score columns of a CSV row can describe different rules.

---

## 3. Questions for the decision register

Every row below is a proposal for SPEC §13. None is resolved here, and no code that depends on one may be written until an author confirms it (CLAUDE.md rule 1). "Proposed choice" is offered only so that nothing is left blank.

### 3.1 Rows already in §13 that need an author's attention first

| Topic (as in §13) | Present status | Why it comes first |
| --- | --- | --- |
| Direction of execution error | open | §4 step 5 forbids writing that step until it is decided. It blocks `episode.py`, and with it everything after `rules.py`. |
| Execution error on background agents | open | As above. It also fixes the e_action test in §11. |
| Seed scheme | proposed | The rule is stated but not the formula (row N6). The hand-checked trace and every reproducibility test depend on it. |
| Tooling | proposed | No versions are pinned, and `pytest` is not installed on the machine used. The machine has Python 3.14 and DEAP 1.4; whether those are the versions to pin is for the authors (row N19). |
| Start values; Update timing; Score unit | proposed | All three are built into `episode.py` and its hand-checked trace. A later change means redoing the trace. |
| DR assessment; IR assessment | proposed | §2.2 M1 and M2 show that neither base rule is what the legacy mode 1 scripts ran. The authors may want to confirm `1100` and `1101` knowing this. |
| Elitism; Size control | proposed | The legacy code did the opposite on both (§2.2 M7). Confirming them is confirming a deliberate departure. |
| Aggregate scoring seed | proposed | Fixes one of the two `scoring.py` tests in §11. |

### 3.2 New rows proposed

| # | Topic | Proposed choice | Alternatives | Why it matters |
| --- | --- | --- | --- | --- |
| N1 | Form of the constants | Two fixed terminals, 0 and 1, beside the input: three terminals, each equally likely to be drawn. | One ephemeral constant drawing from {0, 1}, as the legacy code did with its five values: the input is then drawn half the time and each constant a quarter of the time. | It changes how often the input appears in random expressions, and so the make-up of the first generation and of every mutation. |
| N2 | Generator for a mutation's new sub-expression | DEAP `genHalfAndHalf` with height 0–2. | `genFull` or `genGrow` with height 0–2. | §9 gives the depth but not the method. The method sets the size distribution of new material. |
| N3 | Meaning of "depth" | DEAP tree height: a lone terminal has height 0; "depth 1–3" is `min_=1, max_=3`; the limit of 17 is on height. | Counting levels from 1. | An off-by-one here changes the initial population and the size limit. |
| N4 | Tournament draws | DEAP `selTournament` as it is: three draws with replacement; among equal best scores the first drawn is kept, which is a uniformly random choice because the draw order is random. | Draws without replacement; an explicit random tie-break written in this code. | §9 says "three randomly picked rules" and "ties are broken at random". Whether DEAP's behaviour is accepted as meeting that, or new code is wanted, should be stated. Ties are frequent under aggregate scoring, where equal bit strings score equally. |
| N5 | Agent ids and list order | iGSS agents first, in population order (ids 0 to P−1), then unconditional cooperators, then unconditional defectors. | Any other fixed order. | The first shuffle of each episode acts on this list, so the order is part of what a seed reproduces. Under individual scoring it also fixes which agent carries expression i. |
| N6 | Episode seed formula | A fixed, documented function of (run seed, generation) that gives the same value on every machine and Python version, for example a `random.Random` seeded with the string `"<run seed>-<generation>"`. | Run seed + generation; a hash; NumPy `SeedSequence`. | §13 says "derived deterministically" without a formula. Run seed + generation would make run s at generation g+1 share an episode seed with run s+1 at generation g. |
| N7 | Run seeds across the sweep | The same list of 50 run seeds for every environment and both scoring options. | A distinct seed for every (environment, scoring option, repetition). | With shared seeds the two scoring options start from the same initial population, which makes them directly comparable; it also correlates runs across environments. |
| N8 | Baseline seeds | The baseline's episode seeds use the same formula as N6, so that an alone run can reproduce exactly an episode of aggregate scoring. Number of seeds per set: 50. | A separate seed list; a different number. | §8 says "many seeds" without a number. The `scoring.py` test "aggregate score equals alone-run payoff for the same seed" needs the two to share a seed-to-episode mapping. |
| N9 | In-pairs assignment | The first P/2 iGSS agents (by id) carry X and the rest carry Y; P must be even. | Alternating ids; random assignment per seed. | With N5 it fixes the list the shuffle acts on. It also decides what happens for an odd P. |
| N10 | A background type with no agents | Its mean payoff and cooperation rate are written as empty cells. | 0, as the legacy code did (`DR_mode1.py:110`). | N_C = 0 and N_D = 0 are both in the first sweep. A 0 cannot be told apart from a real mean of 0. |
| N11 | Generation numbering and "final generation" | Generations are numbered 1 to G. Generation 1 is the initial random population. Each generation is scored and logged, then bred, except generation G, which is scored and logged and not bred. "Final generation" is generation G. | Number from 0; breed G times and score a population G+1. | §9 lists scoring before breeding but does not say whether the last children are scored. The recovery criterion (§10) is defined on the final generation. |
| N12 | Tie for the most common bit string | A run counts towards the recovery rate only if `10` is strictly the most common; ties are reported separately. | Count a tie as recovery; split it. | §10 and §13 "Recovery criterion" do not cover ties. With P = 32 an even split is possible. |
| N13 | Output files | Names, columns, number format and write mode to be fixed in §10 before `run.py` is written. Proposed: each sweep writes into a new directory under `outputs/` and never appends to an existing file; floats are written unrounded; each row carries a run identifier unique within the sweep and the git commit of the code. | Appending to fixed file names, as the legacy code did. | §10 names two files and three items per row. Column names and formats are covered by CLAUDE.md rule 1. Appending is how legacy rows from different code versions came to share a file. |
| N14 | Where per-rule data go | A third CSV with one row per rule per generation (expression, simplified form, bit string, score), beside the per-generation and per-run files. | P expressions packed into one cell of the per-generation row; logging per-rule rows only every k generations. | §9 requires "every rule's expression, bit string and score" in every generation; §10 says each sweep writes two files with a row per run and a row per generation. The two do not fit together as written. The file is large: P × G rows per run. |
| N15 | Which donors' input reads are counted | iGSS donors only. Background donors do not read the input. | Count what every donor would have read. | §4 says "by each type of donor". The baseline's test of telling rules apart (§8) concerns iGSS agents. |
| N16 | Cooperation counts | Counted on the action **taken**, after execution error; the action decided is counted separately. In in-pairs baseline runs, counts and payoffs are kept separately for the carriers of X and of Y. | Taken only; decided only. | §4 says "cooperation counts" without saying which. The in-pairs run exists to compare X's carriers with Y's (§8). |
| N17 | Shape of `config.py` | Plain module-level names; `b` and `c` given directly; no fallback values anywhere in the code (a missing key is an error, as for `SCORING`); the first-sweep environments built from §6.3 by one function. | b/c ratio and c, as the legacy config has; dataclasses; a file format outside Python. | §11 says what `config.py` contains but not its form. Legacy fallbacks (`config.get(…, default)`) are how unstated values entered (§1.4, rows 31, 36, 37). |
| N18 | Constraints on parameters | P even and at least 2; N at least 2; checked at start-up with an error. | Allow an odd P (last rule not crossed). | Pairing for crossover (§9 step 4), in-pairs runs (N9) and matching (§4 step 2) each assume one of these. |
| N19 | Pinned versions | To be chosen by the authors: Python, DEAP, SymPy, pytest. | — | §13 "Tooling" proposes pinning without versions. DEAP's operators define parts of the model (N2, N4), so the DEAP version is part of the specification. |
| N20 | Order of random draws in one interaction | Fixed in §4, for example: partner draw, then one error draw for every donor in every interaction, whether or not it can change the action. | Draw for error only when it can matter (only when e_action > 0, or only on an intended cooperation under one-way error). | With an unconditional draw, runs that differ only in e_action share their matching sequence. The hand-checked trace depends on the order. Depends on the two open execution-error rows. |
| N21 | SymPy form in the logs | Computed once per distinct expression text and cached; if SymPy fails, the cell records the failure and classification is unaffected. | Simplify every rule in every generation. | §5 says the simplification is logged alongside. It is by far the slowest step if repeated P × G times per run. |
| N22 | Baseline rules as callables | In the baseline, each of the four rules is the named predicate from `rules.py` used directly as the agent's `action_rule`, not a DEAP expression. | Hand-written DEAP expressions compiled for the purpose. | §4 types `action_rule` as a compiled expression; §8 says rules are "assigned by hand". Which object is assigned is not stated. |
| N23 | Language and format of figures | Stays open until §10 has a Figures subsection (§4.2 below). The brief's phase 4 keeps R unless the authors decide otherwise. | Python (Matplotlib), which would add a dependency. | Needed before phase 4, not before mode 1. |

---

## 4. Peripheral code to port later

Nothing here is to be ported until SPEC §10 says what it must produce (CLAUDE.md rule 4).

### 4.1 File by file

| Legacy code | Worth keeping? | What adapting it would take | What it could serve |
| --- | --- | --- | --- |
| `utilities.py:9-23`, `simplify_rule` | Yes, for reading only. | Keep it as a display helper outside classification. Replace the text `eval` with a walk of the DEAP tree, or keep `eval` with the empty built-ins it already has; record a failure in the cell (row N21). It must not feed `evaluate_truth_table`. | The simplified-expression column that §5 requires, and the table of expressions per bit string (§10 "Main output"). |
| `utilities.py:25-51`, `evaluate_truth_table` | No. | — | Replaced by classification of the compiled expression in `rules.py` (§5). Not to be ported (rule 4: model core). |
| `utilities.py:53-61`, `log_results_to_csv` | No. | — | Its append-and-first-row-header behaviour is the problem in §1.6 row 49. A writer with a fixed column list is a few lines and is better written new, once row N13 is decided. |
| `utilities.py:63-121`, `hierarchy_pos` and `plot_deap_trees` | Possibly. | Works on any DEAP tree and needs no change to the drawing. It needs `networkx` and `matplotlib` added to the requirements, an expression text read back from the CSV into a tree, and a rule for which expressions to draw. | A drawing of one example expression per bit string. Only if §10 asks for it. |
| `utilities.py:123-199`, the terminal report in `generate_dashboard` | No. | — | It reports one returned rule and one control value, neither of which exists in the new design. |
| `utilities.py:204-244`, the per-run line chart in `generate_dashboard` | The idea, not the code. | The new per-generation file holds the mean payoff of iGSS agents and of each background type. The chart would read that file and no longer use a control line. About thirty lines; rewriting in the plotting language chosen (row N23) is simpler than adapting. | Proposed figure F5 below. |
| `visualizations.R:10-44`, `chart1_overview` | Yes, as a layout. | Reads one row per run and counts runs by a label, in panels of defectors by cooperators, with rounds and b/c on the horizontal axis. To adapt: read the new per-run file; replace `Identified_Strategy` with the most common bit string of the final generation; replace `NUM_ROUNDS` on the axis with e_action; add scoring option as a panel dimension. | Proposed figure F2 below. |
| `visualizations.R:51-124`, `chart1A_baseline` and `chart1B_simple` | Partly. | Same data as above in other arrangements. The filter on `NUM_IGSS == 10` goes. | Variants of F2, if wanted. |
| `visualizations.R:132-233`, `chart2_*` | No. | Group by `POP_SIZE`, `TREE_MAX_DEPTH` and `PARSIMONY_TAX`. None of the three is a factor in SPEC §6.3, and one had no effect (§2.2 M4). | Nothing in the first sweep. A recovery-rate bar chart can be had from F2's data. |
| `visualizations.R:239-420` and `mode3_visualizations.R:13-171`, the pie charts | No. | Built on `Joint_Strategy`, a mode 3 label that §2.1 item 3 shows to be unreliable, and on the three search parameters above. | Nothing in mode 1. |
| `mode3_visualizations.R:57-83`, `fitness_by_strategy_plot` | The idea, for the baseline. | A box plot of a payoff by label. To adapt: read the baseline output; one box per bit string; payoff per round. | Proposed figure F3 below. |

### 4.2 Proposed text for a "Figures" subsection of SPEC §10

A suggestion for the authors. It describes form only: what is drawn, from which file and columns, and how it is grouped. Column names are placeholders until row N13 is decided.

> *Proposed text for SPEC §10 (not applied):*
>
> **Figures.** Every figure is drawn from the CSV files of one sweep or one baseline, by a script that takes the directory as its argument and writes image files beside the data. No figure is drawn from data held in memory. Rules are shown by bit string, with `10` marked; literature names appear only as labels taken from §16.
>
> | Figure | Shows | Data | Grouping |
> | --- | --- | --- | --- |
> | F1. Composition over time | For each generation, the share of the P rules that have each bit string, averaged over seeds, as stacked areas. | Per-generation file: generation, count of each bit string. | One panel per environment; one figure per model and scoring option. |
> | F2. Final composition | The share of runs in which each bit string is the most common in the final generation, as stacked bars. The share for `10` is the recovery rate. | Per-run file: most common bit string in the final generation. | Horizontal axis: b/c by e_action. Panels: N_D by N_C. One figure per model, with the two scoring options side by side. |
> | F3. Baseline, rules alone | The payoff per round of the iGSS agents when all carry one rule, as one box per rule over seeds, with the mean payoff of each background type marked. | Baseline file, alone runs. | One panel per environment; one figure per model. |
> | F4. Baseline, rules in pairs | For each pair of rules X and Y, the difference between the mean payoff per round of X's carriers and of Y's carriers, as a 4 × 4 grid of cells. | Baseline file, in-pairs runs. | One grid per environment; one figure per model. |
> | F5. One run | By generation: the count of each bit string, and the mean payoff per round of the iGSS agents and of each background type. | Per-generation file, one run. | One figure per run, drawn on request, for checking. |
>
> | Table | Shows | Data |
> | --- | --- | --- |
> | T1. Search against baseline | For each environment and scoring option: which rules the baseline cannot tell apart (from the input values read), which rule the baseline ranks first, and the recovery rate. The rows follow the four readings in the table above. | Baseline file and per-run file. |
> | T2. Expressions behind each bit string | For each bit string, the distinct simplified expressions found in final generations and how many rules had each. | Per-rule output of the final generation. |
>
> Not specified: colours, sizes and file types, which are left to the plotting script.

If the authors adopt F1 and T2, row N14 (where per-rule data go) has to be decided first, because T2 needs the expressions of the final generation.

---

## 5. Build sequence

The order of work in SPEC §11 is confirmed for mode 1: `rules.py`, `episode.py`, `baseline.py`, `scoring.py`, `search.py`, then `run.py` and the first sweep. Each depends only on those before it. Four amendments are proposed for the authors.

**Amendment 1: a step 0 for tooling.** `pytest` is not installed on the machine used, and no `requirements.txt` exists. Proposed: before `rules.py`, the authors fix the pinned versions (row N19) and `requirements.txt` is committed, so that the first tests can be shown failing and then passing as the brief's phase 3 requires.

**Amendment 2: where `config.py` sits.** SPEC §11 lists `config.py` in the table of files but not in the order of work. The brief's phase 3 puts it sixth, after `search.py`. The episode, baseline and scoring all take an environment as an argument, so they need its definition earlier. Two ways to resolve this, for the authors to choose:

- (a) the environment is defined in `episode.py`, as a plain data object beside `Agent`, and `config.py` is written sixth holding only values: the first-sweep environments, the search parameters and `SCORING`;
- (b) `config.py` moves to second place, before `episode.py`.

Under (a) the brief's order stands and §11 gains one line. Either way, SPEC §11 needs to say where the environment's definition lives.

> *Proposed addition to the order of work in SPEC §11 (not applied), under option (a):*
>
> 5a. **`config.py`.** Tests: a run with `SCORING` unset fails; every name defined in `config.py` is read somewhere in `igss/`.

**Amendment 3: what blocks what.**

- `rules.py` depends on no open decision except rows N1 and N3 (for building the expression set used in its tests) and can start as soon as step 0 is done.
- `episode.py` cannot start until the two execution-error rows are decided (§4 step 5 forbids it), together with rows N5, N6 and N20, which fix its hand-checked trace.
- `baseline.py` needs rows N8, N9, N10, N15, N16 and N22.
- `scoring.py` needs the "Aggregate scoring seed" and "Score unit" rows confirmed.
- `search.py` needs rows N2, N4, N11 and N14.
- `run.py` needs rows N7, N12, N13 and N17.

**Amendment 4: the baseline's output.** §8 specifies the baseline's runs and §11 gives `baseline.py` one test, but §10 names no file for the baseline's output. Figures F3, F4 and table T1 above need one. Proposed: §10 states the baseline's file when row N13 is decided.

**Tests that could be added.** These are suggestions only. None is to be written unless the authors approve it and it is added to §11.

- `rules.py`:
  - the constant 0 gives `00`, so that all four bit strings are covered;
  - a two-input expression is classified in the row order of §5 (§14.4 item 4 asks that classification already accept more inputs);
  - an expression whose output is negative on a row gives 0 on that row.
- `episode.py`:
  - in every round each agent is donor exactly once and is never its own partner;
  - the sum of all payoffs equals (b − c) × the number of cooperative actions taken;
  - IR under `1101`: a donor that refuses a partner with standing 0 ends with standing 1 (the case in which the legacy code differs, §2.2 M1);
  - DR under `1100`: after a defection followed by a cooperation by the same donor towards the same partner, the partner's record of the donor is 1 (the case in which the legacy mode 1 code differs, §2.2 M2);
  - the logged input counts add up to the number of interactions in which an iGSS agent was donor.
- `baseline.py`: an in-pairs run of X with X gives the same payoffs as X alone, for the same seed.
- `scoring.py`: under individual scoring, the scores equal the payoffs per round of the corresponding agents in one call to `play_episode` with the same seed.
- `search.py`:
  - changing R leaves the sequence of expressions in generation 1 unchanged (episodes do not draw from DEAP's generator);
  - no expression in any generation is higher than 17;
  - every rule is scored in every generation.
- `config.py`: as in amendment 2.

---

## 6. Assumptions made in writing this file that SPEC.md does not state

1. **Scope of "what mode 1 will need".** Taken to include the baseline, the output files and the command-line entry point, since §11's build plan for mode 1 ends with `run.py` and the first sweep.
2. **Scope of the comparison.** Limited to the six model scripts, `config.py`, `utilities.py` and the two R scripts. `under_development/` and the zip archive were left out, on the ground that §17 names the six scripts and v1.x covers DR and IR only.
3. **Translation of the legacy DR coding.** The SPEC record was taken to be 1 exactly when the donor is not in the partner's `memory`, and `previous_record` to be 1 exactly when `DonorInMemory` is 0. All SPEC bit strings given for legacy DR tuples rest on this.
4. **Meaning of "mirror" in §17.** Taken to be the reversed action rule paired with an "Inverted" assessment rule, and "wrong" to mean that the pair does not behave as its counterpart does. §17 does not define either.
5. **"The original model" in §17.** Not identified. The file reports that the mode 1 update and the modes 2–3 control differ, and does not say which is original.
6. **DEAP's behaviour.** Described from the source of DEAP 1.4 as installed on the machine used. It is assumed, not checked, that the version to be pinned behaves the same way.
7. **"Depth" in §9 and §13.** Read as DEAP tree height when comparing the legacy `min_=1, max_=3` with SPEC's "depth 1–3". This is also put to the authors as row N3.
8. **Status labels.** The three values "same", "contradicts" and "silent" are this file's own. "Contradicts" is used wherever SPEC.md decides a point and the legacy code differs, including points whose §13 status is only "proposed".
9. **Proposed choices in §3.** Each is a placeholder so that no row is blank. None was treated as decided anywhere in this file, with one exception of presentation: the proposed Figures text (§4.2) uses "final generation" and "most common bit string" as §10 already does, before rows N11 and N12 are settled.
10. **History.** `git diff main SSC26` was read at the files' paths before the phase 0 move. It is assumed that `legacy/` holds the `SSC26` versions unchanged, as phase 0 reported.
11. **Hand derivations.** The bit strings in §2.1 and the mirror rules were worked out by hand for this file and have not been checked by code, since no code was to be written in this phase.

---

## 7. Author review, first pass (14 November 2026)

This section records the answers an author gave in review of §1 and §2. It is a record, not a change to SPEC.md: the SPEC changes that follow from it are applied separately, once the points under discussion are settled. Row numbers are those of §1 and §2.

### 7.1 Standing instructions given in the review

| Instruction | Effect on the working rules |
| --- | --- |
| Where the legacy code and SPEC.md disagree, SPEC.md is followed, and the author is told afterwards. | No further confirmation is needed for the "contradicts" rows of §1, except those listed in §7.3. |
| Where SPEC.md is silent on a DEAP matter, an informed choice is made, coded, and recorded for later review. | Replaces "stop and ask" (CLAUDE.md rule 1) for DEAP mechanics. Each such choice goes into SPEC §13 marked as chosen by the assistant and awaiting review. |
| Where SPEC.md is silent on an inconsequential point (the example given: `agent_id`), a reasonable choice is made, recorded, and reported; coding does not wait. | As above. |
| `config.py` and utilities: follow good practice, stay close to the legacy layout unless something is clearly better, keep the code readable and reproducible as research code. SPEC.md is revised for these after implementation, once the author confirms. | Rows N13 and N17 need no answer before coding. |
| Modes 2 and 3 are considered only when they are reached. What the legacy code inherited from the earlier agent-based model and its pseudocode is dropped. | §2.1 items 1–3 need no further action for mode 1. |
| Outputs are designed afresh, to be clean and coherent. A row never describes two different rules (rows 40, M9). | Decided in the output design; reported after coding. |

### 7.2 Answers by row

| Row | Answer |
| --- | --- |
| 1 (N1) | The constants 0 and 1 are two fixed terminals. |
| 4 (N3) | "Depth" is taken to be DEAP tree height. Recorded as an assumption; plausible alternatives are to be reported later for discussion. |
| 11 (N5) | Left to the assistant: choose, record, report. |
| 14, M3 | The opposite polarity of the legacy DR input is noted. DR and IR are each to be implemented in the way natural to its own literature, kept as coherent with each other as possible. SPEC.md is followed for now. To be looked at again once early output can be interpreted. |
| 16, 18, M1, M2 | The legacy DR "forgiveness after punishment" step and the legacy IR "unchanged" case came from an earlier agent-based model. Both are dropped. The base rules stay `1100` (DR) and `1101` (IR). No assessment rule that reads the donor's own previous standing is included at this stage. |
| 21 | c = 1 by convention in all runs; b is varied. The config holds b. |
| 22 | Execution error is implemented in the rebuild. Its direction and its scope remain to be stated (§7.3). |
| 28 (N10) | Nothing is logged for a background type with no agents, provided the output can still be analysed and charted across all environments together. Otherwise a mean payoff of 0 is logged. |
| 29 | A baseline exists for every possible rule. It need not be recalculated in every run. |
| 30 | Left to the assistant: decide and record for later consideration. |
| 31, 32 | Left to the assistant: decide, record, and report after coding. |
| 33 (N18) | P is always even. SPEC.md may be updated to say so. |
| 36 | Elitism: left to the assistant. Recorded for possible discussion once output exists. |
| 40 | What a run reports: left to the assistant, bearing in mind that statistical tests and statements such as "the rule most often found" in each scenario will be wanted. Recorded for possible discussion once output exists. |
| 44 | Labels: to be reviewed once output exists. |
| 48 (N6–N8) | Seeds: the assistant proposes and implements a seed architecture, justifies it, and adjusts SPEC.md to match. |

**Purpose, as restated by the author.** iGSS is to find canonical reciprocity strategies without the usual evolutionary-tournament set-up. What is to be explained is the evolved rule. Rules may be simple for now. They are to emerge through genetic programming in a simple implementation, with nothing built in that forces an outcome. The present work establishes the framework.

### 7.3 Points held for discussion before coding

1. **Scoring (rows 24–26).** The author regards the move to P copies, one episode and payoff per round as a key change and is not yet persuaded. To be discussed before implementation; the outcome is to be judged again once output exists.
2. **Size penalty (row 27).** The author wants a size penalty in the model, as a parameter that can be varied. SPEC §13 "Size control" at present proposes none.
3. **What can be varied in runs.** The author wants to vary the environment (the split between iGSS agents, unconditional cooperators and unconditional defectors), the game parameters (population, b, number of rounds, the "shadow of the future") and the search parameters (size penalty, depth and others). SPEC §6.3 fixes R, P and every search parameter. To be confirmed and specified.
4. **Execution error.** Direction (one-way or two-way) and scope (every donor, or iGSS agents only) are still open in SPEC §13, and §4 step 5 forbids code for that step until they are stated.
5. **§3 of this file** has not yet been reviewed.

**Outcome (15 November 2026).** The author accepted the four recommendations put to them, asked for a lighter specification process, and asked for mode 1 to be built. They are now in SPEC v1.2:

1. **Scoring.** Both options kept as specified. Two parameters added: E, the number of episodes averaged per scoring (1 in the base configuration), and N_I, the number of iGSS agents per episode, separate from P under aggregate scoring (equal to P in the base configuration).
2. **Size penalty.** `PARSIMONY_TAX` per node, subtracted from the selection score only, in units of payoff per round, 0 in the base configuration and varied as a search factor.
3. **What can be varied.** One base configuration and named sweeps; every parameter can be varied; R is the only control on the length of the game, with no random stopping.
4. **Execution error.** One-way, applied to every donor, in both models; `e_action` only.

The remaining rows of §3 were covered by the standing instructions of §7.1 and are recorded in SPEC §13 with status "assistant".

### 7.4 To be looked at again once output exists

- DR and IR input coding and how coherent the two models are (row 14).
- The scoring design (rows 24–26).
- Elitism (row 36) and what a run reports (row 40).
- Labels (row 44).
- The meaning of "depth" and its alternatives (row 4).
- The seed architecture (rows 30, 48).
- "Unnamed" strategies that behave exactly like a named one (modes 2 and 3; SPEC §10, "A caution on unnamed"). Noted by an author on 15 November 2026 for the analysis stage: grouping by name alone understates how often a named behaviour is found.
- **Individual scoring as a whole is set aside** (author, 15 November 2026). It is not the present focus. It is to be revisited after the aggregate-scoring runs, once at least a base version of the model is running. The next item belongs to it.
- Whose assessment rule is applied under individual scoring (SPEC §14): the partner's, with the observer and private-assessment set-ups of the IR literature as alternatives.
