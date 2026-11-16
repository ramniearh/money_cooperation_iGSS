# iGSS — specification for the DR and IR models

This file is the single source of truth for rebuilding the iGSS code from scratch. It is limited to direct reciprocity (DR) and indirect reciprocity (IR).

The code must match this file, and any mismatch is a bug. A change in behaviour is made here first, or in the same commit as the code, and is recorded in the change log (§18).

- **Status:** v1.8, written alongside the first build of modes 1, 2 and 3 and the first exploratory sweeps.
  - Modes 1, 2 and 3 are specified and built. The main focus for now is aggregate scoring; individual scoring is built and kept as a separate question.
  - No results appear here.
- **Every choice is listed** in the decision register (§13). A choice not written here must not be made silently in code.
  - A choice that can change which rules evolve is agreed with an author before it is coded.
  - A choice about DEAP mechanics, or one that cannot change which rules evolve, may be made by the assistant. It is then recorded in §13 with status "assistant" and reported to the authors for review.
- **Terms are mechanical.** Literature names appear only in the cross-reference table (§16). Interpretive vocabulary (for example "institution") is not used at this stage.
- **Sections are numbered** so that commits, issues and comments can cite them (for example "SPEC §4").

*AI-generated: compiled by Claude (Anthropic) from working conversations with the authors, for checking and amendment by them.*

---

## 1. Question and scope

**Question.** Given a specified environment, can iGSS reliably find the action rules the literature calls canonical reciprocity strategies?

**Scope of version 1.x.**

- DR and IR are run as separate models. A combined model comes later.
- Three modes: the action rule evolves (mode 1), the assessment rule evolves (mode 2), or both (mode 3).
- Each action rule has one input. There are no junk inputs.
- Execution error is the only error.
- Rules are scored in two ways, run side by side and compared:
  - **individual:** each rule is scored by the payoff of the agent carrying it;
  - **aggregate:** each rule is scored by the mean payoff of a group in which every iGSS agent carries it.

The two scoring options are kept on purpose. They belong to a larger question this framework is meant to address later: at what level selection operates in social evolution, and how aggregate outcomes relate to individual ones. Each option is built as its own, explicitly named piece of code, never as a silent default.

## 2. The model in plain words

**The game.** Agents meet in random pairs. In each meeting one agent, the donor, decides whether to help the other, its partner. Helping costs the donor c and gives the partner b, with b larger than c. Everyone is better off if everyone helps, but each donor saves c by not helping.

**Who is there.**

- P iGSS agents, whose behaviour is being searched for.
- Fixed background agents: N_C who always help and N_D who never help.

**What an iGSS agent knows about its partner.** One yes/no fact:

- **DR:** did this partner help me the last time it was my donor?
- **IR:** what is this partner's standing, 0 or 1? Standings are public. A fixed rule sets them, for example "you get standing 0 if you refuse someone with standing 1".

**What a rule is.** A rule is a small arithmetic formula that turns that one fact into "help" or "don't help". There are only four possible behaviours:

- always help;
- always refuse;
- help if the fact is 1;
- help if the fact is 0.

The search writes formulas, and the code reads off which behaviour each formula produces.

**An episode.** All agents play R rounds. In each round every agent acts once as donor, with a randomly chosen partner. Sometimes an intended help fails (execution error). At the end, each agent has a total payoff.

**Scoring rules.** Two options, compared side by side:

- **Individual:** all P rules share one episode, one rule per iGSS agent. A rule's score is its own agent's payoff.
- **Aggregate:** each rule gets an episode of its own in which all P iGSS agents carry it. A rule's score is the mean payoff of those P agents.

**Breeding: one generation, worked through with P = 4.**

1. **Scoring.** Rules r1–r4 are scored: say 10, 4, 8 and 2.
2. **Selection.** This example uses a tournament of size 2, repeated four times; the model itself uses size 3.
   - Each time, two rules are picked at random and the higher-scoring one is kept.
   - Say the picks are (r2, r3) → r3; (r1, r4) → r1; (r1, r2) → r1; (r3, r4) → r3.
   - The parents are r3, r1, r1 and r3. Good rules get more copies; poor ones tend to vanish.
3. **Crossover.**
   - The parents are paired: (r3, r1) and (r1, r3).
   - With probability 0.5, each pair swaps a randomly chosen piece of formula, so material moves between rules.
4. **Mutation.** With probability 0.2, each child has a random piece of its formula replaced by a new random piece.
5. **Next generation.** The four resulting rules form the next generation.
   - Under individual scoring they go to four fresh agents in a new episode.
   - Under aggregate scoring each gets a new episode of its own.

**Repeat** for G generations. At every generation, record how many rules produce each of the four behaviours.

**Before any search: the baseline.** Each of the four behaviours is run by hand, with every iGSS agent using it. This shows in advance how each performs and which ones can be told apart.

**The result.** For each environment, over many random seeds: how often each behaviour comes to dominate, compared with what the baseline predicts. The question is whether "help if the fact is 1" (the behaviour the literature calls canonical) dominates where the baseline says it should.

## 3. Vocabulary and symbols

| Term | Meaning |
| --- | --- |
| donor | The agent deciding whether to pay cost c so that its partner receives benefit b. |
| partner | The agent the donor was randomly matched with in the current interaction. Interactions are directed: A as donor with partner B is a different event from B as donor with partner A. |
| action rule | Run by the donor before acting. Maps the donor's input bit to cooperate (1) or defect (0). Evolved in mode 1. |
| assessment rule | Run by the partner after the action. Maps the donor's action plus one other bit to a stored value: a **record** (DR) or a **standing** (IR). Fixed in mode 1. |
| strategy | An action rule together with the assessment rule it runs with. In mode 1 the assessment rule is fixed, so each action rule corresponds to exactly one strategy. |
| mode | Which rule is evolved: 1 = action rule, 2 = assessment rule, 3 = both. |
| iGSS agent | An agent whose action rule comes from the evolving population of rules. |
| background agents | Agents with fixed behaviour: unconditional cooperators and unconditional defectors. |
| execution error | The probability e_action that an intended cooperation fails, so that the donor defects instead. It is one-way (a decision to defect is never changed) and applies to every donor, background agents included. |
| round | Every agent acts once as donor. |
| episode | One play-through of R rounds from a fresh start. "Episode" is used rather than "instance", because in code an instance is any object created from a class. |
| episode state | Records, standings and payoffs. Reset at the start of every episode. |
| generation | One cycle of evolution: score every rule, then produce the next set of rules. The same thing as a generation in DEAP. |
| run | G generations for one environment, one scoring option and one random seed. |
| environment | The parameters fixed for a run: model (DR or IR), assessment rule, N_C, N_D, b, c, e_action and R. |
| scoring option | How a rule's score is computed: individual or aggregate. |
| canonical rule | The action rule the literature names for that model, fixed in advance by its bit string. It is marked in the results and changes nothing that is run or measured. |

All inputs are coded so that 1 is the event that would normally favour cooperating. This is a coding convention, not an evaluation: "standing = 1" means only what the assessment rule makes it mean.

| Symbol | Meaning |
| --- | --- |
| P | Size of the evolving population of rules. Always even. |
| N_I | Number of iGSS agents in an episode. Under individual scoring N_I = P, because each rule has one agent. Under aggregate scoring N_I is a parameter of its own; it equals P in the base configuration. Always even. |
| N_C | Number of background unconditional cooperators. |
| N_D | Number of background unconditional defectors. |
| N | Total agents in an episode: N_I + N_C + N_D. |
| G | Generations per run. |
| R | Rounds per episode. |
| E | Episodes played per scoring, and averaged. 1 in the base configuration. |
| b, c | Benefit to the partner and cost to the donor of cooperating. By convention c = 1 in all runs, and b is varied. |
| e_action | Execution error. |
| e_assessment | Assessment error. Not used in version 1.x. |

## 4. The episode

An episode is played by one plain Python function. Given a list of agents and an environment, it plays R rounds and returns:

- each agent's payoff;
- cooperation counts;
- how often each input value occurred.

The baseline and both scoring options call this same function.

**Agents** are plain Python objects, not Mesa agents:

```python
@dataclass
class Agent:
    agent_id: int
    kind: str                      # "igss", "always_cooperate", "always_defect"
    action_rule: Callable | None   # compiled expression; None for background agents
    payoff: float = 0.0
    records: dict[int, int] = field(default_factory=dict)  # DR: partner_id -> 0/1
    standing: int = 1                                       # IR
    assessment_rule: Callable | None = None                 # modes 2-3; None = the base rule
```

**Agent list.** The list given to an episode holds the iGSS agents first, then the unconditional cooperators, then the unconditional defectors. `agent_id` is the position in that list, from 0.

**Start of an episode.**

- Payoffs start at 0.
- DR: every record starts at 1. A partner never met before is treated as having cooperated.
- IR: every standing starts at 1.

**One round.**

1. Shuffle a copy of the agent list into a random order. The copy starts in `agent_id` order every round.
2. Each agent in turn acts as donor, matched with a partner drawn at random from the other agents (uniformly, from the others in `agent_id` order).
3. The donor reads its input: its own record of the partner (DR), or the partner's standing (IR).
4. The donor decides. Background agents ignore the input; an iGSS agent applies its action rule.
5. Execution error: one random number is drawn in every interaction, whatever the decision. If the donor decided to cooperate and the number is below e_action, the action taken is defect. A decision to defect is never changed. This applies to every donor, background agents included.
   - Because the draw is made in every interaction, runs that differ only in e_action share the same sequence of matchings.
6. Payoffs: if the action taken is cooperate, the donor loses c and the partner gains b.
7. The partner applies its assessment rule to the action taken (the one fixed rule in mode 1; see §14 for modes 2 and 3). The result is stored at once, so later donors in the same round already see it.
   - DR: the partner updates its own record of the donor.
   - IR: the donor's public standing is set from the action taken and the partner's current standing (the value the donor read at step 3).

   This step applies to every donor, background agents included.

**Encounter rate.** DR works only if pairs meet again. A given donor is matched with a given partner about R/(N−1) times per episode. With N = 96 and R = 400 (the base configuration), that is about 4 meetings per ordered pair; with no background agents (N = 32), about 13.

**What every episode logs:**

- payoff per agent;
- cooperation counts (actions taken), per donor and by type of donor and partner;
- how often each input value (0 or 1) was read by each iGSS donor. Background donors do not read the input.

The last item shows whether two rules can be told apart (§8).

## 5. How to read the truth tables

A rule with yes/no inputs is fully described by its output for every combination of inputs. The **bit string** is that output column, read top to bottom in a fixed row order: **all-ones first, all-zeros last, first input most significant.**

| Inputs | Row order | Rows | Possible rules |
| --- | --- | --- | --- |
| 1 input X | X=1, X=0 | 2 | 4 |
| 2 inputs X, Y | (1,1), (1,0), (0,1), (0,0) | 4 | 16 |
| 3 inputs X, Y, Z | (1,1,1), (1,1,0), …, (0,0,0) | 8 | 256 |

Example: the action rule "cooperate if the partner cooperated with me last time" outputs 1 when the input is 1 and 0 when it is 0. Its bit string is `10`.

**From an evolved expression to a bit string.** The search produces arithmetic expressions such as `sub(1, mul(0, partner_standing))`.

1. DEAP **compiles** the expression into an ordinary Python function.
2. That function is called once on every input row, and an output above 0 counts as 1.
3. The results, read down the rows, form the bit string: here `11`.

This is exact and involves no parsing. A SymPy simplification of the expression (here `1`) is logged alongside it, for human reading only. It plays no part in classification.

Many different expressions give the same bit string. The bit string is what the agent does; the expression is how the search happened to write it. Results are grouped by bit string, and every expression is kept in the logs.

In the code, every named rule is defined as a predicate on named inputs. Its bit string is derived from the predicate, never typed by hand.

## 6. Mode 1 specification

In mode 1 the action rule evolves and the assessment rule is fixed, the same for every agent.

### 6.1 DR

**Input:** `partner_cooperated_with_me_last_time`. This is the donor's own record of what this partner did the last time the partner was donor and this agent was its partner.

**Fixed assessment rule:** the partner's record of the donor becomes whatever the donor just did.

- Inputs (`donor_cooperated`, `previous_record`); bit string `1100`.
- A variant environment uses `1000`: a record that has dropped to 0 never returns to 1.

**Action rules:**

| Bits | Cooperate if 1? | Cooperate if 0? | What it does |
| --- | --- | --- | --- |
| `11` | yes | yes | always cooperate |
| `10` | yes | no | **follow the record (canonical)** |
| `01` | no | yes | reverse the record |
| `00` | no | no | always defect |

### 6.2 IR

**Input:** `partner_standing`.

**Fixed assessment rule (base, `1101`):** after each interaction, the donor's standing becomes 0 only if it refused a partner whose standing was 1. Otherwise it becomes 1. The inputs are (`donor_cooperated`, `partner_standing`).

Variant environments use:

- `1100`: standing = whether the donor cooperated;
- `1001`: standing 1 for helping a 1 or refusing a 0;
- `1000`: standing 1 only for helping a 1.

The full list of 16 is in §14.

**Action rules:**

| Bits | Cooperate if 1? | Cooperate if 0? | What it does |
| --- | --- | --- | --- |
| `11` | yes | yes | always cooperate |
| `10` | yes | no | **follow the standing (canonical)** |
| `01` | no | yes | reverse the standing |
| `00` | no | no | always defect |

### 6.3 Environments for the first sweep

Every parameter of the model can be varied. `config.py` holds one base configuration and a set of named sweeps, each of which varies a few factors over the base (§11). The parameters fall into three groups:

- **environment:** N_I, N_C, N_D;
- **game:** b, R, e_action;
- **search:** P, G, E, tournament size, crossover and mutation probabilities, initial and mutation depths, depth limit, size penalty.

R is the only control on how long the game lasts. There is no random stopping: a rule cannot see the round number, so a fixed R creates no end-of-game effect. For DR, what matters is the number of meetings per ordered pair, about R/(N−1).

The first sweep is below. Each combination is one environment, run under both scoring options.

| Factor | Values |
| --- | --- |
| model | DR, IR |
| assessment rule | DR: `1100`; IR: `1101` (variants later) |
| N_C | 0, 32 |
| N_D | 0, 32 |
| b/c | 2, 5 (c = 1) |
| e_action | 0, 0.05 |
| R | 400 |

**Known collapse.** When no agent ever defects (N_D = 0, e_action = 0, and no defecting rule present), the input is always 1. "Follow the record" or "follow the standing" then behaves exactly like "always cooperate", and the two cannot be told apart. The baseline will show where else this happens.

## 7. Two ways of scoring rules

The evolutionary machinery (selection, crossover and mutation) is identical under both options. Only the scoring function differs.

|  | Individual scoring | Aggregate scoring |
| --- | --- | --- |
| Episodes per generation | E | E per rule |
| Who is in an episode | all P rules, one per iGSS agent, plus background | N_I copies of one rule, plus background |
| A rule's score | the payoff of the agent carrying it, per round, averaged over the E episodes | the mean payoff of the N_I iGSS agents, per round, averaged over the E episodes |
| What it rewards | doing well among other, different rules | a group in which everyone uses the rule doing well |
| Relation to the legacy code | new | the legacy design, which had 10 copies and three episodes of 100 rounds |

**How much play a score rests on.** With the base values (N_I = P = 32, R = 400, E = 1), an aggregate score rests on 12,800 donor decisions and an individual score on 400. Individual scores are therefore noisier. E is the parameter that reduces this noise.

**Same random seeds within a generation (aggregate scoring).** The E episodes of every rule in one generation use the same E seeds, so differences in score come from the rules, not from luck in matching. A side effect: rules with the same bit string get exactly the same score, so each distinct bit string needs simulating only once per generation.

**What the agents run.** In a scored episode an iGSS agent decides from its rule's bit string, read off the compiled expression as in §5. This is the same decision as calling the expression, and avoids evaluating deep expressions millions of times.

**Size penalty.** The score used for selection is the score above minus `PARSIMONY_TAX` × the number of nodes in the expression. The penalty is in units of payoff per round. It is 0 in the base configuration and is varied as a search factor. The score that is logged and reported is always the one without the penalty; the selection score is logged beside it. A penalty is a prior on which rules are found: `10`, `11` and `00` can each be written with one node, and `01` needs at least three.

**Why the two options can disagree.**

- With no defectors present, aggregate scoring rates "always cooperate" and "follow the record" equally and highly, because a group of either does well.
- Under individual scoring, an agent that defects among cooperators earns more than they do. Defection can then spread unless the cooperators condition on what it does.

Where the two options agree and where they differ is itself a result.

**In code.**

- There are two separately named functions, `score_individual` and `score_aggregate`.
- The config key `SCORING` has no default and must be set explicitly for every run.
- Its value is written into every output row.

## 8. The baseline

The baseline has no search and no generations. A strategy is assigned by hand to every iGSS agent, one episode is played with the background agents, and this is repeated over `BASELINE_SEEDS` seeds, for every environment.

- This is exactly what **aggregate scoring** measures: a strategy among copies of itself, in an environment of unconditional cooperators and defectors. A baseline episode and an aggregate-scoring episode with the same seed give the same payoff.
- It shows, before any search, how each strategy performs in each environment: payoff, cooperation rates and which input values occur.

| Mode | Strategies in the baseline |
| --- | --- |
| 1 | the 4 action rules, with the base assessment rule |
| 2 | action rule `10` with each of the 16 assessment rules |
| 3 | every pair, counted by class (§14.3): 34 |

**Telling strategies apart.** Each episode logs which input values the iGSS agents actually read. Suppose an input value never occurs in an environment, for example input 0 when nobody ever defects. Rules that differ only in what they do on that value then behave identically there and cannot be told apart. Because every strategy is run on the same seeds, strategies that cannot be told apart get exactly equal payoffs; table T1 (§10) lists the strategies that tie for the best payoff.

The baseline is a command of its own (§11). It is run once for the environments of a sweep and written to its own file; search runs never recompute it. In the baseline each action rule is the named predicate of `rules.py`, used directly.

**No runs in pairs.** Earlier versions also ran pairs of rules, half the iGSS agents each, as a predictor for individual scoring. The even split was arbitrary and the runs were removed (v1.4). How rules fare against each other is what individual scoring itself tests; a baseline for it (for example one agent with rule X among agents with rule Y) is left open (§13).

## 9. The search

**Population.** Each run has one DEAP population of P expressions. Under individual scoring, expression i is the rule of iGSS agent i in that generation's episode.

**Initialisation.** Random expressions, from DEAP's standard ramped half-and-half method (`genHalfAndHalf`), with depth 1–3.

**Depth** means DEAP's tree height throughout: a lone terminal has height 0.

**Expressions.**

- Primitives: `add`, `sub`, `mul`.
- Terminals: the input (`partner_cooperated_with_me_last_time` or `partner_standing`) and the constants 0 and 1. These are three fixed terminals, each equally likely to be drawn.
- An output above 0 means cooperate.

**One generation.**

1. Compile each expression and read off its bit string.
2. Score every rule with the chosen scoring function, and subtract the size penalty (§7) to get its selection score.
3. Log the generation.
4. **Selection:** a tournament of size 3, repeated P times (DEAP `selTournament`). Each time, three rules are drawn at random, with replacement, and the one with the highest selection score is kept. Among equal best, the first drawn is kept, which is a random choice because the draw order is random.
5. **Crossover:** selected parents are paired in order. With probability 0.5, each pair exchanges a randomly chosen sub-expression (DEAP `cxOnePoint`).
6. **Mutation:** with probability 0.2, each child has a randomly chosen sub-expression replaced by a new random one of depth 0–2 (DEAP `mutUniform`, with `genHalfAndHalf` building the new piece).
7. **Depth limit:** any child deeper than 17 is replaced by one of its parents (DEAP `staticLimit`). The replacement arrives only in the value the operator returns, so the code must use that value; the too-deep child itself is left where it was.
8. The children are the next generation. No rule is carried over unchanged (no elitism).

**Numbering.** Generations are numbered 1 to G. Generation 1 is the initial random population. Every generation is scored and logged; every generation except G is then bred. The final generation is generation G.

Agents do not persist from one generation to the next; only rules do, through their offspring.

**Random numbers.**

- Episodes draw from their own random-number generators, kept separate from the generator DEAP's operators use. A change to the episode code therefore cannot silently shift the evolutionary stream.
- **DEAP's stream.** DEAP draws from Python's global `random` module. It is seeded once per run with the run seed.
- **Episode seeds.** The seed of episode k of generation g is the text `"episode|<run seed>|<g>|<k>"`, given to a new `random.Random`. Python turns a text seed into a generator state in the same way on every machine and version. Both scoring options use this formula.
- **Run seeds.** Within a sweep, repetition i of every combination has run seed `BASE_SEED + i`. Every environment and both scoring options therefore start from the same initial expressions, so that they are compared on common starting points. Runs are independent across repetitions, not across combinations.
- **Baseline seeds.** Baseline episode i uses the text `"baseline|<i>"`.

**Every generation logs:**

- every rule's expression, bit string, score and selection score (written to file for generation 1, every `RULES_LOG_EVERY`-th generation and generation G; with `RULES_LOG_EVERY = 1`, for every generation);
- how many rules have each bit string;
- the mean payoff of the iGSS agents and of each background type;
- cooperation rates;
- the input values read.

**What to expect.** A run produces a population, not a single winner. It may:

- converge on one bit string;
- settle into a stable mix;
- drift between rules that behave identically;
- cycle.

The result of a run is the composition of the population over time, plus the expressions behind each bit string.

## 10. Outputs and reading results

**Files.** Each sweep writes into a new directory, `outputs/<sweep>-<date>-<time>/`. Nothing is ever appended to an existing file.

| File | One row per | Columns |
| --- | --- | --- |
| `runs.csv` | run | the parameter columns; the columns of the final generation's row in `generations.csv`; `tied`; `recovered` |
| `generations.csv` | generation of a run | the parameter columns; `generation`; `most_common`; `most_common_name`; `most_common_count`; `mean_size`; `payoff_igss`, `payoff_c`, `payoff_d`; `coop_rate`, `coop_igss`, `coop_igss_to_igss`, `coop_igss_to_c`, `coop_igss_to_d`; `reads_one`, `reads_zero` |
| `composition.csv` | strategy present in a generation of a run | `run_id`, `generation`, `strategy`, `strategy_name`, `count` |
| `rules.csv` | rule in a logged generation | `run_id`, `generation`, `rule_index`, `strategy`, `strategy_name`, `action_expression`, `action_simplified`, `assessment_expression`, `assessment_simplified`, `size`, `height`, `score`, `selection_score` |
| `labels.csv` (in every output directory) | model and strategy name | `model`, `strategy_name`, `colour_key`, `named_reciprocity` |
| `baseline.csv` (written by the baseline command, in its own directory) | baseline episode | the environment columns; `strategy`; `strategy_name`; `seed_index`; then the same payoff, cooperation and read columns as `generations.csv` |

- **Parameter columns:** `sweep`, `run_id`, `code_version`, `run_seed`, `mode`, `scoring`, and every parameter of the configuration in lower case (`model`, `assessment_rule`, `n_igss`, `n_c`, `n_d`, `b`, `c`, `e_action`, `rounds`, `pop_size`, `generations`, and so on). Every row therefore records the environment, the scoring option and the seed. `assessment_rule` is the base rule of the model.
- `run_id` is unique within a sweep. `code_version` is the git commit of the code, with `+changes` added when the working copy differs from it.
- **Strategies.** A strategy is written as `TT:<action rule>/<assessment rule>`, for example `TT:10/1100`. The prefix keeps leading zeros safe in spreadsheet software. Where the action rule is `11` or `00` the assessment rule cannot matter and is written `any` (§14.3): `TT:11/any`. In mode 1 the assessment rule shown is the base rule.
- **Where the canonical strategies appear.** `strategy_name` and `most_common_name` give the literature's name (§16) for a strategy: "Tit-for-Tat", "Grim Trigger", "Standing", "Image Scoring", "Stern Judging", "Shunning", "unconditional cooperation", "unconditional defection", or "unnamed". `labels.csv` lists the names possible in each model; the figures take their legends and colours from it.
- `most_common` is the most common strategy of the generation and `most_common_count` the number of rules that have it. If several tie, all are given, joined by `+`. `tied` is 1 when that happens in the final generation.
- `recovered` is 1 when one strategy alone is the most common in the final generation and §16 names it as a reciprocity strategy of that model. In mode 1 this means `10` with the base assessment rule.
- In `rules.csv` an expression column is empty when that rule does not evolve in the run's mode. `size` counts the nodes of all evolved expressions; `height` is that of the deepest.
- Payoffs are per round. Under aggregate scoring, the per-generation payoffs, rates and read counts are averages over the P rules' episodes.
- `coop_…` columns are shares of interactions in which the donor cooperated (action taken). `reads_one` and `reads_zero` are the number of times iGSS donors read each input value, per episode. For `11/any` and `00/any` under aggregate scoring they are those of an episode under the base assessment rule (§14.3).
- **A type with no agents has empty cells.** When N_C = 0, `payoff_c` and `coop_igss_to_c` are empty, and likewise for N_D. R reads an empty cell as `NA`, so such rows drop out of means and charts and are never mistaken for a payoff of 0.
- `…_simplified` is SymPy's expansion of the expression, for reading only. It is left empty when the rule has more than `SIMPLIFY_MAX_SIZE` nodes, because expanding very large expressions is slow.
- Numbers are written unrounded.

**How to read `baseline.csv`.** Each row is one episode in which every iGSS agent uses the strategy in `strategy`.

- `payoff_igss` is what the iGSS agents earned per round; `payoff_c` and `payoff_d` are what the unconditional cooperators and unconditional defectors earned per round in that episode (empty if there are none).
- `coop_igss` is the share of the iGSS agents' turns as donor in which they cooperated; `coop_igss_to_c`, for example, is that share when the partner was an unconditional cooperator.
- `reads_one` and `reads_zero` count how often an iGSS agent, about to decide, found its input to be 1 or 0: in DR, whether the partner in front of it cooperated with it last time; in IR, the partner's standing. If `reads_zero` is 0, no iGSS agent ever met a partner with a 0, so rules that differ only in what they do on a 0 (`10` and `11`) acted identically.

**Main output.** For each environment and scoring option, over 50 seeds: how many rules have each strategy, at the end of each run and over its course. The expressions behind each strategy are reported alongside.

**The canonical rule** is `10` in both models. It is marked in the output. It is fixed in advance only so that it is clear beforehand which outcome counts as finding the canonical strategy. It changes nothing that is run or measured.

**Recovery rate.** The share of runs with `recovered` = 1. A tie for most common does not count as recovery; ties are reported separately.

**Comparing search with baseline.** When the search does not return `10`, the baseline shows why. Only the last case in the table below counts against iGSS.

| Example | Baseline shows | Search result | Reading |
| --- | --- | --- | --- |
| defectors present, e_action = 0.05 | `10` distinguishable and best for this scoring option | `10` most common in 47 of 50 runs | iGSS works here |
| few rounds, so pairs rarely meet again | `00` beats `10` | `00` most common | iGSS agrees with the baseline; `10` is not favoured here |
| no defectors, e_action = 0 | `10` and `11` behave identically | either, or a mix | nothing can be concluded |
| defectors present, e_action = 0.05 | `10` distinguishable and best | `10` most common in 12 of 50 runs | a genuine search failure |

The numbers in this table are illustrative, not results.

**Figures.** Every figure is drawn from the CSV files of one sweep and its baseline by one R script, `R/figures.R`. It writes PNG and CSV files into `<sweep directory>/figures/`, and computes nothing that is not in the CSV files. No image is produced during a run: the legacy code saved two images per run (a payoff chart and a drawing of the returned expression); here the per-run chart is F5, drawn on request, and expressions are listed in T2.

```
Rscript R/figures.R <sweep directory> [--baseline <baseline directory>] [--runs all | id1,id2,...]
```

Strategies are grouped and coloured by their literature name (§16), each name with a fixed colour. Panels and bars are labelled by the parameters that vary in the sweep (for example `n_c=0, n_d=32`), so the same script serves any sweep and any mode. Figures are drawn separately for each mode and model.

| Figure | Shows | Data | Grouping |
| --- | --- | --- | --- |
| F1. Composition over time | For each generation, the share of the P rules with each strategy name, averaged over seeds, as stacked areas. | `composition.csv`. | One panel per combination of the varying parameters; one figure per scoring option. |
| F2. Final composition | The share of runs in which each strategy name is the most common in the final generation, as stacked bars; ties are their own category. | `runs.csv`: `most_common_name`, `tied`. | One bar per combination of the varying parameters; the two scoring options side by side. |
| F3. Baseline | The payoff per round of the iGSS agents when all use one strategy, as one box per strategy over seeds, with the mean payoff of each background type marked. | `baseline.csv`. | One panel per environment. |
| F5. One run | By generation: the number of rules with each strategy name, and the payoff per round of the iGSS agents and of each background type. | `composition.csv` and `generations.csv`, one `run_id`. | One figure per run, only with `--runs`. |
| F6. Final composition against a parameter | Both measures of final composition (below) for each strategy name, as one line per name, against the values of one varying parameter. | `runs.csv` and `composition.csv`, final generation. | One figure per varying parameter and scoring option; one column of panels per combination of the other varying parameters; one row of panels per measure. |
| F7. Final composition over two parameters | One measure of final composition for each strategy name, as a grid of shaded cells over the first two varying parameters, with the value written in each cell. | As F6. | Drawn when at least two parameters vary. One figure per measure and scoring option; one row of panels per strategy name; one column per combination of the remaining varying parameters. |
| F8. Most common strategy by generation | For each generation, the share of runs in which each strategy name is the most common, as stacked areas; ties are their own category. It shows what F2 would show had the runs stopped at that generation. | `generations.csv`: `most_common`, `most_common_name`. | One panel per combination of the varying parameters; one figure per scoring option. |
| F9. Cooperation against a parameter | The cooperation rates of the final generation (`coop_rate`, `coop_igss`, `coop_igss_to_igss`, `coop_igss_to_c`, `coop_igss_to_d`) against the values of one varying parameter: a dot for each run and a line through the means over runs. | `runs.csv`. | One figure per varying parameter and scoring option; one row of panels per rate; one column per combination of the other varying parameters. A rate with no agents of that type (an empty cell) is left out. |

There is no F4: it showed the runs in pairs, which were removed in v1.4.

**Two measures of final composition.** F6, F7 and T4 report how often a strategy name is found at the end of the runs in two ways:

- **ends most common:** the share of runs in which that name alone is the most common in the final generation. This is the measure behind F2 and `recovered`. Ties are their own category.
- **final share:** the share of the P rules that have that name in the final generation, averaged over runs.

The two differ when a name is common without being the most common. F6 and F7 leave out names that are 0 in every panel of the figure.

**Reading the cooperation rates (F9, T5).** `coop_rate` counts every donor, so with background agents present most of it is their fixed behaviour; the four `coop_igss…` rates are those of the iGSS agents. Under aggregate scoring each value is an average over the episodes of all P rules of the final generation, so it includes the rules that are not the most common.

| Table | Shows | Data |
| --- | --- | --- |
| T1. Search against baseline | For each environment and scoring option: the recovery rate and the share of ties; and the strategies with the highest payoff in the baseline. Strategies listed together there cannot be told apart in that environment. | `baseline.csv` and `runs.csv`. |
| T2. Expressions behind each strategy | For each strategy, the distinct simplified expressions in final generations and how many rules had each. | `rules.csv`, final generation. |
| T3. Most common strategies | For each combination of parameters, which strategies (by bit strings, not only by name) ended most common, and in how many runs. | `runs.csv`. |
| T4. Final composition by name | For each combination of parameters and each strategy name: the number of runs, and both measures of final composition (above). The numbers behind F6 and F7. | `runs.csv` and `composition.csv`, final generation. |
| T5. Cooperation by combination | For each combination of parameters and each cooperation rate: the number of runs, and the mean, lowest and highest value over runs in the final generation. The numbers behind F9. | `runs.csv`. |

**A caution on "unnamed".** In modes 2 and 3 many unnamed strategies behave exactly like a named one. For example, with stored values starting at 1, the assessment rules `1111`, `1110`, `1011` and `1010` never produce a 0 (§14.2), so `10` with any of them acts as unconditional cooperation. F1 and F2 group by name and so show these as "unnamed"; T1 and T3 and the baseline (F3) show which strategies are behaviourally the same.

## 11. Build plan for mode 1

The code is plain Python, with DEAP used for the search only. Mesa is not used: the legacy code used only Mesa's base classes, and Mesa 3 kept every model object in memory.

**Repository layout (proposed, §13):**

```
igss/            new code (the package below)
R/figures.R      figures and tables from the output files (§10)
tests/           one test file per module
docs/SPEC.md     this file
legacy/          the August 2026 code, read-only
outputs/         sweep results; not committed
```

| File (in `igss/`) | Contents |
| --- | --- |
| `rules.py` | Input names, row order, every named rule as a predicate, and the function that turns a compiled expression into a bit string. |
| `episode.py` | The `Agent` definition and `play_episode`. |
| `scoring.py` | `score_individual` and `score_aggregate`. |
| `baseline.py` | The baseline: each strategy alone. |
| `search.py` | The evolution loop. |
| `config.py` | Every parameter and environment. Every parameter is read by the code; a parameter that does nothing is a bug. |
| `run.py` | Command-line entry point: one quick run, a full sweep, the baseline for a sweep, or a dry run listing what would be run. It also writes the CSV files of §10. |

**`config.py`.** It is a plain Python module with three parts:

- `BASE`: one dictionary holding every parameter except `MODE`, `MODEL` and `SCORING`, which have no base value and must be given by a sweep or on the command line;
- `SWEEPS`: named sweeps, each a dictionary from a parameter to the list of values it takes. A sweep runs every combination of its lists, with every other parameter at its base value, `SEEDS_PER_CELL` times;
- `expand_sweep` and `validate`: the functions that turn a sweep into one configuration per run and check each one. No code anywhere supplies a fallback for a missing parameter; a missing or invalid parameter stops the run with an error.

The assessment rule of each model is named in `BASE` by the name of its predicate in `rules.py`, not by a bit string.

**Commands.**

```
python -m igss.run quick --mode 1 --model DR --scoring aggregate
python -m igss.run dry-run first
python -m igss.run sweep first --workers 4
python -m igss.run baseline first
```

**Order of work.** Each step is finished, tested and reviewed before the next begins.

1. **`rules.py`.** Tests:
   - the expression `partner_standing` gives `10`;
   - the constant 1 gives `11`;
   - `sub(1, partner_standing)` gives `01`;
   - every named rule's bit string matches the tables in this file.
2. **`episode.py`.** Tests:
   - a hand-checked trace for each model: 3 agents, 2 rounds and a scripted sequence of random draws, with every payoff and every record or standing checked by hand;
   - with e_action = 0, every action taken equals the decision;
   - background agents always decide what their kind says;
   - with e_action = 1, no cooperation ever happens;
   - the same seed gives the same output.
3. **`baseline.py`.** Tests: with no defectors and e_action = 0, `10` and `11` give identical results and input 0 is never read; the baseline has 4, 16 and 34 strategies in modes 1, 2 and 3; a strategy whose evolved assessment rule is the base rule plays exactly as in mode 1.
4. **`scoring.py`.** Tests:
   - a rule's aggregate score equals its alone-run payoff for the same seed;
   - rules with the same bit string get the same aggregate score within a generation.
5. **`search.py`.** Tests:
   - a run is fully reproducible from its seed;
   - the logged bit-string counts add up to P in every generation;
   - changing R leaves the expressions of generation 1 unchanged, and no expression is ever higher than the depth limit.
6. **`config.py`.** Tests: a configuration without `SCORING` fails; every parameter in `BASE` is read somewhere in `igss/`; every run of the first sweep passes `validate`.
7. **`run.py`** and the first sweep. Test: a small sweep writes the files and columns of §10.

**Expectations to check, not tests.** If any of these fail, look for a bug before reading it as a finding.

- Aggregate scoring, defectors present, enough rounds: `10` or `11` should do well and `00` badly.
- Individual scoring, very few rounds (pairs almost never meet again): `00` should spread.

## 12. Working practices for building with an AI assistant

The legacy code drifted partly because choices were filled in silently: a default, a row order, a control rule. These practices keep every choice visible. `CLAUDE.md` turns them into standing instructions for Claude Code.

1. **Specification first.** No code is written for anything not described here. When the code needs a choice this file does not make, the choice is added here: agreed with an author first if it can change which rules evolve, otherwise recorded with status "assistant" and reported.
2. **Small steps.** One function or one change at a time, read line by line before it is accepted.
3. **Ask what was assumed.** After each change, the assistant lists every choice it made that this file does not specify.
4. **Hand-checkable tests**, as listed in §11.
5. **One codebase, under version control.** There are no parallel implementations in chat sessions. Each change is a commit whose message names the section it implements.
6. **Numbers only from the authors' own runs.** No result is reported unless the committed code produced it on the authors' machines.
7. **Independent review.** A fresh session reviews each change against this file, without seeing the session that wrote it.

## 13. Decision register

- **Decided:** agreed by the authors.
- **Proposed:** a default written here so that nothing is left silent, not yet confirmed.
- **Assistant:** chosen by the AI assistant while building, under the rule in the header of this file, and awaiting the authors' review.
- **Review:** decided for now, and marked by an author to be looked at again once output exists.

| Topic | Choice | Status | Alternatives to revisit |
| --- | --- | --- | --- |
| Models | DR and IR separate | decided | combined DR+IR model |
| Order of modes | mode 1 first | decided | — |
| Inputs | one per action rule; no junk inputs | decided | junk inputs |
| Input coding | 1 = the event that would normally favour cooperating. DR and IR each follow their own literature, kept as coherent with each other as possible. | review | polarity-flipped control run |
| Terms | donor / partner; action rule / assessment rule; iGSS agent; episode; no interpretive labels | decided | — |
| Expressions | `add`, `sub`, `mul`; constants 0 and 1 as two fixed terminals; above 0 = cooperate | decided | Boolean primitives as a cross-check; `if-then` primitive |
| Classification | evaluate the compiled expression on every input row; SymPy for reading only | decided | — |
| Framework | plain Python + DEAP; no Mesa | decided | Mesa for visualisation later |
| Scoring | both individual and aggregate, as separate named functions, compared | review | — |
| iGSS agents per episode | N_I, a parameter separate from P under aggregate scoring; N_I = P under individual scoring and in the base configuration | decided | one symbol for both, as in v1.1 |
| Episodes per scoring | E, averaged; 1 in the base configuration | decided | — |
| Baseline | each strategy alone (every iGSS agent uses it), over many seeds; computed once per sweep, not per run. Runs in pairs removed in v1.4. | decided | a baseline for individual scoring, for example one agent with X among agents with Y |
| Main focus | aggregate scoring: a rule among copies of itself, in varying environments of cooperators and defectors. Individual scoring is built and kept as a separate question, to be taken up again once aggregate scoring has been run and a base version of the model is working. | decided | — |
| Rounds | R = 400 in the base configuration, raised from 200 when the populations became 32/32/32, to keep about 4 meetings per ordered pair | assistant | other values; the `rounds` sweep varies it |
| Main output | count of each bit string per environment and scoring option, with expressions; `10` marked | decided | — |
| SSOT location | this file, in the repository | decided | — |
| Error types in mode 1 | execution error only | proposed | assessment error (modes 2–3; §14.4) |
| Direction of execution error | one-way: an intended cooperation fails with probability e_action; a decision to defect is never changed. The same in DR and IR. | decided | two-way (either action flips), the convention of the repeated prisoner's dilemma |
| Execution error on background agents | applies to every donor | decided | iGSS agents only |
| Order of random draws | per interaction: partner draw, then one error draw whatever the decision | assistant | draw for error only when it can matter |
| Generations | non-overlapping: a fresh episode each generation | proposed | one continuous episode in which a few agents' rules are replaced each round |
| Who applies the assessment rule | the partner, after observing the action | proposed | (matters only from mode 2 on) |
| Start values | DR records 1; IR standings 1 | decided | 0, or random |
| Update timing | stored values change immediately, within the round | decided | apply all updates at the end of each round |
| Score unit | payoff per round | decided | total payoff |
| Aggregate scoring seed | same seeds for all rules within a generation | assistant | independent seeds |
| Seed scheme | as in §9 "Random numbers": the run seed seeds DEAP's stream; episode seeds are texts built from (run seed, generation, episode); run seeds are shared across the combinations of a sweep | review | a distinct run seed for every combination and repetition |
| DR assessment | `1100`. The legacy step that reset a defecting donor's own record is dropped. | decided | `1000` |
| IR assessment | `1101`. The legacy case that left a standing unchanged is dropped; no rule reads the donor's own previous standing. | decided | `1100`, `1001`, `1000` as further environments |
| Selection | DEAP `selTournament`, size 3: draws with replacement; among equal best the first drawn is kept | assistant | other tournament sizes; draws without replacement |
| Crossover / mutation | one-point crossover p = 0.5; uniform mutation p = 0.2, new sub-expression depth 0–2, built by `genHalfAndHalf` | assistant | `genFull` or `genGrow` for the new sub-expression |
| Meaning of depth | DEAP tree height (a lone terminal is 0) | review | counting levels from 1 |
| Size control | depth limit 17 (child replaced by one of its parents), and a per-node penalty `PARSIMONY_TAX` on the selection score, 0 in the base configuration and varied as a search factor (§7) | decided | preferring the smaller expression on ties. Any size pressure acts as a prior on which rules are found. |
| Elitism | none | review | carry the best rule forward unchanged |
| Recovery criterion | one strategy alone is the most common in the final generation, and §16 names it as a reciprocity strategy of the model (in mode 1: `10`); ties reported separately | decided | majority over the last X generations; its share of the population |
| Repository layout | as in §11 | proposed | — |
| Tooling | Python 3.14; DEAP 1.4.3, SymPy 1.14.0 and pytest 9.2.0 pinned in `requirements.txt`. DEAP's operators define parts of the model, so its version is part of this specification. | assistant | — |
| Figures | as in §10: one R script, `R/figures.R`; no images during runs; no drawing of expression trees | assistant | drawing expression trees, as the legacy code did |
| Population sizes | N_I = N_C = N_D = 32 in the base configuration, with 0 cooperators or 0 defectors as further environments. Equal groups replace the 32/10/10 of v1.2, which had no motivation. | decided | 10/10/10, as the legacy code had, for speed; analytical results for mode 1 |
| Literature names in output | `most_common_name` and `labels.csv`, derived from §16; a strategy §16 does not name is "unnamed" | assistant | — |
| Mode | `MODE` (1, 2 or 3) has no base value, like `MODEL` and `SCORING` | assistant | — |
| Which assessment rule applies (modes 2 and 3) | aggregate scoring: the strategy's own assessment rule is the one rule of the episode, applied after every interaction whoever the partner is. Individual scoring: the partner's rule; when the partner is a background agent, the base rule. | decided | — |
| Mode 3 operators | each crossover or mutation acts on one of the two expressions, chosen with equal chance | decided | acting on both |
| Baseline in modes 2 and 3 | every strategy class alone: 16 and 34 | decided | — |
| Counting strategies | by class: `11` and `00` with any assessment rule are one class each (§14.3) | assistant | counting all 64 pairs separately |
| Rule that `11/any` and `00/any` are simulated with | the base assessment rule, under aggregate scoring and in the baseline (§14.3). Before v1.8 it was the rule of whichever member came last in the population, and `1111` in the baseline, so the logged read counts of these classes were arbitrary. Scores were never affected. | assistant | simulating every member with its own rule; leaving the read counts of these classes empty |
| Size in mode 3 | the size penalty counts the nodes of both expressions | assistant | — |
| Even populations | P and N_I are even; checked at start-up | decided | — |
| Benefit and cost | c = 1 in all runs; the configuration holds b and c | decided | holding the ratio b/c |
| Agent ids and list order | iGSS agents, then cooperators, then defectors; id = position | assistant | — |
| Shuffle | each round shuffles a fresh copy of the list in id order | assistant | shuffling the previous round's order |
| What agents run when scored | the rule's bit string, read from the compiled expression (§7) | assistant | calling the compiled expression at every decision |
| Generation numbering | 1 to G; generation 1 is the initial population; G is scored and not bred | assistant | numbering from 0 |
| Baseline rules | the named predicates of `rules.py`, used directly | assistant | hand-written DEAP expressions |
| Baseline seeds | `BASELINE_SEEDS` episodes per set, 50 in the base configuration | assistant | — |
| Absent background type | empty cells | decided | 0 |
| Output files | as in §10: a new directory per sweep, four files, unrounded numbers, `run_id` and `code_version` in every row | assistant | — |
| Per-rule logging | `rules.csv`, written every `RULES_LOG_EVERY` generations (10 in the base configuration) plus the first and last | assistant | every generation (set it to 1) |
| Simplified form | SymPy expansion, only up to `SIMPLIFY_MAX_SIZE` nodes (50) | assistant | — |
| Shape of `config.py` | `BASE`, `SWEEPS`, `expand_sweep`, `validate` (§11); no fallback values | assistant | — |
| Varying the fixed assessment rule | `DR_ASSESSMENT_RULE` and `IR_ASSESSMENT_RULE` are separate parameters, and each model reads only its own. A sweep that varies one of them must run that model alone; otherwise it is refused, because the other model's combinations would be run several times, identically. | assistant | one parameter for both models |
| Base values of swept parameters | in `BASE`: N_C = 32, N_D = 32, b = 5, e_action = 0. These apply only in sweeps that do not vary them. | assistant | — |
| Base seed | `BASE_SEED` = 1 | assistant | — |
| Quick run and smoke sweep | `quick`: the base configuration for 10 generations, one seed. `smoke`: a very small sweep used by the tests. Neither produces results. | assistant | — |
| Parallel runs | `--workers` runs several runs at once; rows are written in the same order whatever the number of workers | assistant | — |
| The `base` sweep | the base configuration in every mode and model, aggregate scoring only, 10 seeds each: a first look before the full sweeps | assistant | — |
| Sweeps other than the first | `rounds` and `parsimony` in `config.py` are examples with placeholder values, for the authors to edit | assistant | — |
| Exploratory sweeps for IR in mode 2: design | §14.5: aggregate scoring; a full grid over N_C {0, 32} × N_D {0, 32} × b {2, 5, 10}; R, G and search parameters each varied alone; 10 seeds per combination | decided | 20 or 50 seeds; N_C and N_D of 16 or 64; individual scoring |
| Exploratory sweeps for IR in mode 2: execution error | `ir2_env_e005`: the environment grid again with e_action = 0.05, as a sweep of its own so that the runs at e_action = 0 are not repeated | decided | other values of e_action; e_action crossed with the parameters varied alone |
| Exploratory sweeps for IR in mode 2: values of the parameters varied alone | R: 100, 400, 1600. Size penalty: 0, 0.01, 0.05. Depth limit: 3, 5, 17. Tournament size: 2, 3, 7. Proposed by the assistant, including the choice of tournament size, and confirmed by an author after the first run. | decided | other values; P; crossover and mutation probabilities; initial depths |
| Varying G | one sweep with G = 200, read at earlier generations, because a shorter run is exactly the start of a longer one with the same seed (§14.5); a test checks this | assistant | separate runs for each G |
| Second measure of final composition | "final share" (mean share of the final population) reported beside "ends most common" in F6, F7 and T4 (§10). It adds a reading of the same runs and changes nothing that is run. | assistant | majority over the last X generations |
| First look at individual scoring | `individual_look` (§14.6): both models, modes 1 to 3, the four backgrounds, b = 10, e_action = 0, G = 200, 20 seeds. An author asked for b = 10 and all backgrounds, and accepted the assistant's values for the rest. | decided | e_action of 0.05; E above 1; runs started from a chosen rule; a baseline for individual scoring |
| Broad exploratory sweep | §14.7: ten named sweeps over both models, all modes and both scoring options; G = 200; 20 seeds (10 for the search parameters and the long runs). Drafted by the assistant at an author's request. | proposed | every factor and value in §14.7; 50 seeds; other background sizes; the points §14.7 lists as needing new code |
| Assessment rule as a varying parameter in figures | when the fixed assessment rule varies within a sweep, the figures label panels with it and T1 matches baseline and runs on it | assistant | — |
| Cooperation figure | F9 and T5 (§10): the five logged cooperation rates of the final generation, as means over runs with each run shown | assistant | cooperation by generation |
| One sweep per parameter varied alone | a sweep in `config.py` is a full grid, so each parameter varied alone is a named sweep of its own, with its own output directory | assistant | a sweep type that varies one factor at a time |

**Parameters for the first sweep.**

| Parameter | Value |
| --- | --- |
| P, N_I | 32 |
| E | 1 |
| size penalty | 0 |
| N_C, N_D | 0 or 32 each |
| b, c | b/c = 2 or 5, c = 1 |
| e_action | 0, 0.05 |
| R | 400 |
| G | 100 |
| seeds per environment and scoring option | 50 |
| initial expression depth | 1–3 |

## 14. Modes 2 and 3

Modes 2 and 3 use the same episode, scoring options, baseline and search as mode 1. Only the rule that evolves changes.

| Mode | Evolved | Fixed | Strategies per model |
| --- | --- | --- | --- |
| 1 | action rule (1 input) | assessment rule: the base rule | 4 |
| 2 | assessment rule (2 inputs) | action rule `10` | 16 |
| 3 | both, as a pair of expressions | — | 4 × 16 = 64, in 34 classes (§14.3) |

**The assessment expression.** It has two inputs, in this order: `donor_cooperated`, then `previous_record` (DR) or `partner_standing` (IR). It uses the same primitives and constants as the action expression, is generated and bred in the same way, and an output above 0 means the stored value becomes 1. It is classified by its output on the four rows of §5.

**Which assessment rule applies.** Every interaction updates a stored value: in DR the partner's record of the donor, in IR the donor's public standing. This happens whoever the donor and the partner are, background agents included.

- **Aggregate scoring.** Every iGSS agent in an episode carries the same strategy, so the episode has one assessment rule: the strategy's. It is applied after every interaction. It can be thought of as the rule of a ledger that every agent can read.
- **Individual scoring.** The iGSS agents carry different assessment rules, so there is no single rule. The partner's rule is applied. When the partner is a background agent, which has no evolved rule, the model's base rule is applied.

**Mode 3 breeding.** Each crossover or mutation acts on one of the two expressions, chosen with equal chance. Crossover exchanges material between the two parents' expressions of the same kind. The depth limit applies to each expression; the size penalty counts the nodes of both.

### 14.1 DR assessment rules

The inputs are (`donor_cooperated`, `previous_record`), and the output is the partner's new record of the donor. Rows: (1,1), (1,0), (0,1), (0,0).

| Bits | New record |
| --- | --- |
| `1111` | always 1 |
| `1110` | 0 only if the donor defects while the record is already 0 |
| `1101` | 1 after cooperation; after defection, the opposite of the previous record |
| `1100` | what the donor did (mode 1 base) |
| `1011` | 0 only if the donor cooperates while the record is 0 |
| `1010` | never changes |
| `1001` | 1 if the action matches the previous record |
| `1000` | 1 only if the donor cooperates and the record was 1 |
| `0111` | 0 only if the donor cooperates and the record was 1 |
| `0110` | 1 if the action differs from the previous record |
| `0101` | flips at every interaction |
| `0100` | 1 only if the donor cooperates and the record was 0 |
| `0011` | the opposite of what the donor did |
| `0010` | 1 only if the donor defects and the record was 1 |
| `0001` | 1 only if the donor defects and the record was 0 |
| `0000` | always 0 |

### 14.2 IR assessment rules (second-order)

The inputs are (`donor_cooperated`, `partner_standing`), and the output is the donor's new standing. The row order is the same.

| Bits | New standing of the donor |
| --- | --- |
| `1111` | always 1 |
| `1110` | 0 only for refusing a partner with standing 0 |
| `1101` | 0 only for refusing a partner with standing 1 (mode 1 base) |
| `1100` | what the donor did; partner's standing ignored |
| `1011` | 0 only for helping a partner with standing 0 |
| `1010` | the partner's standing; action ignored |
| `1001` | 1 if the action matches the partner's standing |
| `1000` | 1 only for helping a partner with standing 1 |
| `0111` | 0 only for helping a partner with standing 1 |
| `0110` | 1 if the action differs from the partner's standing |
| `0101` | the opposite of the partner's standing |
| `0100` | 1 only for helping a partner with standing 0 |
| `0011` | the opposite of what the donor did |
| `0010` | 1 only for refusing a partner with standing 1 |
| `0001` | 1 only for refusing a partner with standing 0 |
| `0000` | always 0 |

In both models, stored values start at 1. Rules `1111`, `1110`, `1011` and `1010` therefore never produce a 0 and behave like "always 1".

### 14.3 Classes of strategy

There are 64 pairs per model, each identified by its two bit strings. When the action rule is `11` or `00`, the stored value is never read. The assessment rule then evolves without any selection pressure, and its bit string carries no information. Such pairs are therefore counted as one class each, written `11/any` and `00/any`, which leaves 2 + 32 = 34 classes. Under aggregate scoring all members of a class get the same score.

**Which rule these two classes are simulated with.** Their payoffs and cooperation rates do not depend on the assessment rule, but the stored values do, and so do the logged read counts. Under aggregate scoring and in the baseline, `11/any` and `00/any` are therefore simulated with the base assessment rule, whichever rules the members carry. What they log is then exactly what `11` and `00` log in mode 1. Under individual scoring nothing changes: every agent applies the rule it carries (§14).

### 14.4 Points settled, and points left open

1. **Whose assessment rule is applied:** settled as above.
2. **Weak selection on IR assessment rules under individual scoring.** If my rule sets your public standing, the consequences fall mostly on the other agents who later decide whether to help you, not on me. My own payoff then barely depends on my assessment rule, so selection can hardly favour one over another. In DR the record is used only by its owner, so this does not arise, and aggregate scoring does not have the problem. This is a property of the set-up to be checked, not a reason to change it.
3. **Error types.** Only execution error is implemented. Assessment error (`e_assessment`: the action is misrecorded) is left for later; it affects rules differently from execution error and should be varied separately.
   - Under two-way execution error in IR, an always-defector would occasionally help by accident and, under `1101`, regain standing 1. This is one reason execution error is one-way (§13).
4. **Third input for IR.** `donor_standing` as a third assessment input, giving 8 rows and 256 rules, is left for later. The classification code accepts any number of inputs.
5. **What counts as found.** `recovered` is 1 when the single most common strategy is one §16 names as a reciprocity strategy of the model. Unnamed strategies that behave like a named one are not counted; T1 and T3 (§10) show them.

### 14.5 Exploratory sweeps for IR in mode 2

A first look at how the final composition depends on the parameters, in one model and one mode: IR, mode 2, aggregate scoring. The object of interest is how often each named strategy (§16) is found at the end of the runs, to be set later against what the evolutionary literature without iGSS reports for the same parameters.

The environment is varied as a full grid. Every other parameter is varied alone, with the rest at their base values (§6.3, §13). Each row below is one named sweep in `config.py`, with 10 seeds per combination.

| Sweep | Varied | Values | Combinations |
| --- | --- | --- | --- |
| `ir2_env` | N_C × N_D × b | N_C: 0, 32; N_D: 0, 32; b: 2, 5, 10 | 12 |
| `ir2_env_e005` | N_C × N_D × b, with e_action = 0.05 | the same grid as `ir2_env` | 12 |
| `ir2_rounds` | R | 100, 400, 1600 | 3 |
| `ir2_generations` | G | 200, read at earlier generations | 1 |
| `ir2_parsimony` | size penalty (`PARSIMONY_TAX`) | 0, 0.01, 0.05 | 3 |
| `ir2_depth` | depth limit (`DEPTH_LIMIT`) | 3, 5, 17 | 3 |
| `ir2_tournament` | tournament size (`TOURNAMENT_SIZE`) | 2, 3, 7 | 3 |

- **Length of play.** R is the only control on how long the game lasts (§6.3). In IR a donor reads a public standing, so what R changes is how many times each standing is updated and read, not how often a pair meets again.
- **Generations.** Nothing in a run depends on G except when it stops: the run seed and the episode seeds (§9) do not involve G. A run with a smaller G is therefore exactly the first generations of a run with a larger G and the same seed. G is varied by running 200 generations once and reading the composition at earlier generations (F1 and F8, §10).
- **Search parameters.** The size penalty is in units of payoff per round, per node (§7). The depth limit cannot be below the largest initial depth, 3. Tournament size sets the strength of selection: with size 2 the better of two random rules is kept, with size 7 the best of seven.
- **The base combination is repeated.** Every sweep contains the base configuration (N_C = N_D = 32, b = 5, R = 400, and so on) with the same seeds, so those runs are identical across the sweeps for G up to 100.
- **Execution error.** e_action is 0 in every sweep except `ir2_env_e005`, which repeats the environment grid with e_action = 0.05, the second value of the first sweep (§6.3). The two grids share run seeds, and runs that differ only in e_action share their matchings (§4), so each run of one has a counterpart in the other. With errors, a 0 can arise even when there are no defectors, so the collapse of §6.3 no longer holds. The figures of `ir2_env_e005` label only the parameters that vary within it; its e_action is in the sweep's name and in every row.
- P, the crossover and mutation probabilities and the initial depths stay at their base values. These are left for later sweeps (§13).

These sweeps are small and exploratory. They are not the main output of §10, which rests on 50 seeds.

### 14.6 A first look at individual scoring

A small sweep, `individual_look`, to see how the model behaves under individual scoring before a broader sweep is designed. It is not a result.

| Factor | Values |
| --- | --- |
| model | DR, IR |
| mode | 1, 2, 3 |
| scoring | individual |
| N_C | 0, 32 |
| N_D | 0, 32 |
| b | 10 |
| e_action | 0 |
| G | 200 |
| seeds per combination | 20 |

Everything else is at its base value. That is 24 combinations and 480 runs.

**How individual scoring relates to the literature.** Selection on each agent's own payoff is the usual set-up of the literature on indirect reciprocity. Its classical models fix one assessment rule for the whole population and let the action rules compete: that is mode 1 under individual scoring. Modes 2 and 3 under individual scoring, where agents carry different assessment rules, are the less usual case.

**Points to keep in mind when reading it.**

- **Background partners.** A background agent never reads a stored value, so the rule it applies as partner does not change what it does. In IR it does change the donor's public standing, which iGSS donors read later. With N_C = N_D = 32, about two thirds of an agent's partners are background agents, so about two thirds of the assessments use the base rule.
- **Weak selection on assessment rules** (§14.4, point 2). In IR the assessment rule an agent carries mostly affects other agents. In modes 2 and 3 its share of the population is therefore expected to drift.
- **Names in modes 2 and 3.** A strategy is named after the assessment rule the agent itself carries. The standings in an episode are set by a mix of rules, so a name being the most common does not mean that rule is the one in force.
- **The starting mix matters.** A rule's payoff depends on which other rules are present. The initial random expressions therefore decide more than the speed of the search: they can decide where a run ends.
- **No baseline.** The baseline (§8) measures a strategy among copies of itself, which is what aggregate scoring rewards. It does not predict the outcome under individual scoring; a baseline for it is still open (§13).
- **Noise.** With E = 1, a score rests on 400 decisions (§7).

### 14.7 A broad exploratory sweep (proposed)

A wider look at both models, all three modes and both scoring options, designed after the exploratory sweeps of §14.5 and §14.6. It is a proposal: its factors and values are not yet confirmed by an author (§13). It is made of ten named sweeps in `config.py`, each a full grid, which can be run separately and in any order.

Every sweep uses G = 200, because 100 generations were read as too few in mode 2 (§14.5), and 20 seeds per combination unless the table says otherwise. Parameters not listed are at their base values.

| Sweep | Question | Fixed | Varied | Combinations | Runs |
| --- | --- | --- | --- | --- | --- |
| `broad_individual` | Where does reciprocity hold under individual scoring? | individual scoring | model × mode × N_C {0, 8, 32} × N_D {0, 8, 32} × b {2, 5, 10} × e_action {0, 0.05} | 324 | 6480 |
| `broad_aggregate` | The same for aggregate scoring. | aggregate scoring | model × mode × N_C {0, 32} × N_D {0, 32} × b {2, 3, 5} × e_action {0, 0.05} | 144 | 2880 |
| `broad_norms_ir` | With the assessment rule fixed, as in the classical models, which rule lets `10` hold? | IR, mode 1, b = 10 | scoring × assessment rule {`1101`, `1100`, `1001`, `1000`} × N_C {0, 32} × N_D {0, 32} × e_action {0, 0.05} | 64 | 1280 |
| `broad_norms_dr` | The same for DR. | DR, mode 1, b = 10 | scoring × assessment rule {`1100`, `1000`} × N_C {0, 32} × N_D {0, 32} × e_action {0, 0.05} | 32 | 640 |
| `broad_rounds` | Does length of play matter in DR, where pairs must meet again? | DR | mode × scoring × R {100, 400, 1600} | 18 | 360 |
| `broad_episodes` | Does a less noisy score change the outcome under individual scoring? | individual scoring | model × mode × E {1, 4} | 12 | 240 |
| `broad_parsimony` | Search: size penalty. | 10 seeds | model × mode × scoring × `PARSIMONY_TAX` {0, 0.01, 0.05} | 36 | 360 |
| `broad_depth` | Search: depth limit. | 10 seeds | model × mode × scoring × `DEPTH_LIMIT` {3, 5, 17} | 36 | 360 |
| `broad_tournament` | Search: strength of selection. | 10 seeds | model × mode × scoring × `TOURNAMENT_SIZE` {2, 3, 7} | 36 | 360 |
| `broad_long` | Has a run settled by generation 200? | G = 500, 10 seeds | model × mode × scoring | 12 | 120 |

That is 13,080 runs.

**Reasons for the values.**

- **Background sizes.** Under individual scoring a small number of background agents may act differently from an equal number, so `broad_individual` adds 8 between 0 and 32. Under aggregate scoring the two-by-two grid of §14.5 is kept.
- **b.** Under aggregate scoring selection uses only the order of the scores, and the order was read as changing between b = 2 and b = 5, so `broad_aggregate` adds 3 and drops 10. Under individual scoring a higher b is the more informative end, so `broad_individual` keeps 10.
- **Execution error** is crossed with both environment grids. It was read as changing little in IR mode 2 under aggregate scoring, where the action rule is fixed; DR and modes 1 and 3 have not been looked at.
- **Fixed assessment rules.** `broad_norms_ir` and `broad_norms_dr` run mode 1 with each named assessment rule of §16 as the fixed rule. Under individual scoring this is the set-up of the classical literature (§14.6).
- **Length of play** is varied for DR only: in IR a donor reads a public standing, and R only averages out noise (§14.5). With N = 96, R of 100, 400 and 1600 gives about 1, 4 and 17 meetings per ordered pair.
- **Generations** are varied as in §14.5, by one longer sweep read at earlier generations.

**The baseline** (§8) is computed for `broad_aggregate`, the two `broad_norms` sweeps and `broad_rounds`. It does not predict the outcome under individual scoring (§14.6).

**Not in this sweep, because each needs code that does not exist yet:**

- the initial mix as a factor: runs started from a chosen rule or mix of rules, to separate "can it emerge" from "does it hold";
- a baseline for individual scoring: one agent with rule X among agents with rule Y;
- assessment error (§14.4, point 3);
- private standings, where agents can disagree about a standing.

## 15. Later rungs

- **Modes 2 and 3**, once the open issues in §14.4 are settled.
- **Third-order IR:** `donor_standing` as a third assessment input.
- **Junk inputs:** random, uninformative inputs, to test whether the search learns to ignore them.
- **Combined DR+IR model:** both informative inputs available to one action rule. Known risk: in the legacy combined runs, private records and public standings carried nearly the same information.
- **Continuous episodes:** one episode per run, with a few agents' rules replaced each round, as an alternative to generations.
- **Other extensions:**
  - assessment error;
  - Pavlov / win-stay-lose-shift, which needs the agent's own last action;
  - an input set with no built-in split between action and assessment.

## 16. Literature cross-reference

This is the only place literature names appear. Each name refers to a strategy: an action rule plus the assessment rule it runs with.

| Model | Action rule | Assessment rule | Literature name |
| --- | --- | --- | --- |
| DR | `10` follow the record | `1100` | Tit-for-Tat |
| DR | `10` follow the record | `1000` | Grim Trigger |
| IR | `10` follow the standing | `1101` | Standing ("simple standing") |
| IR | `10` follow the standing | `1100` | Image Scoring |
| IR | `10` follow the standing | `1001` | Stern Judging |
| IR | `10` follow the standing | `1000` | Shunning |
| both | `11` | any | unconditional cooperation |
| both | `00` | any | unconditional defection |

## 17. Legacy code (August 2026)

The legacy code lives in `legacy/`. It is frozen as a reference: it is not to be extended, imported or fixed.

Known problems in the six model scripts (`DR_mode1.py` to `IR_mode3.py`):

- the DR memory dictionaries use the wrong row order, so "Grim Ledger" and "Sticky Memory" are mislabelled;
- the DR control rule in modes 2–3 departs from the original model;
- the mode-3 mirror labels are wrong;
- tree depth and tournament size were typed into the scripts rather than read from the config.

Other notes:

- The legacy design corresponds to aggregate scoring.
- DR results can be relabelled from the logged `Best_Rule_SymPy` column if they are ever needed.
- The legacy "Test summary" entries marked "Yes" are not established.
- The other legacy files (utilities, logging, graph generation, R visualisation) are inventoried in `docs/legacy_inventory.md` and compared with this file in `docs/rebuild_plan.md`.

Further differences found in that comparison (details in `docs/rebuild_plan.md` §2.2):

- the legacy IR update left a donor's standing unchanged when it refused a partner with standing 0; it was not `1101`;
- the legacy DR mode 1 update was `1000` plus a reset of the donor's own record when it defected; it was not `1100`;
- the legacy DR input was coded with 1 = the partner defected, the opposite of this file's coding;
- the legacy sweep grid varied `TREE_MAX_DEPTH`, which no script read;
- legacy scores within one generation were drawn in different generations, and the expression and score columns of a CSV row could describe different rules.

Both legacy update rules came from an earlier agent-based model. They are dropped, not reproduced.

## 18. Change log

Newest first. Git history records every edit; this log records changes in meaning.

| Version | Date | Change |
| --- | --- | --- |
| v1.8 | 16 Nov 2026 | Corrections after a review. The classes `11/any` and `00/any` are simulated with the base assessment rule under aggregate scoring and in the baseline (§14.3), so their logged read counts no longer depend on the order of the population; scores are unchanged. A sweep that varies one model's assessment rule while running the other model is refused (§13). |
| v1.7 | 16 Nov 2026 | A broad exploratory sweep drafted as a proposal (§14.7): ten named sweeps over both models, all modes and both scoring options, including mode 1 with each named assessment rule as the fixed rule. Figures treat the fixed assessment rule as a parameter that can vary. |
| v1.6 | 16 Nov 2026 | A first look at individual scoring added (§14.6): the sweep `individual_look`, and points to keep in mind when reading individual scoring in modes 2 and 3. |
| v1.5.2 | 16 Nov 2026 | Figure F9 and table T5 added (§10): cooperation rates of the final generation against the varied parameters. |
| v1.5.1 | 15 Nov 2026 | Sweep `ir2_env_e005` added (§14.5): the IR mode 2 environment grid with e_action = 0.05. The values of the parameters varied alone are now decided (§13). |
| v1.5 | 15 Nov 2026 | Exploratory sweeps for IR in mode 2 added (§14.5): an environment grid, and R, G, size penalty, depth limit and tournament size each varied alone. Figures F6, F7 and F8 and table T4 added (§10), with a second measure of final composition, "final share", beside "ends most common". Figures now have a minimum width. |
| v1.2 | 15 Nov 2026 | Written with the first build of mode 1, after an author's review of `docs/rebuild_plan.md`. Execution error decided: one-way, every donor, both models. Constants are two fixed terminals. N_I (iGSS agents per episode) separated from P for aggregate scoring; E (episodes per scoring) added. A size penalty, `PARSIMONY_TAX`, added as a search parameter, 0 in the base configuration. Every parameter can now be varied through named sweeps. Seed scheme, generation numbering, agent order and DEAP details written out. Output files and columns specified; a background type with no agents is logged as empty cells. Figures proposed. `config.py` and the commands described. New status values "assistant" and "review" in §13, and the header now lets the assistant make and record choices that cannot change which rules evolve. Legacy findings added to §17. |
| v1.4 | 15 Nov 2026 | Modes 2 and 3 built and §14 written out: assessment expressions, which assessment rule applies under each scoring option, mode 3 breeding, strategy classes. Runs in pairs removed from the baseline, the output and the figures (F4 dropped); the baseline is now each strategy alone, in every mode. The main focus is stated as aggregate scoring. R raised to 400. `MODE` added as a parameter without a base value. Outputs now identify strategies as `TT:action/assessment` with their literature names; `composition.csv` replaces the `count_…` columns; `rules.csv` has one expression column per kind of rule. `recovered` redefined for all modes. Table T3 added. |
| v1.3 | 15 Nov 2026 | Base population sizes changed to N_I = N_C = N_D = 32 (the first sweep now uses N_C and N_D of 0 or 32). Literature names added to the output as labels: `most_common_name` and `labels.csv`, and the figure legends. §8 now says why the baseline has runs in pairs; §10 says how to read `baseline.csv`. |
| v1.2.1 | 15 Nov 2026 | Figures built as `R/figures.R` and §10 "Figures" rewritten to match: panels are labelled by the parameters that vary, F2 shows ties, T1 defined, no images during runs. |
| v1.1 | 14 Nov 2026 | Moved from a Claude Doc into the repository, with numbered sections. Clarifications that make no change in meaning: the IR standing update now says the partner's standing used is the value the donor read; aggregate scoring is described as the "mean" payoff throughout (v1.0 said "total" once). Execution error reopened. v1.0 silently assumed one-way error (the IR convention) for both models, and its tests pulled in opposite directions on whether background agents are subject to error. The direction of error and its scope are now open decisions in §13, and the e_action test waits on them. New proposed rows: seed scheme, repository layout, tooling. Depth-limit wording aligned with DEAP (`staticLimit` returns one of the parents). Figures marked as open. |
| v1.0 | 13 Nov 2026 | Basis for coding mode 1. Plain-language walk-through added. "World" renamed "episode". Individual and aggregate scoring kept as two explicitly named options, compared side by side. Mode 1 fully specified, with environments for the first sweep, build plan and tests. Every remaining choice written into the decision register as decided or proposed. Modes 2 and 3 in outline, with assessment-rule tables and open issues, including separate error types. |
| v0.5 | 13 Nov 2026 | Junk inputs dropped; modes 1–3 described; baseline as rules alone and in pairs; one episode per generation; N_C, N_D; "iGSS agent". |
| v0.4 | 13 Nov 2026 | Canonical rule marked in the output instead of "target"; environment = parameters only; symbols table; agent definition. |
| v0.3 | 13 Nov 2026 | First version as a proposed architecture: arithmetic expressions with constants {0, 1}; classification by evaluating compiled expressions; no Mesa; decision register. |
| v0.2 | 13 Nov 2026 | Separate DR and IR models; inputs coded so 1 favours cooperating; second-order assessment. |
| v0.1 | 13 Nov 2026 | First compilation, from the 28 Aug 2026 `norms.py`. |
