"""
igss_money_model.py
====================
Prototype implementation of the Inverse Generative Social Science (iGSS) model
described in:

    Ferraciolli & Renzini, "The Institutionalization of Money:
    Advancing Fundamental Institutional Theory Through Inverse Generative
    Social Science" (AAMAS COINE, short paper).

Architecture
------------
* Mesa  – spatial agent-based simulation (Sugarscape-style grid)
* DEAP  – genetic programming to evolve agent decision rules

Three scenarios from the paper are implemented:
    Scenario 1  –  "Methodologically individualist iGSS"
                   Agents have only primitive unilateral GIVE transfers.
    Scenario 2  –  "Institutional iGSS – Exchange"
                   Agents also have SPOT_EXCHANGE (conditional bilateral swap).
    Scenario 3  –  "Institutional iGSS – Debt"
                   Agents also have INTERPERSONAL_DEBT (IOU issuance/clearing).

The fitness metric (macro target) is:
    Σ over all agents of the number of *distinct* good types consumed per step.
This operationalises the paper's "high level of division of labour" criterion.

Dependencies
------------
    pip install mesa deap numpy

Usage
-----
    python igss_money_model_claude_sonnet.py

    Adjust SCENARIO (1, 2, or 3), N_GENERATIONS, and GRID_SIZE at the top.
"""

import random
import operator
import copy
from collections import defaultdict

import numpy as np

# ── Mesa ──────────────────────────────────────────────────────────────────────
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# ── DEAP ──────────────────────────────────────────────────────────────────────
from deap import base, creator, gp, tools, algorithms

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

SCENARIO     = 3          # 1 = individualist, 2 = exchange, 3 = debt
N_GOODS      = 5          # number of distinct good types
GRID_SIZE    = 10         # square grid side length
N_AGENTS     = 10         # number of economic agents
INV_CAPACITY = 6          # max goods per inventory slot per agent
INTERACTION_RANGE = 2     # Manhattan distance within which agents can interact
N_GENERATIONS = 999        # GP evolutionary generations
POP_SIZE      = 40        # GP population size (per-agent rule pool)
STEPS_PER_EVAL = 20       # simulation steps per fitness evaluation
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# GOOD TYPES
# ═════════════════════════════════════════════════════════════════════════════

GOOD_TYPES = [f"G{i}" for i in range(N_GOODS)]   # ["G0", "G1", "G2", "G3"]


# ═════════════════════════════════════════════════════════════════════════════
# DECISION-RULE LANGUAGE  (GP primitives)
# ═════════════════════════════════════════════════════════════════════════════
# Each rule is a callable(agent, neighbours) → action_dict | None
# We encode rules as DEAP GP trees that return a float "propensity score"
# for each candidate action. The action with the highest score is taken.
#
# Terminals and primitives operate on a flat feature vector derived from the
# agent's local state:
#   [own_inv_G0, …, own_inv_G{N-1},
#    nb_inv_G0,  …, nb_inv_G{N-1},
#    own_n_debts, own_n_ious,
#    macro_div_labour]                    ← global state variable

N_FEATURES = N_GOODS * 2 + 2 + 1   # 13 floats for N_GOODS=4

def _safe_div(a, b):
    return a / b if abs(b) > 1e-8 else 0.0

def _safe_sub(a, b):  return a - b
def _safe_add(a, b):  return a + b
def _safe_mul(a, b):  return a * b
def _gt(a, b):        return float(a > b)
def _lt(a, b):        return float(a < b)
def _eq(a, b):        return float(abs(a - b) < 0.5)
def _if(cond, a, b):  return a if cond > 0.5 else b
def _neg(a):          return -a
def _abs(a):          return abs(a)

def build_pset():
    """Build the DEAP PrimitiveSet for the rule language."""
    pset = gp.PrimitiveSet("RULE", arity=N_FEATURES)

    pset.addPrimitive(_safe_add, 2, name="add")
    pset.addPrimitive(_safe_sub, 2, name="sub")
    pset.addPrimitive(_safe_mul, 2, name="mul")
    pset.addPrimitive(_safe_div, 2, name="div")
    pset.addPrimitive(_gt,       2, name="gt")
    pset.addPrimitive(_lt,       2, name="lt")
    pset.addPrimitive(_eq,       2, name="eq")
    pset.addPrimitive(_if,       3, name="if_")
    pset.addPrimitive(_neg,      1, name="neg")
    pset.addPrimitive(_abs,      1, name="abs_")

    # Constants
    for c in [0.0, 0.5, 1.0, 2.0, 5.0]:
        pset.addTerminal(c)

    # Rename feature arguments to readable names
    feature_names = (
        [f"own_{g}" for g in GOOD_TYPES] +
        [f"nb_{g}"  for g in GOOD_TYPES] +
        ["own_debts", "own_ious", "macro_dol"]
    )
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f"ARG{i}": name})

    return pset


PSET = build_pset()

# DEAP fitness & individual setup
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                   pset=PSET)


# ═════════════════════════════════════════════════════════════════════════════
# ACTION TYPES
# ═════════════════════════════════════════════════════════════════════════════

class Action:
    """Base class for agent actions."""
    pass

class GiveAction(Action):
    """Unilateral transfer: give one unit of `good` to `target`."""
    def __init__(self, good, target_id):
        self.good      = good
        self.target_id = target_id

class SpotExchangeAction(Action):
    """Propose bilateral swap: give `good_out`, receive `good_in` from `target`."""
    def __init__(self, good_out, good_in, target_id):
        self.good_out  = good_out
        self.good_in   = good_in
        self.target_id = target_id

class DebtAction(Action):
    """Issue an IOU: receive `good` from `target`, promise to repay later."""
    def __init__(self, good, target_id):
        self.good      = good
        self.target_id = target_id

class ClearDebtAction(Action):
    """Clear an existing debt to `creditor_id` by transferring `good`."""
    def __init__(self, good, creditor_id):
        self.good       = good
        self.creditor_id = creditor_id


# ═════════════════════════════════════════════════════════════════════════════
# AGENT
# ═════════════════════════════════════════════════════════════════════════════

class EconomicAgent(Agent):
    """
    An institutional agent that:
      - produces a primary good at its production site
      - holds an inventory of multiple goods (bounded)
      - follows a GP-evolved decision rule each step
      - can GIVE, SPOT_EXCHANGE (scenario ≥2), DEBT/CLEAR (scenario 3)
    """

    def __init__(self, unique_id, model, primary_good, rule_fn):
        super().__init__(model)
        self.unique_id    = unique_id
        self.primary_good = primary_good      # good type this agent produces
        self.rule_fn      = rule_fn           # compiled GP function

        # Inventory: dict good → count
        self.inventory = defaultdict(int)
        self.inventory[primary_good] = INV_CAPACITY  # start with own good

        # Debt ledger (scenario 3)
        self.debts_owed  = {}   # creditor_id → (good_type, amount)
        self.ious_held   = {}   # debtor_id   → (good_type, amount)

        # Consumption log (for fitness)
        self.consumed_this_step = set()

    # ── Production ────────────────────────────────────────────────────────────

    def produce(self):
        """Replenish primary good up to capacity (spatially situated production)."""
        deficit = INV_CAPACITY - self.inventory[self.primary_good]
        if deficit > 0:
            self.inventory[self.primary_good] += max(1, deficit // 2)

    # ── Consumption ───────────────────────────────────────────────────────────

    def consume(self):
        """Consume one unit of every good type currently held."""
        self.consumed_this_step = set()
        for g in GOOD_TYPES:
            if self.inventory[g] > 0:
                self.inventory[g] -= 1
                self.consumed_this_step.add(g)

    # ── Feature vector ────────────────────────────────────────────────────────

    def feature_vector(self, neighbour=None):
        """
        Build the N_FEATURES-length input to the GP rule function.
        If no neighbour is provided, neighbour features are zeroed.
        """
        own_inv  = [float(self.inventory[g]) for g in GOOD_TYPES]
        nb_inv   = ([float(neighbour.inventory[g]) for g in GOOD_TYPES]
                    if neighbour else [0.0] * N_GOODS)
        own_dbt  = float(len(self.debts_owed))
        own_iou  = float(len(self.ious_held))
        macro    = float(self.model.division_of_labour)
        return own_inv + nb_inv + [own_dbt, own_iou, macro]

    # ── Decision & Action ─────────────────────────────────────────────────────

    def decide(self):
        """
        Evaluate rule against each neighbour and each possible action type,
        choose the highest-scoring (agent, action) pair.
        """
        neighbours = self._get_neighbours()
        if not neighbours:
            return

        best_score  = -np.inf
        best_action = None

        for nb in neighbours:
            fv = self.feature_vector(nb)

            # Score GIVE actions
            for g in GOOD_TYPES:
                if self.inventory[g] > 0:
                    score = self._eval_rule(fv)
                    # bias score toward goods where neighbour has less
                    bias  = max(0, self.inventory[g] - nb.inventory[g])
                    score += bias * 0.01
                    if score > best_score:
                        best_score  = score
                        best_action = GiveAction(g, nb.unique_id)

            # Scenario 2+: SPOT EXCHANGE
            if self.model.sim_scenario >= 2:
                for g_out in GOOD_TYPES:
                    for g_in in GOOD_TYPES:
                        if g_out == g_in:
                            continue
                        if self.inventory[g_out] > 0 and nb.inventory[g_in] > 0:
                            score = self._eval_rule(fv) + 0.05  # exchange premium
                            if score > best_score:
                                best_score  = score
                                best_action = SpotExchangeAction(g_out, g_in, nb.unique_id)

            # Scenario 3: DEBT
            if self.model.sim_scenario >= 3:
                for g in GOOD_TYPES:
                    if nb.inventory[g] > 0 and nb.unique_id not in self.debts_owed:
                        score = self._eval_rule(fv) - 0.02  # slight debt aversion
                        if score > best_score:
                            best_score  = score
                            best_action = DebtAction(g, nb.unique_id)

                for creditor_id, (g, _) in list(self.debts_owed.items()):
                    if self.inventory[g] > 0:
                        score = self._eval_rule(fv) + 0.10  # debt-clearing premium
                        if score > best_score:
                            best_score  = score
                            best_action = ClearDebtAction(g, creditor_id)

        if best_action:
            self._execute(best_action)

    def _eval_rule(self, feature_vector):
        try:
            return float(self.rule_fn(*feature_vector))
        except Exception:
            return 0.0

    def _get_neighbours(self):
        """Return agents within INTERACTION_RANGE (Manhattan distance)."""
        x, y = self.pos
        result = []
        for dx in range(-INTERACTION_RANGE, INTERACTION_RANGE + 1):
            for dy in range(-INTERACTION_RANGE, INTERACTION_RANGE + 1):
                if dx == 0 and dy == 0:
                    continue
                cell = self.model.grid.get_cell_list_contents(
                    [(x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE])
                result.extend([a for a in cell if isinstance(a, EconomicAgent)
                                and a.unique_id != self.unique_id])
        return result

    def _get_agent_by_id(self, uid):
        return self.model.agent_map.get(uid)

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute(self, action):
        if isinstance(action, GiveAction):
            target = self._get_agent_by_id(action.target_id)
            if target and self.inventory[action.good] > 0:
                self.inventory[action.good]  -= 1
                target.inventory[action.good] += 1

        elif isinstance(action, SpotExchangeAction):
            target = self._get_agent_by_id(action.target_id)
            if (target
                    and self.inventory[action.good_out] > 0
                    and target.inventory[action.good_in] > 0):
                self.inventory[action.good_out]  -= 1
                target.inventory[action.good_in] -= 1
                self.inventory[action.good_in]   += 1
                target.inventory[action.good_out] += 1

        elif isinstance(action, DebtAction):
            target = self._get_agent_by_id(action.target_id)
            if target and target.inventory[action.good] > 0:
                target.inventory[action.good] -= 1
                self.inventory[action.good]   += 1
                self.debts_owed[action.target_id]  = (action.good, 1)
                target.ious_held[self.unique_id]   = (action.good, 1)

        elif isinstance(action, ClearDebtAction):
            creditor = self._get_agent_by_id(action.creditor_id)
            if (creditor
                    and self.inventory[action.good] > 0
                    and action.creditor_id in self.debts_owed):
                self.inventory[action.good]    -= 1
                creditor.inventory[action.good] += 1
                del self.debts_owed[action.creditor_id]
                creditor.ious_held.pop(self.unique_id, None)

    # ── Mesa step ─────────────────────────────────────────────────────────────

    def step(self):
        self.produce()
        self.decide()
        self.consume()


# ═════════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════════

class MoneyEmergenceModel(Model):
    """
    Sugarscape-style grid model.
    Agents are fixed at production sites (one dominant good per region).
    The macro target is maximising division of labour (variety of goods consumed).
    """

    def __init__(self, rule_population, scenario=SCENARIO):
        super().__init__()
        self.sim_scenario      = scenario
        self.grid              = MultiGrid(GRID_SIZE, GRID_SIZE, torus=True)
        self.division_of_labour = 0.0   # global macro-state variable
        self.agent_map         = {}

        self.datacollector = DataCollector(
            model_reporters={"Division_of_Labour": "division_of_labour"},
            agent_reporters={"Consumed": lambda a: len(a.consumed_this_step)}
        )

        self._spawn_agents(rule_population)

    def _spawn_agents(self, rule_population):
        """
        Place N_AGENTS agents on the grid, assigning each a primary good
        according to their spatial region (divides grid into N_GOODS strips).
        """
        positions = random.sample(
            [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)],
            N_AGENTS)

        for idx, pos in enumerate(positions):
            # Assign primary good by x-region
            primary_good = GOOD_TYPES[pos[0] * N_GOODS // GRID_SIZE]
            rule_fn      = rule_population[idx % len(rule_population)]
            agent        = EconomicAgent(idx, self, primary_good, rule_fn)
            self.grid.place_agent(agent, pos)
            self.agent_map[idx] = agent

    def step(self):
        self.agents.shuffle_do("step")
        self._update_macro()
        self.datacollector.collect(self)

    def _update_macro(self):
        """
        Compute division of labour:
            mean over agents of (# distinct good types consumed this step)
            normalised by N_GOODS.
        """
        scores = [len(a.consumed_this_step) / N_GOODS
                  for a in self.agents]
        self.division_of_labour = float(np.mean(scores)) if scores else 0.0

    def run(self, steps=STEPS_PER_EVAL):
        for _ in range(steps):
            self.step()
        return self.division_of_labour


# ═════════════════════════════════════════════════════════════════════════════
# iGSS FITNESS EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def compile_individual(ind):
    """Compile a DEAP GP tree into a callable Python function."""
    return gp.compile(ind, PSET)


def evaluate_rule_population(population):
    """
    iGSS fitness: run the full ABM with this rule population and return
    the mean division-of-labour score.

    This is the macro-to-micro inversion step: we measure a macro target
    (DoL) and score the micro rule-set that produced it.
    """
    rule_fns = [compile_individual(ind) for ind in population]
    model    = MoneyEmergenceModel(rule_fns, scenario=SCENARIO)
    dol      = model.run(steps=STEPS_PER_EVAL)
    return (dol,)


# ═════════════════════════════════════════════════════════════════════════════
# GENETIC PROGRAMMING SETUP (DEAP)
# ═════════════════════════════════════════════════════════════════════════════

def build_toolbox():
    toolbox = base.Toolbox()

    # Tree generation
    toolbox.register("expr",       gp.genHalfAndHalf, pset=PSET, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate",  evaluate_rule_population_wrapper)
    toolbox.register("select",    tools.selTournament, tournsize=3)
    toolbox.register("mate",      gp.cxOnePoint)
    toolbox.register("expr_mut",  gp.genFull, min_=0, max_=2)
    toolbox.register("mutate",    gp.mutUniform, expr=toolbox.expr_mut, pset=PSET)

    # Bloat control
    toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"),
                                              max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                              max_value=8))
    return toolbox


# NOTE: DEAP's eaSimple evaluates individuals, not populations.
# We wrap here so that an *individual* tree is evaluated by running it
# as the shared rule for all agents — a "representative agent" baseline.
# In the relational scenarios the whole population is passed; here we
# broadcast a single rule to keep DEAP's interface consistent.

def evaluate_rule_population_wrapper(individual):
    """Wrap a single GP tree as a uniform rule for all agents."""
    return evaluate_rule_population([individual] * N_AGENTS)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN  –  iGSS EVOLUTIONARY LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_igss():
    print("=" * 60)
    print(f"iGSS Money Emergence Model")
    print(f"Scenario : {SCENARIO}  ({'individualist' if SCENARIO==1 else 'exchange' if SCENARIO==2 else 'debt'})")
    print(f"Agents   : {N_AGENTS}   Grid: {GRID_SIZE}×{GRID_SIZE}   Goods: {N_GOODS}")
    print(f"GP       : pop={POP_SIZE}  gen={N_GENERATIONS}  steps/eval={STEPS_PER_EVAL}")
    print("=" * 60)

    toolbox = build_toolbox()

    population = toolbox.population(n=POP_SIZE)
    hof        = tools.HallOfFame(3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg",  np.mean)
    stats.register("max",  np.max)
    stats.register("min",  np.min)

    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.6, mutpb=0.3,
        ngen=N_GENERATIONS,
        stats=stats, halloffame=hof,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    best = hof[0]
    print(f"Best fitness (Division of Labour): {best.fitness.values[0]:.4f}")
    print(f"\nBest rule tree:\n{str(best)}")

    # ── Interpret emergent institution ────────────────────────────────────────
    _interpret_institution(best)

    # ── Run final simulation with best rule & collect time-series ────────────
    rule_fn   = compile_individual(best)
    model     = MoneyEmergenceModel([rule_fn] * N_AGENTS, scenario=SCENARIO)
    model.run(steps=50)
    df        = model.datacollector.get_model_vars_dataframe()

    print("\nDivision-of-Labour time series (last 10 steps):")
    print(df.tail(10).to_string())

    return hof, logbook, df


def _interpret_institution(individual):
    """
    Heuristic institutional interpretation of the best-evolved rule,
    following the paper's definition:
        'any set of decision rules shared across agents, regarding commodities
         or other agents, and enabling either the deferral of economic
         transactions in time and/or the structuring of more complex chains
         of production in space.'
    """
    rule_str = str(individual)
    print("\n── Institutional Interpretation ──────────────────────────────")

    if SCENARIO == 1:
        print("Scenario 1 (individualist): agents use unilateral GIVE only.")
        print("  → Rules likely converge on simple redistribution heuristics.")
        print("  → Limited institutional complexity expected (paper confirms).")

    elif SCENARIO == 2:
        print("Scenario 2 (exchange): SPOT_EXCHANGE primitives active.")
        has_exchange_bias = any(k in rule_str for k in ["nb_G", "gt", "lt"])
        if has_exchange_bias:
            print("  ✓ Rule references neighbour inventories — bilateral awareness.")
            print("  → Possible emergence of COMMODITY MONEY: one good preferred")
            print("    as a medium because it maximises swap opportunities.")
        else:
            print("  → Rule does not strongly reference neighbour inventory.")
            print("    Weak bilateral coordination detected.")

    elif SCENARIO == 3:
        print("Scenario 3 (debt): INTERPERSONAL_DEBT & CLEAR_DEBT active.")
        has_debt_terms = any(k in rule_str for k in ["own_debts", "own_ious"])
        if has_debt_terms:
            print("  ✓ Rule references debt state — intertemporal awareness.")
            print("  → Possible emergence of CREDIT MONEY: IOUs as deferred")
            print("    claims enabling time-space stretching of production chains.")
        else:
            print("  → Rule does not reference debt state directly.")

    fitness = individual.fitness.values[0]
    if fitness > 0.5:
        print(f"\n  DoL = {fitness:.3f}  (>0.5) → Substantial division of labour.")
        print("  Emergent rule-set qualifies as a MESO-LEVEL INSTITUTION")
        print("  in the paper's sense: shared rule enabling coordinated surplus")
        print("  distribution across spatially separated agents.")
    elif fitness > 0.25:
        print(f"\n  DoL = {fitness:.3f}  (>0.25) → Partial coordination achieved.")
        print("  Proto-institutional behaviour; not yet a stable institution.")
    else:
        print(f"\n  DoL = {fitness:.3f}  → Low coordination. Consistent with paper's")
        print("  finding that asocial agents fail to generate rich institutions.")

    print("─" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    hof, logbook, df = run_igss()