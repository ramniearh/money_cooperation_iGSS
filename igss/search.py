"""The evolution loop (SPEC §9, §14).

What evolves depends on the mode:
  mode 1 - the action rule (the assessment rule is the environment's base rule);
  mode 2 - the assessment rule (the action rule is the canonical one);
  mode 3 - both, as a pair of expressions.
Each member of the population is a "genome": a list holding one expression
per evolved rule.

DEAP's operators draw from Python's global `random` module, which is seeded
once per run with the run seed. Episodes draw from generators of their own
(episode.make_rng), so the two streams never mix.
"""

import copy
import functools
import operator
import random
from collections import Counter

from deap import base, creator, gp, tools

from igss import config, rules, scoring
from igss.episode import Environment

EVOLVED_RULES = {1: ("action",), 2: ("assessment",), 3: ("action", "assessment")}

# DEAP keeps its classes in a global registry; create them once.
if not hasattr(creator, "ScoreMax"):
    creator.create("ScoreMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Genome"):
    creator.create("Genome", list, fitness=creator.ScoreMax)


def episode_seed(run_seed, generation, episode):
    """The text seed of episode `episode` of a generation (SPEC §9)."""
    return f"episode|{run_seed}|{generation}|{episode}"


def environment_of(cfg):
    return Environment(
        model=cfg["MODEL"],
        assessment_rule=config.assessment_rule(cfg),
        n_c=cfg["N_C"],
        n_d=cfg["N_D"],
        b=cfg["B"],
        c=cfg["C"],
        e_action=cfg["E_ACTION"],
        rounds=cfg["ROUNDS"],
    )


class Operators:
    """DEAP's operators, set from the configuration, one set per evolved rule."""

    def __init__(self, cfg):
        self.kinds = EVOLVED_RULES[cfg["MODE"]]
        self.cfg = cfg
        self.psets = {kind: rules.primitive_set(cfg["MODEL"], kind) for kind in self.kinds}
        # A child deeper than the limit is replaced by one of its parents. The
        # replacement arrives only in the value the operator returns.
        limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg["DEPTH_LIMIT"])
        self.mate = limit(gp.cxOnePoint)
        self.mutate = {kind: limit(self._mutation(kind)) for kind in self.kinds}

    def _mutation(self, kind):
        new_piece = functools.partial(gp.genHalfAndHalf, min_=self.cfg["MUT_DEPTH_MIN"],
                                      max_=self.cfg["MUT_DEPTH_MAX"])

        def mutate(expression):
            return gp.mutUniform(expression, expr=new_piece, pset=self.psets[kind])
        return mutate

    def new_genome(self):
        return creator.Genome(
            gp.PrimitiveTree(gp.genHalfAndHalf(self.psets[kind], min_=self.cfg["INIT_DEPTH_MIN"],
                                               max_=self.cfg["INIT_DEPTH_MAX"]))
            for kind in self.kinds)

    def which(self):
        """The expression an operator acts on: in mode 3, one of the two with equal chance."""
        return random.randrange(len(self.kinds)) if len(self.kinds) > 1 else 0

    def cross(self, first, second):
        k = self.which()
        first[k], second[k] = self.mate(first[k], second[k])

    def mutate_genome(self, genome):
        k = self.which()
        genome[k], = self.mutate[self.kinds[k]](genome[k])


@functools.lru_cache(maxsize=200_000)
def _bits_of_text(expression_text, model, kind):
    return rules.bits_of_expression(expression_text, model, kind)


def strategy_of(genome, kinds, model):
    """(action bits, assessment bits) of a genome; assessment bits are None in mode 1."""
    bits = {kind: _bits_of_text(str(expression), model, kind)
            for kind, expression in zip(kinds, genome)}
    return (bits.get("action", rules.CANONICAL_ACTION_BITS), bits.get("assessment"))


def score_population(strategies, env, cfg, generation):
    """Score one generation with the scoring option the configuration names."""
    seeds = [episode_seed(cfg["RUN_SEED"], generation, k)
             for k in range(cfg["EPISODES_PER_SCORE"])]
    if cfg["SCORING"] == "individual":
        return scoring.score_individual(strategies, env, seeds)
    if cfg["SCORING"] == "aggregate":
        return scoring.score_aggregate(strategies, env, seeds, cfg["N_IGSS"])
    raise ValueError(f"SCORING must be one of {scoring.SCORING_OPTIONS}")


def run_evolution(cfg):
    """One run: G generations. Returns one record per generation."""
    config.validate(cfg)
    env = environment_of(cfg)
    model = cfg["MODEL"]
    base_bits = rules.ASSESSMENT_BITS[env.assessment_rule]
    operators = Operators(cfg)
    kinds = operators.kinds

    random.seed(cfg["RUN_SEED"])
    population = [operators.new_genome() for _ in range(cfg["POP_SIZE"])]
    records = []

    for generation in range(1, cfg["GENERATIONS"] + 1):
        # 1. Classify every genome by its compiled expressions.
        strategies = [strategy_of(genome, kinds, model) for genome in population]

        # 2. Score, then subtract the size penalty to get the selection score.
        scores, summary = score_population(strategies, env, cfg, generation)
        sizes = [sum(len(expression) for expression in genome) for genome in population]
        for genome, score, size in zip(population, scores, sizes):
            genome.fitness.values = (score - cfg["PARSIMONY_TAX"] * size,)

        # 3. Log. Strategies are counted by class (SPEC §14.3); where the
        #    assessment rule is not evolved, the base rule is the one in force.
        effective = [(action, base_bits if assessment is None else assessment)
                     for action, assessment in strategies]
        ids = [rules.strategy_id(action, assessment) for action, assessment in effective]
        name_of = {strategy_id: rules.literature_name(model, action, assessment)
                   for strategy_id, (action, assessment) in zip(ids, effective)}
        named = {strategy_id for strategy_id, (action, assessment) in zip(ids, effective)
                 if rules.is_named_reciprocity(model, action, assessment)}
        counts = Counter(ids)
        top = max(counts.values())
        most_common = sorted(strategy_id for strategy_id in counts if counts[strategy_id] == top)
        records.append({
            "generation": generation,
            "counts": dict(sorted(counts.items())),
            "names": name_of,
            "most_common": most_common,
            "most_common_count": top,
            "most_common_is_named_reciprocity": [s in named for s in most_common],
            "mean_size": sum(sizes) / len(sizes),
            "summary": summary,
            "rules": [
                {
                    "rule_index": i,
                    "strategy": ids[i],
                    "strategy_name": name_of[ids[i]],
                    **{kind + "_expression": str(expression)
                       for kind, expression in zip(kinds, genome)},
                    "size": sizes[i],
                    "height": max(expression.height for expression in genome),
                    "score": scores[i],
                    "selection_score": genome.fitness.values[0],
                }
                for i, genome in enumerate(population)
            ],
        })
        if generation == cfg["GENERATIONS"]:
            break  # the final generation is scored and logged, not bred

        # 4. Selection: P tournaments (draws with replacement; among equal
        #    best the first drawn is kept).
        parents = tools.selTournament(population, len(population), cfg["TOURNAMENT_SIZE"])
        children = [copy.deepcopy(parent) for parent in parents]
        # 5. Crossover: parents paired in order.
        for i in range(0, len(children) - 1, 2):
            if random.random() < cfg["CXPB"]:
                operators.cross(children[i], children[i + 1])
        # 6. Mutation (7. the depth limit is applied inside both operators).
        for child in children:
            if random.random() < cfg["MUTPB"]:
                operators.mutate_genome(child)
        # 8. The children are the next generation; nothing is carried over unchanged.
        population = children

    return records
