"""The evolution loop for mode 1: the action rule evolves (SPEC §9).

DEAP's operators draw from Python's global `random` module, which is seeded
once per run with the run seed. Episodes draw from generators of their own
(episode.make_rng), so the two streams never mix.
"""

import functools
import operator
import random
from collections import Counter

from deap import base, creator, gp, tools

from igss import config, rules, scoring
from igss.episode import Environment

# DEAP keeps its classes in a global registry; create them once.
if not hasattr(creator, "ScoreMax"):
    creator.create("ScoreMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Expression"):
    creator.create("Expression", gp.PrimitiveTree, fitness=creator.ScoreMax)


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


def make_toolbox(cfg):
    """DEAP's operators, set from the configuration."""
    pset = rules.primitive_set(cfg["MODEL"])
    toolbox = base.Toolbox()
    toolbox.register("new_expression", gp.genHalfAndHalf, pset=pset,
                     min_=cfg["INIT_DEPTH_MIN"], max_=cfg["INIT_DEPTH_MAX"])
    toolbox.register("individual", tools.initIterate, creator.Expression, toolbox.new_expression)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=cfg["TOURNAMENT_SIZE"])
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("new_piece", gp.genHalfAndHalf,
                     min_=cfg["MUT_DEPTH_MIN"], max_=cfg["MUT_DEPTH_MAX"])
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.new_piece, pset=pset)
    # A child deeper than the limit is replaced by one of its parents.
    limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg["DEPTH_LIMIT"])
    toolbox.decorate("mate", limit)
    toolbox.decorate("mutate", limit)
    return toolbox


@functools.lru_cache(maxsize=100_000)
def _bits_of_text(expression_text, model):
    return rules.action_bits_of_expression(expression_text, model)


def score_population(bit_strings, env, cfg, generation):
    """Score one generation with the scoring option the configuration names."""
    seeds = [episode_seed(cfg["RUN_SEED"], generation, k)
             for k in range(cfg["EPISODES_PER_SCORE"])]
    if cfg["SCORING"] == "individual":
        return scoring.score_individual(bit_strings, env, seeds)
    if cfg["SCORING"] == "aggregate":
        return scoring.score_aggregate(bit_strings, env, seeds, cfg["N_IGSS"])
    raise ValueError(f"SCORING must be one of {scoring.SCORING_OPTIONS}")


def most_common_bits(counts):
    """The most common bit strings of a generation (several if they tie)."""
    top = max(counts.values())
    return [bits for bits in rules.ALL_ACTION_BITS if counts[bits] == top]


def run_evolution(cfg):
    """One run: G generations. Returns one record per generation."""
    config.validate(cfg)
    env = environment_of(cfg)
    toolbox = make_toolbox(cfg)
    model = cfg["MODEL"]

    random.seed(cfg["RUN_SEED"])
    population = toolbox.population(n=cfg["POP_SIZE"])
    records = []

    for generation in range(1, cfg["GENERATIONS"] + 1):
        # 1. Classify every rule by its compiled expression.
        texts = [str(expression) for expression in population]
        bit_strings = [_bits_of_text(text, model) for text in texts]

        # 2. Score, then subtract the size penalty to get the selection score.
        scores, summary = score_population(bit_strings, env, cfg, generation)
        for expression, score in zip(population, scores):
            expression.fitness.values = (score - cfg["PARSIMONY_TAX"] * len(expression),)

        # 3. Log.
        counts = Counter(bit_strings)
        records.append({
            "generation": generation,
            "counts": {bits: counts[bits] for bits in rules.ALL_ACTION_BITS},
            "most_common": most_common_bits(counts),
            "mean_size": sum(len(expression) for expression in population) / len(population),
            "summary": summary,
            "rules": [
                {
                    "rule_index": i,
                    "expression": text,
                    "size": len(expression),
                    "height": expression.height,
                    "bit_string": bits,
                    "score": score,
                    "selection_score": expression.fitness.values[0],
                }
                for i, (expression, text, bits, score)
                in enumerate(zip(population, texts, bit_strings, scores))
            ],
        })
        if generation == cfg["GENERATIONS"]:
            break  # the final generation is scored and logged, not bred

        # 4. Selection: P tournaments.
        children = [toolbox.clone(parent) for parent in toolbox.select(population, len(population))]
        # 5. Crossover: parents paired in order.
        #    The operators' return values must be used: when a child is too
        #    deep, DEAP's staticLimit returns a parent in its place and leaves
        #    the too-deep child where it was.
        for i in range(0, len(children) - 1, 2):
            if random.random() < cfg["CXPB"]:
                children[i], children[i + 1] = toolbox.mate(children[i], children[i + 1])
        # 6. Mutation (7. the depth limit is applied inside mate and mutate).
        for i in range(len(children)):
            if random.random() < cfg["MUTPB"]:
                children[i], = toolbox.mutate(children[i])
        # 8. The children are the next generation; nothing is carried over unchanged.
        population = children

    return records
