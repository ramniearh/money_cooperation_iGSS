"""The two scoring options (SPEC §7). They are separate functions on purpose.

Both take the strategies of one generation and the text seeds of the
generation's episodes, and return one score per strategy (payoff per round,
without any size penalty) together with a summary of the episodes played.

A strategy is a pair (action bits, assessment bits). The assessment bits are
None when the assessment rule is not evolved (mode 1): the environment's base
rule then applies.
"""

from collections import Counter

from igss import rules
from igss.episode import average_summaries, make_agents, make_rng, play_episode, summarise

SCORING_OPTIONS = ("individual", "aggregate")


def _assessment(bits):
    return None if bits is None else rules.rule_from_bit_string(bits)


def score_individual(strategies, env, seeds):
    """All strategies share each episode, one per iGSS agent.

    A strategy's score is the payoff per round of the agent carrying it,
    averaged over the episodes. Each iGSS agent applies its own assessment
    rule when it is the partner; background agents apply the base rule.
    """
    totals = [0.0] * len(strategies)
    summaries = []
    for seed_text in seeds:
        agents = make_agents(
            [rules.rule_from_bit_string(action) for action, _ in strategies],
            env.n_c, env.n_d,
            assessment_rules=[_assessment(assessment) for _, assessment in strategies],
        )
        result = play_episode(agents, env, make_rng(seed_text))
        for i in range(len(strategies)):
            totals[i] += result.payoff[i] / env.rounds
        summaries.append(summarise(result, agents, env))
    scores = [total / len(seeds) for total in totals]
    return scores, average_summaries(summaries)


def score_aggregate(strategies, env, seeds, n_igss):
    """Each strategy gets episodes of its own, in which all `n_igss` iGSS agents carry it.

    A strategy's score is the mean payoff per round of those agents, averaged
    over the episodes. Its assessment rule is the one rule of the episode:
    every agent applies it, background agents included. Every strategy uses
    the same seeds, so strategies of the same class get exactly the same score
    and are simulated only once.
    """
    classes = [rules.strategy_class(action, assessment) for action, assessment in strategies]
    representative = dict(zip(classes, strategies))
    score_of, summary_of = {}, {}
    for cls, (action, assessment) in representative.items():
        summaries = []
        for seed_text in seeds:
            agents = make_agents(
                [rules.rule_from_bit_string(action)] * n_igss, env.n_c, env.n_d,
                assessment_rules=[_assessment(assessment)] * n_igss,
                background_assessment=_assessment(assessment),
            )
            result = play_episode(agents, env, make_rng(seed_text))
            summaries.append(summarise(result, agents, env))
        summary_of[cls] = average_summaries(summaries)
        score_of[cls] = summary_of[cls]["payoff_igss"]
    scores = [score_of[cls] for cls in classes]
    # The generation's summary averages over the strategies, so each class
    # weighs as many times as there are strategies in it.
    counts = Counter(classes)
    summary = average_summaries([summary_of[c] for c in counts], [counts[c] for c in counts])
    return scores, summary
