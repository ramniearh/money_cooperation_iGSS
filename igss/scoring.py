"""The two scoring options (SPEC §7). They are separate functions on purpose.

Both take the rules of one generation as bit strings and the text seeds of the
generation's episodes, and return one score per rule (payoff per round, without
any size penalty) together with a summary of the episodes played.
"""

from collections import Counter

from igss import rules
from igss.episode import IGSS, average_summaries, make_agents, make_rng, play_episode, summarise

SCORING_OPTIONS = ("individual", "aggregate")


def score_individual(bit_strings, env, seeds):
    """All rules share each episode, one rule per iGSS agent.

    A rule's score is the payoff per round of the agent carrying it, averaged
    over the episodes.
    """
    totals = [0.0] * len(bit_strings)
    summaries = []
    for seed_text in seeds:
        agents = make_agents([rules.rule_from_bit_string(b) for b in bit_strings], env.n_c, env.n_d)
        result = play_episode(agents, env, make_rng(seed_text))
        for i in range(len(bit_strings)):
            totals[i] += result.payoff[i] / env.rounds
        summaries.append(summarise(result, agents, env))
    scores = [total / len(seeds) for total in totals]
    return scores, average_summaries(summaries)


def score_aggregate(bit_strings, env, seeds, n_igss):
    """Each rule gets episodes of its own, in which all `n_igss` iGSS agents carry it.

    A rule's score is the mean payoff per round of those agents, averaged over
    the episodes. Every rule uses the same seeds, so rules with the same bit
    string get exactly the same score and are simulated only once.
    """
    score_of = {}
    summary_of = {}
    for bits in set(bit_strings):
        summaries = []
        for seed_text in seeds:
            agents = make_agents([rules.rule_from_bit_string(bits)] * n_igss, env.n_c, env.n_d)
            result = play_episode(agents, env, make_rng(seed_text))
            summaries.append(summarise(result, agents, env))
        summary_of[bits] = average_summaries(summaries)
        score_of[bits] = summary_of[bits]["payoff_igss"]
    scores = [score_of[bits] for bits in bit_strings]
    # The generation's summary averages over the rules, so each bit string
    # weighs as many times as there are rules with it.
    counts = Counter(bit_strings)
    summary = average_summaries([summary_of[b] for b in counts], [counts[b] for b in counts])
    return scores, summary
