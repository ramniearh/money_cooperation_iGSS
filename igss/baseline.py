"""The baseline: strategies assigned by hand, no search (SPEC §8).

Every iGSS agent uses the same strategy, one episode is played, and this is
repeated over seeds. It is exactly what aggregate scoring measures, so it
shows in advance how each strategy performs in an environment and which
strategies cannot be told apart there.
"""

from igss import rules
from igss.episode import SUMMARY_COLUMNS, make_agents, make_rng, play_episode, summarise


def baseline_seed(seed_index):
    """The text seed of baseline episode `seed_index` (SPEC §9)."""
    return f"baseline|{seed_index}"


def strategies_of_mode(mode):
    """One strategy per class the mode can produce (SPEC §14).

    Mode 1: the 4 action rules with the base assessment rule (None).
    Mode 2: the canonical action rule with each of the 16 assessment rules.
    Mode 3: every action rule with every assessment rule, one per class.
    """
    actions = [rules.CANONICAL_ACTION_BITS] if mode == 2 else rules.ALL_ACTION_BITS
    assessments = [None] if mode == 1 else rules.ALL_ASSESSMENT_BITS
    by_class = {}
    for action in actions:
        for assessment in assessments:
            by_class.setdefault(rules.strategy_class(action, assessment), (action, assessment))
    return list(by_class.values())


def run_alone(strategy, env, n_igss, seed_text):
    """One episode in which all iGSS agents use one strategy. Returns its summary.

    The action rule is the named predicate of rules.py. The strategy's
    assessment rule, if it has one, is applied by every agent.
    """
    action_bits, assessment_bits = strategy
    assessment = None if assessment_bits is None else rules.rule_from_bit_string(assessment_bits)
    agents = make_agents([rules.ACTION_RULE_OF_BITS[action_bits]] * n_igss, env.n_c, env.n_d,
                         assessment_rules=[assessment] * n_igss,
                         background_assessment=assessment)
    return summarise(play_episode(agents, env, make_rng(seed_text)), agents, env)


def baseline_rows(env, mode, n_igss, n_seeds):
    """Every baseline episode of an environment, as output rows (SPEC §10)."""
    base_bits = rules.ASSESSMENT_BITS[env.assessment_rule]
    for action_bits, assessment_bits in strategies_of_mode(mode):
        effective = base_bits if assessment_bits is None else assessment_bits
        for seed_index in range(n_seeds):
            summary = run_alone((action_bits, assessment_bits), env, n_igss,
                                baseline_seed(seed_index))
            yield {
                "strategy": rules.strategy_id(action_bits, effective),
                "strategy_name": rules.literature_name(env.model, action_bits, effective),
                "seed_index": seed_index,
                **summary,
            }


BASELINE_COLUMNS = ("strategy", "strategy_name", "seed_index") + SUMMARY_COLUMNS
