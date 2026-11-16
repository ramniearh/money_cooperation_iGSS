"""Tests for igss/scoring.py (SPEC §11, step 4)."""

import pytest

from igss import baseline, rules, scoring
from igss.episode import Environment, make_agents, make_rng, play_episode


def environment(model):
    return Environment(
        model=model,
        assessment_rule={"DR": "what_donor_did", "IR": "did_not_refuse_a_one"}[model],
        n_c=3, n_d=3, b=2, c=1, e_action=0.05, rounds=30,
    )


@pytest.mark.parametrize("model", ["DR", "IR"])
@pytest.mark.parametrize("mode", [1, 2, 3])
def test_aggregate_score_equals_alone_run_payoff_for_the_same_seed(model, mode):
    env = environment(model)
    for strategy in baseline.strategies_of_mode(mode):
        scores, _ = scoring.score_aggregate([strategy], env, seeds=["seed"], n_igss=6)
        alone = baseline.run_alone(strategy, env, n_igss=6, seed_text="seed")
        assert scores[0] == pytest.approx(alone["payoff_igss"])


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_same_strategy_same_aggregate_score(model):
    env = environment(model)
    strategies = [("10", None), ("11", None), ("10", None), ("00", None), ("10", None)]
    scores, _ = scoring.score_aggregate(strategies, env, seeds=["a", "b"], n_igss=6)
    assert scores[0] == scores[2] == scores[4]


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_unconditional_strategies_score_the_same_whatever_their_assessment_rule(model):
    # SPEC §14.3: with action rule `11` or `00` the stored value is never used.
    env = environment(model)
    strategies = [("11", bits) for bits in rules.ALL_ASSESSMENT_BITS]
    scores, _ = scoring.score_aggregate(strategies, env, seeds=["a"], n_igss=6)
    assert len(set(scores)) == 1


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_individual_scores_are_the_agents_payoffs_per_round(model):
    env = environment(model)
    bit_strings = ["10", "11", "01", "00"]
    scores, _ = scoring.score_individual([(b, None) for b in bit_strings], env, seeds=["seed"])
    agents = make_agents([rules.rule_from_bit_string(b) for b in bit_strings], env.n_c, env.n_d)
    result = play_episode(agents, env, make_rng("seed"))
    assert scores == [result.payoff[i] / env.rounds for i in range(4)]


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_more_episodes_are_averaged(model):
    env = environment(model)
    strategies = [("10", None), ("00", None)]
    one_a, _ = scoring.score_individual(strategies, env, seeds=["a"])
    one_b, _ = scoring.score_individual(strategies, env, seeds=["b"])
    both, _ = scoring.score_individual(strategies, env, seeds=["a", "b"])
    assert both == pytest.approx([(a + b) / 2 for a, b in zip(one_a, one_b)])


@pytest.mark.parametrize("model, base", [("DR", "1100"), ("IR", "1101")])
def test_individual_scoring_with_base_rule_for_all_equals_mode_1(model, base):
    env = environment(model)
    mode_1, _ = scoring.score_individual([("10", None), ("00", None)], env, seeds=["a"])
    evolved, _ = scoring.score_individual([("10", base), ("00", base)], env, seeds=["a"])
    assert mode_1 == evolved


@pytest.mark.parametrize("model", ["DR", "IR"])
@pytest.mark.parametrize("action", ["11", "00"])
def test_unconditional_classes_are_simulated_with_the_base_rule(model, action):
    # SPEC §14.3: a class whose action rule never reads the stored value is simulated with the
    # base assessment rule, so what it logs (read counts included) is what mode 1 logs, whatever
    # assessment rules its members carry and in whatever order they come.
    env = environment(model)
    _, as_in_mode_1 = scoring.score_aggregate([(action, None)], env, seeds=["a"], n_igss=6)
    members = [(action, bits) for bits in ("0000", "1111", "0011")]
    for strategies in (members, members[::-1]):
        _, summary = scoring.score_aggregate(strategies, env, seeds=["a"], n_igss=6)
        assert summary == as_in_mode_1
