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
def test_aggregate_score_equals_alone_run_payoff_for_the_same_seed(model):
    env = environment(model)
    for name, bits in rules.ACTION_BITS.items():
        scores, _ = scoring.score_aggregate([bits], env, seeds=["seed"], n_igss=6)
        alone = baseline.run_alone(name, env, n_igss=6, seed_text="seed")
        assert scores[0] == pytest.approx(alone["x"]["payoff"])


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_same_bit_string_same_aggregate_score(model):
    env = environment(model)
    scores, _ = scoring.score_aggregate(["10", "11", "10", "00", "10"], env,
                                        seeds=["a", "b"], n_igss=6)
    assert scores[0] == scores[2] == scores[4]


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_individual_scores_are_the_agents_payoffs_per_round(model):
    env = environment(model)
    bit_strings = ["10", "11", "01", "00"]
    scores, _ = scoring.score_individual(bit_strings, env, seeds=["seed"])
    agents = make_agents([rules.rule_from_bit_string(b) for b in bit_strings], env.n_c, env.n_d)
    result = play_episode(agents, env, make_rng("seed"))
    assert scores == [result.payoff[i] / env.rounds for i in range(4)]


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_more_episodes_are_averaged(model):
    env = environment(model)
    one_a, _ = scoring.score_individual(["10", "00"], env, seeds=["a"])
    one_b, _ = scoring.score_individual(["10", "00"], env, seeds=["b"])
    both, _ = scoring.score_individual(["10", "00"], env, seeds=["a", "b"])
    assert both == pytest.approx([(a + b) / 2 for a, b in zip(one_a, one_b)])
