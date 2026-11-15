"""Tests for igss/episode.py (SPEC §11, step 2)."""

import pytest

from igss import rules
from igss.episode import (ALWAYS_COOPERATE, ALWAYS_DEFECT, IGSS, Environment,
                          make_agents, make_rng, play_episode, summarise)


def environment(model, **changes):
    values = dict(
        model=model,
        assessment_rule={"DR": "what_donor_did", "IR": "did_not_refuse_a_one"}[model],
        n_c=1, n_d=1, b=3, c=1, e_action=0.0, rounds=2,
    )
    values.update(changes)
    return Environment(**values)


class ScriptedRng:
    """Stands in for the random-number generator, so a trace can be checked by hand.

    orders:   the donor order of each round, as agent ids
    partners: for each interaction, the position of the partner among the
              other agents in id order
    errors:   for each interaction, the number drawn for execution error
    """

    def __init__(self, orders, partners, errors):
        self.orders, self.partners, self.errors = list(orders), list(partners), list(errors)

    def shuffle(self, agents):
        order = self.orders.pop(0)
        agents.sort(key=lambda agent: order.index(agent.agent_id))

    def randrange(self, n):
        return self.partners.pop(0)

    def random(self):
        return self.errors.pop(0)


# The script shared by the two traces below.
#   Agents: 0 = iGSS with `follow_input`, 1 = unconditional cooperator,
#           2 = unconditional defector.  b = 3, c = 1, e_action = 0.5.
#   Round 1, donor order 2, 0, 1:
#     donor 2 -> partner 0;  donor 0 -> partner 2;  donor 1 -> partner 0 (error draw 0.1)
#   Round 2, donor order 1, 0, 2:
#     donor 1 -> partner 2;  donor 0 -> partner 1;  donor 2 -> partner 1
def scripted_rng():
    return ScriptedRng(
        orders=[[2, 0, 1], [1, 0, 2]],
        partners=[0, 1, 0, 1, 0, 1],
        errors=[0.9, 0.9, 0.1, 0.9, 0.9, 0.9],
    )


def test_hand_checked_trace_ir():
    # Assessment rule: standing becomes 0 only for refusing a partner with standing 1.
    # Round 1 (all standings start at 1)
    #   2 -> 0: defector refuses a 1.            standing[2] = 0
    #   0 -> 2: reads 0, defects. Refused a 0.   standing[0] = 1
    #   1 -> 0: decides C, draw 0.1 < 0.5 fails; refused a 1.  standing[1] = 0
    #   payoffs 0, 0, 0
    # Round 2
    #   1 -> 2: decides C, succeeds. payoff[1] = -1, payoff[2] = 3.  standing[1] = 1
    #   0 -> 1: reads 1, cooperates. payoff[0] = -1, payoff[1] = 2.  standing[0] = 1
    #   2 -> 1: defector refuses a 1.            standing[2] = 0
    env = environment("IR", e_action=0.5)
    agents = make_agents([rules.follow_input], n_c=1, n_d=1)
    result = play_episode(agents, env, scripted_rng())

    assert result.payoff == {0: -1, 1: 2, 2: 3}
    assert [agent.standing for agent in agents] == [1, 1, 0]
    assert result.reads == {0: [1, 1]}          # iGSS agent read one 0 and one 1
    assert result.cooperated == {0: 1, 1: 1, 2: 0}
    assert result.cooperations[(IGSS, ALWAYS_COOPERATE)] == 1
    assert result.cooperations[(ALWAYS_COOPERATE, ALWAYS_DEFECT)] == 1


def test_hand_checked_trace_dr():
    # Assessment rule: the partner's record of the donor becomes what the donor did.
    # Round 1 (no records yet; an absent record counts as 1)
    #   2 -> 0: defects.                         records[0][2] = 0
    #   0 -> 2: reads own record of 2 = 0, defects.  records[2][0] = 0
    #   1 -> 0: decides C, draw 0.1 < 0.5 fails. records[0][1] = 0
    # Round 2
    #   1 -> 2: cooperates. payoff[1] = -1, payoff[2] = 3.  records[2][1] = 1
    #   0 -> 1: reads own record of 1 = 0, defects.  records[1][0] = 0
    #   2 -> 1: defects.                         records[1][2] = 0
    env = environment("DR", e_action=0.5)
    agents = make_agents([rules.follow_input], n_c=1, n_d=1)
    result = play_episode(agents, env, scripted_rng())

    assert result.payoff == {0: 0, 1: -1, 2: 3}
    assert agents[0].records == {2: 0, 1: 0}
    assert agents[1].records == {0: 0, 2: 0}
    assert agents[2].records == {0: 0, 1: 1}
    assert result.reads == {0: [2, 0]}          # iGSS agent read 0 twice
    assert result.cooperated == {0: 0, 1: 1, 2: 0}


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_without_error_background_agents_do_what_their_kind_says(model):
    env = environment(model, n_c=3, n_d=3, rounds=50)
    agents = make_agents([rules.follow_input] * 4, n_c=3, n_d=3)
    result = play_episode(agents, env, make_rng("test"))
    for agent in agents:
        if agent.kind == ALWAYS_COOPERATE:
            assert result.cooperated[agent.agent_id] == env.rounds
        if agent.kind == ALWAYS_DEFECT:
            assert result.cooperated[agent.agent_id] == 0


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_without_error_action_taken_equals_decision(model):
    # A group that always decides to cooperate cooperates in every interaction.
    env = environment(model, n_c=2, n_d=0, rounds=50)
    agents = make_agents([rules.always_cooperate] * 4, n_c=2, n_d=0)
    result = play_episode(agents, env, make_rng("test"))
    assert sum(result.cooperated.values()) == len(agents) * env.rounds


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_with_certain_error_nobody_cooperates(model):
    env = environment(model, n_c=3, n_d=1, e_action=1.0, rounds=50)
    agents = make_agents([rules.always_cooperate] * 4, n_c=3, n_d=1)
    result = play_episode(agents, env, make_rng("test"))
    assert sum(result.cooperated.values()) == 0
    assert all(payoff == 0 for payoff in result.payoff.values())


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_same_seed_same_output(model):
    env = environment(model, n_c=3, n_d=3, e_action=0.05, rounds=50)
    results = []
    for _ in range(2):
        agents = make_agents([rules.follow_input] * 4, n_c=3, n_d=3)
        results.append(play_episode(agents, env, make_rng("same seed")))
    assert results[0] == results[1]


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_bookkeeping_adds_up(model):
    env = environment(model, n_c=3, n_d=3, e_action=0.05, rounds=50)
    agents = make_agents([rules.follow_input, rules.reverse_input] * 2, n_c=3, n_d=3)
    result = play_episode(agents, env, make_rng("test"))
    n_cooperations = sum(result.cooperated.values())
    # every agent is donor once per round
    assert sum(result.interactions.values()) == len(agents) * env.rounds
    # each cooperation moves -c and +b
    assert sum(result.payoff.values()) == pytest.approx((env.b - env.c) * n_cooperations)
    # every iGSS donor reads its input once per round
    assert all(sum(counts) == env.rounds for counts in result.reads.values())


def test_ir_refusing_a_zero_keeps_standing_one():
    # The case in which the legacy code differed (docs/rebuild_plan.md, M1).
    # Round 1: 1 -> 0, the defector refuses a 1: standing[1] = 0.
    # Round 2: 0 -> 1, the iGSS agent reads 0 and refuses: its standing stays 1.
    env = environment("IR", n_c=0, n_d=1, rounds=2)
    agents = make_agents([rules.follow_input], n_c=0, n_d=1)
    rng = ScriptedRng(orders=[[1, 0], [0, 1]], partners=[0, 0, 0, 0], errors=[0.9] * 4)
    play_episode(agents, env, rng)
    assert [agent.standing for agent in agents] == [1, 0]


def test_absent_type_is_summarised_as_none():
    env = environment("IR", n_c=0, n_d=2, rounds=10)
    agents = make_agents([rules.follow_input] * 2, n_c=0, n_d=2)
    summary = summarise(play_episode(agents, env, make_rng("test")), agents, env)
    assert summary["payoff_c"] is None
    assert summary["coop_igss_to_c"] is None
    assert summary["payoff_d"] is not None
