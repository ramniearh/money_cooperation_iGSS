"""Tests for igss/baseline.py (SPEC §11, step 3)."""

import pytest

from igss import baseline
from igss.episode import Environment


def environment(model, **changes):
    values = dict(
        model=model,
        assessment_rule={"DR": "what_donor_did", "IR": "did_not_refuse_a_one"}[model],
        n_c=4, n_d=0, b=2, c=1, e_action=0.0, rounds=30,
    )
    values.update(changes)
    return Environment(**values)


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_no_defectors_no_error_follow_and_always_cooperate_identical(model):
    # SPEC §6.3, the known collapse: input 0 is never read, so `10` and `11` cannot be told apart.
    env = environment(model)
    follow = baseline.run_alone("follow_input", env, n_igss=6, seed_text="seed")
    always = baseline.run_alone("always_cooperate", env, n_igss=6, seed_text="seed")
    assert follow == always
    assert follow["x"]["reads_zero"] == 0


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_pair_of_a_rule_with_itself_equals_the_rule_alone(model):
    env = environment(model, n_d=4, e_action=0.05)
    alone = baseline.run_alone("follow_input", env, n_igss=6, seed_text="seed")
    pair = baseline.run_pair("follow_input", "follow_input", env, n_igss=6, seed_text="seed")
    assert pair["summary"] == alone["summary"]
    assert (pair["x"]["payoff"] + pair["y"]["payoff"]) / 2 == pytest.approx(alone["x"]["payoff"])


def test_baseline_has_four_alone_sets_and_six_pair_sets():
    rows = list(baseline.baseline_rows(environment("IR", n_d=2), n_igss=4, n_seeds=3))
    assert sum(row["kind"] == "alone" for row in rows) == 4 * 3
    assert sum(row["kind"] == "pair" for row in rows) == 6 * 3
    assert all(set(row) == set(baseline.BASELINE_COLUMNS) for row in rows)
    # in an alone run the columns of rule Y are empty
    assert all(row["payoff_y"] is None for row in rows if row["kind"] == "alone")
