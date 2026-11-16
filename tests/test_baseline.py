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
    follow = baseline.run_alone(("10", None), env, n_igss=6, seed_text="seed")
    always = baseline.run_alone(("11", None), env, n_igss=6, seed_text="seed")
    assert follow == always
    assert follow["reads_zero"] == 0


def test_number_of_strategies_per_mode():
    # Mode 1: 4 action rules. Mode 2: 16 assessment rules. Mode 3: 64 pairs, of which the
    # 32 with action rule `11` or `00` form two classes (SPEC §14.3): 2 + 32 = 34.
    assert len(baseline.strategies_of_mode(1)) == 4
    assert len(baseline.strategies_of_mode(2)) == 16
    assert len(baseline.strategies_of_mode(3)) == 34


@pytest.mark.parametrize("model, base", [("DR", "1100"), ("IR", "1101")])
def test_evolved_base_rule_equals_mode_1(model, base):
    # A strategy whose assessment rule is the base rule plays exactly as in mode 1.
    env = environment(model, n_d=4, e_action=0.05)
    assert (baseline.run_alone(("10", base), env, 6, "seed")
            == baseline.run_alone(("10", None), env, 6, "seed"))


@pytest.mark.parametrize("mode, n_strategies", [(1, 4), (2, 16), (3, 34)])
def test_baseline_rows(mode, n_strategies):
    rows = list(baseline.baseline_rows(environment("IR", n_d=2), mode, n_igss=4, n_seeds=2))
    assert len(rows) == n_strategies * 2
    assert all(set(row) == set(baseline.BASELINE_COLUMNS) for row in rows)
    names = {row["strategy"]: row["strategy_name"] for row in rows}
    if mode != 2:  # mode 2 fixes the action rule at `10`
        assert names["TT:11/any"] == "unconditional cooperation"
    assert names["TT:10/1101"] == "Standing"


@pytest.mark.parametrize("model", ["DR", "IR"])
def test_unconditional_classes_use_the_base_rule_in_every_mode(model):
    # SPEC §14.3: `11/any` and `00/any` are run with the base assessment rule, so the baseline
    # of mode 3 logs for them exactly what the baseline of mode 1 logs for `11` and `00`.
    env = environment(model, n_d=4, e_action=0.05)

    def rows_of(mode):
        rows = baseline.baseline_rows(env, mode, n_igss=6, n_seeds=2)
        return [row for row in rows if row["strategy"].endswith("/any")]

    assert len(rows_of(1)) == 4
    assert rows_of(3) == rows_of(1)
