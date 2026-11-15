"""Tests for igss/search.py (SPEC §11, step 5)."""

import random

import pytest

from igss import config, search


def small_config(mode, model, scoring_option, **changes):
    cfg = {
        **config.BASE,
        "MODE": mode, "MODEL": model, "SCORING": scoring_option,
        "N_IGSS": 8, "POP_SIZE": 8, "N_C": 2, "N_D": 2,
        "E_ACTION": 0.05, "ROUNDS": 15, "GENERATIONS": 6,
        "RUN_SEED": 7,
    }
    cfg.update(changes)
    return cfg


CASES = [(mode, m, s) for mode in (1, 2, 3) for m in ("DR", "IR")
         for s in ("individual", "aggregate")]


@pytest.mark.parametrize("mode, model, scoring_option", CASES)
def test_run_is_reproducible_from_its_seed(mode, model, scoring_option):
    cfg = small_config(mode, model, scoring_option)
    first = search.run_evolution(cfg)
    random.seed(12345)  # whatever happened to the global stream in between must not matter
    assert search.run_evolution(cfg) == first


@pytest.mark.parametrize("mode, model, scoring_option", CASES)
def test_counts_add_up_to_population_size(mode, model, scoring_option):
    cfg = small_config(mode, model, scoring_option)
    records = search.run_evolution(cfg)
    assert [record["generation"] for record in records] == list(range(1, 7))
    for record in records:
        assert sum(record["counts"].values()) == cfg["POP_SIZE"]
        assert len(record["rules"]) == cfg["POP_SIZE"]  # every rule is scored in every generation


def test_what_evolves_in_each_mode():
    def first_rule(mode):
        return search.run_evolution(small_config(mode, "IR", "aggregate", GENERATIONS=1))[0]["rules"][0]
    assert "action_expression" in first_rule(1) and "assessment_expression" not in first_rule(1)
    assert "assessment_expression" in first_rule(2) and "action_expression" not in first_rule(2)
    assert "action_expression" in first_rule(3) and "assessment_expression" in first_rule(3)


def test_mode_2_always_uses_the_canonical_action_rule():
    for record in search.run_evolution(small_config(2, "DR", "aggregate")):
        assert all(strategy.startswith("TT:10/") for strategy in record["counts"])


def test_mode_1_strategies_carry_the_base_assessment_rule():
    for record in search.run_evolution(small_config(1, "IR", "aggregate")):
        assert set(record["counts"]) <= {"TT:11/any", "TT:10/1101", "TT:01/1101", "TT:00/any"}


def test_episodes_do_not_touch_deaps_stream():
    # Changing the number of rounds changes every episode, but not generation 1.
    def first_generation(rounds):
        records = search.run_evolution(small_config(3, "IR", "individual", ROUNDS=rounds))
        return [(rule["action_expression"], rule["assessment_expression"])
                for rule in records[0]["rules"]]
    assert first_generation(15) == first_generation(40)


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_depth_limit_is_respected(mode):
    cfg = small_config(mode, "IR", "aggregate", DEPTH_LIMIT=4, GENERATIONS=15)
    for record in search.run_evolution(cfg):
        assert all(rule["height"] <= 4 for rule in record["rules"])


def test_initial_depths_come_from_the_configuration():
    cfg = small_config(1, "IR", "aggregate", INIT_DEPTH_MIN=2, INIT_DEPTH_MAX=2, GENERATIONS=1)
    assert {rule["height"] for rule in search.run_evolution(cfg)[0]["rules"]} == {2}


def test_parsimony_tax_lowers_the_selection_score_only():
    untaxed = search.run_evolution(small_config(3, "IR", "aggregate", GENERATIONS=1))
    taxed = search.run_evolution(small_config(3, "IR", "aggregate", GENERATIONS=1,
                                              PARSIMONY_TAX=0.1))
    for a, b in zip(untaxed[0]["rules"], taxed[0]["rules"]):
        assert a["score"] == b["score"]
        assert a["selection_score"] == a["score"]
        assert b["selection_score"] == pytest.approx(b["score"] - 0.1 * b["size"])


def test_unknown_scoring_fails():
    with pytest.raises(ValueError):
        search.run_evolution(small_config(1, "IR", "something else"))
