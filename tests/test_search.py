"""Tests for igss/search.py (SPEC §11, step 5)."""

import random

import pytest

from igss import config, search


def small_config(model, scoring_option, **changes):
    cfg = {
        **config.BASE,
        "MODEL": model, "SCORING": scoring_option,
        "N_IGSS": 8, "POP_SIZE": 8, "N_C": 2, "N_D": 2,
        "E_ACTION": 0.05, "ROUNDS": 15, "GENERATIONS": 6,
        "RUN_SEED": 7,
    }
    cfg.update(changes)
    return cfg


CASES = [(m, s) for m in ("DR", "IR") for s in ("individual", "aggregate")]


@pytest.mark.parametrize("model, scoring_option", CASES)
def test_run_is_reproducible_from_its_seed(model, scoring_option):
    cfg = small_config(model, scoring_option)
    first = search.run_evolution(cfg)
    random.seed(12345)  # whatever happened to the global stream in between must not matter
    assert search.run_evolution(cfg) == first


@pytest.mark.parametrize("model, scoring_option", CASES)
def test_counts_add_up_to_population_size(model, scoring_option):
    cfg = small_config(model, scoring_option)
    records = search.run_evolution(cfg)
    assert [record["generation"] for record in records] == list(range(1, 7))
    for record in records:
        assert sum(record["counts"].values()) == cfg["POP_SIZE"]
        assert len(record["rules"]) == cfg["POP_SIZE"]  # every rule is scored in every generation


def test_episodes_do_not_touch_deaps_stream():
    # Changing the number of rounds changes every episode, but not generation 1.
    def first_generation(rounds):
        records = search.run_evolution(small_config("IR", "individual", ROUNDS=rounds))
        return [rule["expression"] for rule in records[0]["rules"]]
    assert first_generation(15) == first_generation(40)


def test_depth_limit_is_respected():
    cfg = small_config("IR", "aggregate", DEPTH_LIMIT=4, GENERATIONS=15)
    for record in search.run_evolution(cfg):
        assert all(rule["height"] <= 4 for rule in record["rules"])


def test_initial_depths_come_from_the_configuration():
    cfg = small_config("IR", "aggregate", INIT_DEPTH_MIN=2, INIT_DEPTH_MAX=2, GENERATIONS=1)
    assert {rule["height"] for rule in search.run_evolution(cfg)[0]["rules"]} == {2}


def test_parsimony_tax_lowers_the_selection_score_only():
    untaxed = search.run_evolution(small_config("IR", "aggregate", GENERATIONS=1))
    taxed = search.run_evolution(small_config("IR", "aggregate", GENERATIONS=1, PARSIMONY_TAX=0.1))
    for a, b in zip(untaxed[0]["rules"], taxed[0]["rules"]):
        assert a["score"] == b["score"]
        assert a["selection_score"] == a["score"]
        assert b["selection_score"] == pytest.approx(b["score"] - 0.1 * b["size"])


def test_unknown_scoring_fails():
    with pytest.raises(ValueError):
        search.run_evolution(small_config("IR", "something else"))
