"""Tests for igss/config.py (SPEC §11, step 6)."""

import pathlib

import pytest

from igss import config

IGSS_DIR = pathlib.Path(config.__file__).parent


def test_configuration_without_scoring_fails():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODEL": "DR"})


def test_configuration_without_model_fails():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "SCORING": "individual"})


def test_base_has_no_model_and_no_scoring():
    assert "SCORING" not in config.BASE
    assert "MODEL" not in config.BASE


def test_every_parameter_is_read_by_the_code():
    # A parameter that nothing reads is a bug (CLAUDE.md rule 7).
    other_code = "".join(path.read_text(encoding="utf-8")
                         for path in IGSS_DIR.glob("*.py") if path.name != "config.py")
    own_code = (IGSS_DIR / "config.py").read_text(encoding="utf-8")
    for name in config.PARAMETERS:
        used_elsewhere = f'"{name}"' in other_code
        used_in_config_functions = own_code.count(f'"{name}"') > 1  # beyond its definition
        assert used_elsewhere or used_in_config_functions, name


@pytest.mark.parametrize("sweep_name", list(config.SWEEPS))
def test_every_run_of_every_sweep_is_valid(sweep_name):
    runs = config.expand_sweep(sweep_name)
    assert len({cfg["RUN_ID"] for cfg in runs}) == len(runs)
    for cfg in runs:
        config.validate(cfg)


def test_first_sweep_matches_spec():
    # SPEC §6.3: 2 models x 2 N_C x 2 N_D x 2 b x 2 e_action, under 2 scoring options, 50 seeds.
    cells = config.sweep_cells("first")
    assert len(cells) == 64
    assert len(config.expand_sweep("first")) == 64 * 50


def test_individual_scoring_needs_one_agent_per_rule():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODEL": "DR", "SCORING": "individual", "N_IGSS": 10})


def test_odd_population_fails():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODEL": "DR", "SCORING": "aggregate", "POP_SIZE": 31})


def test_repetitions_share_run_seeds_across_cells():
    runs = config.expand_sweep("smoke")
    seeds_by_cell = {}
    for cfg in runs:
        seeds_by_cell.setdefault(cfg["RUN_ID"].rsplit("-", 1)[0], []).append(cfg["RUN_SEED"])
    assert len({tuple(seeds) for seeds in seeds_by_cell.values()}) == 1
