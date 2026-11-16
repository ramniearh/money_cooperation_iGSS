"""Tests for igss/config.py (SPEC §11, step 6)."""

import pathlib

import pytest

from igss import config

IGSS_DIR = pathlib.Path(config.__file__).parent


def test_configuration_without_scoring_fails():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODE": 1, "MODEL": "DR"})


def test_configuration_without_model_fails():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODE": 1, "SCORING": "individual"})


def test_configuration_without_mode_fails():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODEL": "DR", "SCORING": "aggregate"})


def test_base_has_no_model_and_no_scoring():
    assert "SCORING" not in config.BASE
    assert "MODE" not in config.BASE
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


def test_ir_mode_2_sweeps_match_spec():
    # SPEC §14.5: IR, mode 2, aggregate scoring, 10 seeds per combination.
    expected_cells = {"ir2_env": 12, "ir2_env_e005": 12, "ir2_rounds": 3, "ir2_generations": 1,
                      "ir2_parsimony": 3, "ir2_depth": 3, "ir2_tournament": 3}
    for sweep_name, n_cells in expected_cells.items():
        cells = config.sweep_cells(sweep_name)
        assert len(cells) == n_cells, sweep_name
        assert len(config.expand_sweep(sweep_name)) == n_cells * 10, sweep_name
        for cfg in cells:
            assert (cfg["MODE"], cfg["MODEL"], cfg["SCORING"]) == (2, "IR", "aggregate")
    assert {(c["N_C"], c["N_D"], c["B"]) for c in config.sweep_cells("ir2_env")} == {
        (n_c, n_d, b) for n_c in (0, 32) for n_d in (0, 32) for b in (2, 5, 10)}
    # ir2_env_e005 is the grid of ir2_env with e_action = 0.05 and nothing else changed.
    without_error = [{**c, "E_ACTION": 0.05} for c in config.sweep_cells("ir2_env")]
    assert config.sweep_cells("ir2_env_e005") == without_error
    assert {c["E_ACTION"] for c in config.sweep_cells("ir2_env")} == {0.0}
    assert [c["ROUNDS"] for c in config.sweep_cells("ir2_rounds")] == [100, 400, 1600]
    assert [c["GENERATIONS"] for c in config.sweep_cells("ir2_generations")] == [200]
    assert [c["PARSIMONY_TAX"] for c in config.sweep_cells("ir2_parsimony")] == [0.0, 0.01, 0.05]
    assert [c["DEPTH_LIMIT"] for c in config.sweep_cells("ir2_depth")] == [3, 5, 17]
    assert [c["TOURNAMENT_SIZE"] for c in config.sweep_cells("ir2_tournament")] == [2, 3, 7]


def test_individual_look_matches_spec():
    # SPEC §14.6: 2 models x 3 modes x 2 N_C x 2 N_D, individual scoring, 20 seeds.
    cells = config.sweep_cells("individual_look")
    assert len(cells) == 24
    assert len(config.expand_sweep("individual_look")) == 24 * 20
    for cfg in cells:
        assert cfg["SCORING"] == "individual"
        assert (cfg["B"], cfg["E_ACTION"], cfg["GENERATIONS"]) == (10, 0.0, 200)
        assert (cfg["ROUNDS"], cfg["EPISODES_PER_SCORE"]) == (400, 1)
    assert {(c["MODE"], c["MODEL"], c["N_C"], c["N_D"]) for c in cells} == {
        (mode, model, n_c, n_d) for mode in (1, 2, 3) for model in ("DR", "IR")
        for n_c in (0, 32) for n_d in (0, 32)}


def test_broad_sweep_matches_spec():
    # SPEC §14.7: combinations and runs of each of the ten sweeps.
    expected = {"broad_individual": (324, 6480), "broad_aggregate": (144, 2880),
                "broad_norms_ir": (64, 1280), "broad_norms_dr": (32, 640),
                "broad_rounds": (18, 360), "broad_episodes": (12, 240),
                "broad_parsimony": (36, 360), "broad_depth": (36, 360),
                "broad_tournament": (36, 360), "broad_long": (12, 120)}
    assert {name for name in config.SWEEPS if name.startswith("broad_")} == set(expected)
    total = 0
    for sweep_name, (n_cells, n_runs) in expected.items():
        cells = config.sweep_cells(sweep_name)
        assert len(cells) == n_cells, sweep_name
        assert len(config.expand_sweep(sweep_name)) == n_runs, sweep_name
        assert {c["GENERATIONS"] for c in cells} == ({500} if sweep_name == "broad_long" else {200})
        total += n_runs
    assert total == 13080

    # The fixed assessment rules of the two norms sweeps are the named rules of SPEC §16.
    def fixed_rules(sweep_name, key):
        return {config.rules.ASSESSMENT_BITS[c[key]] for c in config.sweep_cells(sweep_name)}

    assert fixed_rules("broad_norms_ir", "IR_ASSESSMENT_RULE") == {"1101", "1100", "1001", "1000"}
    assert fixed_rules("broad_norms_dr", "DR_ASSESSMENT_RULE") == {"1100", "1000"}
    assert {c["MODE"] for c in config.sweep_cells("broad_norms_ir")} == {1}


def test_individual_scoring_needs_one_agent_per_rule():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODE": 1, "MODEL": "DR", "SCORING": "individual", "N_IGSS": 10})


def test_odd_population_fails():
    with pytest.raises(ValueError):
        config.validate({**config.BASE, "MODE": 1, "MODEL": "DR", "SCORING": "aggregate", "POP_SIZE": 31})


def test_repetitions_share_run_seeds_across_cells():
    runs = config.expand_sweep("smoke")
    seeds_by_cell = {}
    for cfg in runs:
        seeds_by_cell.setdefault(cfg["RUN_ID"].rsplit("-", 1)[0], []).append(cfg["RUN_SEED"])
    assert len({tuple(seeds) for seeds in seeds_by_cell.values()}) == 1
