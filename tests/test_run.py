"""Tests for igss/run.py (SPEC §11, step 7): a small sweep writes the files of SPEC §10."""

import csv

from igss import baseline, config, run


def read(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_small_sweep_writes_the_three_files(tmp_path):
    run.main(["--out", str(tmp_path), "sweep", "smoke"])
    (directory,) = tmp_path.iterdir()
    runs = read(directory / "runs.csv")
    generations = read(directory / "generations.csv")
    rule_rows = read(directory / "rules.csv")

    n_runs = len(config.expand_sweep("smoke"))
    assert len(runs) == n_runs
    assert len(generations) == n_runs * 5                      # GENERATIONS = 5
    assert len(rule_rows) == n_runs * 4 * 8                    # generations 1, 2, 4, 5; P = 8
    assert tuple(runs[0]) == run.RUN_COLUMNS
    assert tuple(generations[0]) == run.GENERATION_COLUMNS
    assert tuple(rule_rows[0]) == run.RULE_COLUMNS

    for row in generations:
        assert sum(int(row[column]) for column in run.COUNT_COLUMNS) == 8
        assert row["scoring"] in ("individual", "aggregate")   # every row records the scoring option
        assert row["assessment_rule"].startswith("TT:")
        # a background type with no agents has empty cells
        assert (row["payoff_c"] == "") == (row["n_c"] == "0")
        assert row["payoff_d"] != ""
    assert all(row["bit_string"].startswith("TT:") for row in rule_rows)
    assert {row["recovered"] for row in runs} <= {"0", "1"}

    # literature names are output labels taken from SPEC §16
    labels = {(row["model"], row["bit_string"]): row for row in read(directory / "labels.csv")}
    assert labels[("DR", "TT:10")]["literature_name"] == "Tit-for-Tat"
    assert labels[("IR", "TT:10")]["literature_name"] == "Standing"
    assert labels[("IR", "TT:01")]["literature_name"] == "unnamed"
    assert labels[("DR", "TT:10")]["canonical"] == "1"
    for row in runs:
        if row["most_common"] == "TT:10":
            assert row["most_common_name"] == labels[(row["model"], "TT:10")]["literature_name"]


def test_baseline_command_writes_its_file(tmp_path):
    run.main(["--out", str(tmp_path), "baseline", "smoke"])
    (directory,) = tmp_path.iterdir()
    rows = read(directory / "baseline.csv")
    assert tuple(rows[0]) == run.ENVIRONMENT_COLUMNS + baseline.BASELINE_COLUMNS
    # 4 environments (2 models x 2 N_C), 10 sets of runs each, 2 seeds per set
    assert len(rows) == 4 * 10 * 2


def test_dry_run_writes_nothing(tmp_path, capsys):
    run.main(["--out", str(tmp_path), "dry-run", "first"])
    assert list(tmp_path.iterdir()) == []
    assert "Nothing was run" in capsys.readouterr().out
