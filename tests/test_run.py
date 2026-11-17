"""Tests for igss/run.py (SPEC §11, step 7): a small sweep writes the files of SPEC §10."""

import csv

from igss import baseline, config, run


def read(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_small_sweep_writes_its_files(tmp_path):
    run.main(["--out", str(tmp_path), "sweep", "smoke"])
    (directory,) = tmp_path.iterdir()
    runs = read(directory / "runs.csv")
    generations = read(directory / "generations.csv")
    composition = read(directory / "composition.csv")
    rule_rows = read(directory / "rules.csv")

    n_runs = len(config.expand_sweep("smoke"))
    assert len(runs) == n_runs
    assert len(generations) == n_runs * 5                      # GENERATIONS = 5
    assert len(rule_rows) == n_runs * 4 * 8                    # generations 1, 2, 4, 5; P = 8
    assert tuple(runs[0]) == run.RUN_COLUMNS
    assert tuple(generations[0]) == run.GENERATION_COLUMNS
    assert tuple(composition[0]) == run.COMPOSITION_COLUMNS
    assert tuple(rule_rows[0]) == run.RULE_COLUMNS
    assert {row["mode"] for row in runs} == {"1", "2", "3"}

    # the strategy counts of every generation add up to P
    totals = {}
    for row in composition:
        key = (row["run_id"], row["generation"])
        totals[key] = totals.get(key, 0) + int(row["count"])
    assert len(totals) == n_runs * 5 and set(totals.values()) == {8}

    for row in generations:
        assert row["scoring"] in ("individual", "aggregate")   # every row records the scoring option
        assert row["assessment_rule"].startswith("TT:")
        # a background type with no agents has empty cells
        assert (row["payoff_c"] == "") == (row["n_c"] == "0")
        assert row["payoff_d"] != ""
    assert all(row["strategy"].startswith("TT:") for row in rule_rows)
    # every row of every per-run file records the scoring option of its run
    scoring_of = {row["run_id"]: row["scoring"] for row in runs}
    for rows in (generations, composition, rule_rows):
        assert all(row["scoring"] == scoring_of[row["run_id"]] for row in rows)
    assert {row["recovered"] for row in runs} <= {"0", "1"}

    # what evolves is what has an expression
    mode_of = {row["run_id"]: row["mode"] for row in runs}
    for row in rule_rows:
        assert (row["action_expression"] != "") == (mode_of[row["run_id"]] in ("1", "3"))
        assert (row["assessment_expression"] != "") == (mode_of[row["run_id"]] in ("2", "3"))

    # literature names are output labels taken from SPEC §16
    labels = {(row["model"], row["strategy_name"]): row for row in read(directory / "labels.csv")}
    assert labels[("DR", "Tit-for-Tat")]["named_reciprocity"] == "1"
    assert labels[("IR", "Standing")]["named_reciprocity"] == "1"
    assert labels[("IR", "unnamed")]["named_reciprocity"] == "0"
    for row in runs:
        if row["most_common"] == "TT:10/1100" and row["model"] == "DR":
            assert row["most_common_name"] == "Tit-for-Tat"
            assert row["recovered"] == "1"


def test_baseline_command_writes_its_file(tmp_path):
    run.main(["--out", str(tmp_path), "baseline", "smoke"])
    (directory,) = tmp_path.iterdir()
    rows = read(directory / "baseline.csv")
    assert tuple(rows[0]) == run.ENVIRONMENT_COLUMNS + baseline.BASELINE_COLUMNS
    # per mode: 4 environments (2 models x 2 N_C), 2 seeds; 4, 16 and 34 strategies
    assert len(rows) == 4 * 2 * (4 + 16 + 34)


def test_parts_divide_the_runs_of_a_sweep():
    # SPEC §14.8: run i belongs to part k of n when i mod n = k - 1.
    runs = config.expand_sweep("smoke")
    parts = [run.select_part(runs, k, 5) for k in range(1, 6)]
    ids = [cfg["RUN_ID"] for part in parts for cfg in part]
    assert sorted(ids) == sorted(cfg["RUN_ID"] for cfg in runs)
    assert len(ids) == len(set(ids))
    assert [cfg["RUN_ID"] for cfg in parts[1]] == [cfg["RUN_ID"] for cfg in runs[1::5]]


def test_part_argument_is_checked():
    assert run.parse_part("3/40") == (3, 40)
    for text in ("0/4", "5/4", "3", "a/b", "1/0"):
        try:
            run.parse_part(text)
        except ValueError:
            continue
        raise AssertionError(text)


def test_a_part_writes_the_same_rows_as_the_full_sweep(tmp_path):
    full_root, part_root = tmp_path / "full", tmp_path / "part"
    run.main(["--out", str(full_root), "sweep", "smoke"])
    run.main(["--out", str(part_root), "sweep", "smoke", "--part", "2/3"])
    (full,) = full_root.iterdir()
    (part,) = part_root.iterdir()
    assert part.name.startswith("smoke-part2of3-")
    (info,) = read(part / "part.csv")
    assert (info["sweep"], info["part"], info["parts"]) == ("smoke", "2", "3")
    part_ids = {cfg["RUN_ID"] for cfg in run.select_part(config.expand_sweep("smoke"), 2, 3)}
    assert info["runs"] == str(len(part_ids))
    for name in ("runs.csv", "generations.csv", "composition.csv", "rules.csv"):
        expected = [row for row in read(full / name) if row["run_id"] in part_ids]
        assert read(part / name) == expected, name


def test_dry_run_writes_nothing(tmp_path, capsys):
    run.main(["--out", str(tmp_path), "dry-run", "first"])
    assert list(tmp_path.iterdir()) == []
    assert "Nothing was run" in capsys.readouterr().out
