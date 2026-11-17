"""Command-line entry point, and the writer of the output files (SPEC §10, §11).

    python -m igss.run quick --mode 1 --model DR --scoring aggregate
    python -m igss.run dry-run first
    python -m igss.run sweep first --workers 4
    python -m igss.run baseline first

Every command that writes makes a new directory under outputs/ and never
appends to an existing file.
"""

import argparse
import csv
import datetime
import multiprocessing
import pathlib
import subprocess

from igss import baseline, config, rules, search
from igss.episode import SUMMARY_COLUMNS

# --- Columns (SPEC §10) ------------------------------------------------------

_ASSESSMENT_KEYS = ("DR_ASSESSMENT_RULE", "IR_ASSESSMENT_RULE")
PARAMETER_COLUMNS = (
    ("sweep", "run_id", "code_version", "run_seed", "mode", "scoring", "model", "assessment_rule")
    + tuple(name.lower() for name in config.BASE if name not in _ASSESSMENT_KEYS)
)
GENERATION_COLUMNS = (
    PARAMETER_COLUMNS
    + ("generation", "most_common", "most_common_name", "most_common_count", "mean_size")
    + SUMMARY_COLUMNS
)
RUN_COLUMNS = GENERATION_COLUMNS + ("tied", "recovered")
# Every row of every per-run file records the scoring option (CLAUDE.md, hard rule 6).
COMPOSITION_COLUMNS = ("run_id", "scoring", "generation", "strategy", "strategy_name", "count")
RULE_COLUMNS = ("run_id", "scoring", "generation", "rule_index", "strategy", "strategy_name",
                "action_expression", "action_simplified",
                "assessment_expression", "assessment_simplified",
                "size", "height", "score", "selection_score")
LABEL_COLUMNS = ("model", "strategy_name", "colour_key", "named_reciprocity")
ENVIRONMENT_COLUMNS = ("sweep", "code_version", "mode", "model", "assessment_rule",
                       "n_igss", "n_c", "n_d", "b", "c", "e_action", "rounds")


def tt(bits):
    """Bit strings are stored with a prefix so that leading zeros survive spreadsheets."""
    return "TT:" + bits


def code_version():
    """The git commit of the code, with '+changes' if the working copy differs from it."""
    here = pathlib.Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain"], cwd=here,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+changes" if changed else "")


def parameter_values(cfg, version):
    values = {name.lower(): cfg[name] for name in config.BASE if name not in _ASSESSMENT_KEYS}
    values.update({
        "sweep": cfg["SWEEP"],
        "run_id": cfg["RUN_ID"],
        "code_version": version,
        "run_seed": cfg["RUN_SEED"],
        "mode": cfg["MODE"],
        "scoring": cfg["SCORING"],
        "model": cfg["MODEL"],
        "assessment_rule": tt(rules.ASSESSMENT_BITS[config.assessment_rule(cfg)]),
    })
    return values


# --- One run, turned into rows ----------------------------------------------

def run_rows(job):
    """Run one configuration; return its rows for the four per-run files."""
    cfg, version = job
    parameters = parameter_values(cfg, version)
    generation_rows, composition_rows, rule_rows = [], [], []
    for record in search.run_evolution(cfg):
        generation = record["generation"]
        row = dict(parameters)
        row["generation"] = generation
        row["most_common"] = "+".join(record["most_common"])
        row["most_common_name"] = "+".join(record["names"][s] for s in record["most_common"])
        row["most_common_count"] = record["most_common_count"]
        row["mean_size"] = record["mean_size"]
        row.update(record["summary"])
        generation_rows.append(row)

        for strategy, count in record["counts"].items():
            composition_rows.append({
                "run_id": cfg["RUN_ID"], "scoring": cfg["SCORING"],
                "generation": generation, "strategy": strategy,
                "strategy_name": record["names"][strategy], "count": count,
            })

        is_logged = (generation == 1 or generation == cfg["GENERATIONS"]
                     or generation % cfg["RULES_LOG_EVERY"] == 0)
        if is_logged:
            for rule in record["rules"]:
                rule_row = {**rule, "run_id": cfg["RUN_ID"], "scoring": cfg["SCORING"],
                            "generation": generation}
                if rule["size"] <= cfg["SIMPLIFY_MAX_SIZE"]:
                    for kind in ("action", "assessment"):
                        text = rule.get(kind + "_expression")
                        if text is not None:
                            rule_row[kind + "_simplified"] = rules.simplify(text, cfg["MODEL"])
                rule_rows.append(rule_row)

    final = record
    run_row = dict(generation_rows[-1])
    # Recovered: one strategy alone is the most common in the final generation,
    # and it is a reciprocity strategy named in SPEC §16.
    run_row["tied"] = int(len(final["most_common"]) > 1)
    run_row["recovered"] = int(final["most_common_is_named_reciprocity"] == [True])
    return run_row, generation_rows, composition_rows, rule_rows


# --- Writing -----------------------------------------------------------------

def new_output_directory(out_root, name):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = pathlib.Path(out_root) / f"{name}-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _open(directory, name):
    return open(directory / name, "w", newline="", encoding="utf-8")


def _writer(file, columns):
    writer = csv.DictWriter(file, fieldnames=columns)  # None is written as an empty cell
    writer.writeheader()
    return writer


def write_labels(configurations, directory):
    """labels.csv: the names a strategy can get in each model of the sweep (SPEC §16)."""
    with _open(directory, "labels.csv") as file:
        writer = _writer(file, LABEL_COLUMNS)
        for model in sorted({cfg["MODEL"] for cfg in configurations}):
            writer.writerows(rules.label_rows(model))


def write_runs(runs, directory, workers):
    """Run every configuration and write the files of a sweep."""
    version = code_version()
    write_labels(runs, directory)
    jobs = [(cfg, version) for cfg in runs]
    with (_open(directory, "runs.csv") as runs_file,
          _open(directory, "generations.csv") as generations_file,
          _open(directory, "composition.csv") as composition_file,
          _open(directory, "rules.csv") as rules_file):
        runs_writer = _writer(runs_file, RUN_COLUMNS)
        generations_writer = _writer(generations_file, GENERATION_COLUMNS)
        composition_writer = _writer(composition_file, COMPOSITION_COLUMNS)
        rules_writer = _writer(rules_file, RULE_COLUMNS)

        if workers > 1:
            pool = multiprocessing.Pool(workers)
            results = pool.imap(run_rows, jobs)  # results come back in the order of `jobs`
        else:
            pool = None
            results = map(run_rows, jobs)
        for done, (run_row, generation_rows, composition_rows, rule_rows) in enumerate(results, 1):
            runs_writer.writerow(run_row)
            generations_writer.writerows(generation_rows)
            composition_writer.writerows(composition_rows)
            rules_writer.writerows(rule_rows)
            print(f"  run {done} of {len(jobs)} done ({run_row['run_id']})", flush=True)
        if pool:
            pool.close()
            pool.join()


def baseline_environments(sweep_name):
    """The distinct environments of a sweep, whatever the scoring option or search parameters."""
    seen = {}
    for cfg in config.sweep_cells(sweep_name):
        key = (cfg["MODE"], cfg["MODEL"], config.assessment_rule(cfg), cfg["N_IGSS"], cfg["N_C"],
               cfg["N_D"], cfg["B"], cfg["C"], cfg["E_ACTION"], cfg["ROUNDS"],
               cfg["BASELINE_SEEDS"])
        seen.setdefault(key, cfg)
    return list(seen.values())


def write_baseline(sweep_name, directory):
    version = code_version()
    environments = baseline_environments(sweep_name)
    write_labels(environments, directory)
    with _open(directory, "baseline.csv") as file:
        writer = _writer(file, ENVIRONMENT_COLUMNS + baseline.BASELINE_COLUMNS)
        for done, cfg in enumerate(environments, start=1):
            environment_values = {
                "sweep": sweep_name,
                "code_version": version,
                "mode": cfg["MODE"],
                "model": cfg["MODEL"],
                "assessment_rule": tt(rules.ASSESSMENT_BITS[config.assessment_rule(cfg)]),
                "n_igss": cfg["N_IGSS"], "n_c": cfg["N_C"], "n_d": cfg["N_D"],
                "b": cfg["B"], "c": cfg["C"], "e_action": cfg["E_ACTION"], "rounds": cfg["ROUNDS"],
            }
            env = search.environment_of(cfg)
            for row in baseline.baseline_rows(env, cfg["MODE"], cfg["N_IGSS"],
                                              cfg["BASELINE_SEEDS"]):
                writer.writerow({**environment_values, **row})
            print(f"  environment {done} of {len(environments)} done", flush=True)


# --- Commands ----------------------------------------------------------------

def command_quick(args):
    cfg = config.validate({**config.BASE, **config.QUICK, "MODE": args.mode,
                           "MODEL": args.model, "SCORING": args.scoring})
    cfg.update({"SWEEP": "quick", "RUN_ID": "quick-0000-000", "RUN_SEED": cfg["BASE_SEED"]})
    directory = new_output_directory(args.out, "quick")
    write_runs([cfg], directory, workers=1)
    print(f"Written to {directory}. A quick run is a check that the code runs, not a result.")


def command_dry_run(args):
    cells = config.sweep_cells(args.sweep)
    runs = config.expand_sweep(args.sweep)
    print(f"Sweep {args.sweep!r}: {len(cells)} combinations, {len(runs)} runs.")
    for name, values in config.SWEEPS[args.sweep].items():
        print(f"  {name}: {values}")
    print(f"Baseline environments: {len(baseline_environments(args.sweep))}")
    print("First runs:")
    for cfg in runs[:5]:
        print(f"  {cfg['RUN_ID']}  seed {cfg['RUN_SEED']}")
    print("Nothing was run.")


PART_COLUMNS = ("sweep", "part", "parts", "runs", "code_version", "started", "finished")


def parse_part(text):
    """'k/n' -> (k, n), with 1 <= k <= n (SPEC §14.8)."""
    try:
        k, n = (int(value) for value in text.split("/"))
    except ValueError:
        raise ValueError(f"a part is written k/n, for example 3/40, not {text!r}") from None
    if not 1 <= k <= n:
        raise ValueError(f"part {text!r}: k must be between 1 and n")
    return k, n


def select_part(runs, k, n):
    """The runs of part k of n: run i (from 0) belongs to it when i mod n = k - 1 (SPEC §14.8)."""
    return [cfg for i, cfg in enumerate(runs) if i % n == k - 1]


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def command_sweep(args):
    runs = config.expand_sweep(args.sweep)
    if args.part is None:
        directory = new_output_directory(args.out, args.sweep)
        print(f"Sweep {args.sweep!r}: {len(runs)} runs, writing to {directory}")
        write_runs(runs, directory, args.workers)
        return
    k, n = parse_part(args.part)
    runs = select_part(runs, k, n)
    directory = new_output_directory(args.out, f"{args.sweep}-part{k}of{n}")
    print(f"Sweep {args.sweep!r}, part {k} of {n}: {len(runs)} runs, writing to {directory}")
    started = _now()
    write_runs(runs, directory, args.workers)
    # Written last, so a part directory without part.csv did not finish.
    with _open(directory, "part.csv") as file:
        _writer(file, PART_COLUMNS).writerow({
            "sweep": args.sweep, "part": k, "parts": n, "runs": len(runs),
            "code_version": code_version(), "started": started, "finished": _now()})


def command_baseline(args):
    directory = new_output_directory(args.out, f"baseline-{args.sweep}")
    print(f"Baseline for sweep {args.sweep!r}, writing to {directory}")
    write_baseline(args.sweep, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m igss.run", description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs", help="directory the output directories go in")
    commands = parser.add_subparsers(dest="command", required=True)

    quick = commands.add_parser("quick", help="one short run of the base configuration")
    quick.add_argument("--mode", required=True, type=int, choices=rules.MODES)
    quick.add_argument("--model", required=True, choices=rules.MODELS)
    quick.add_argument("--scoring", required=True, choices=config.SCORING_OPTIONS)
    quick.set_defaults(function=command_quick)

    dry_run = commands.add_parser("dry-run", help="list what a sweep would run")
    dry_run.add_argument("sweep", choices=list(config.SWEEPS))
    dry_run.set_defaults(function=command_dry_run)

    sweep = commands.add_parser("sweep", help="run a sweep")
    sweep.add_argument("sweep", choices=list(config.SWEEPS))
    sweep.add_argument("--workers", type=int, default=1, help="processes to run in parallel")
    sweep.add_argument("--part", default=None, metavar="K/N",
                       help="run only part K of N of the sweep, into a directory of its own")
    sweep.set_defaults(function=command_sweep)

    base = commands.add_parser("baseline", help="run the baseline for the environments of a sweep")
    base.add_argument("sweep", choices=list(config.SWEEPS))
    base.set_defaults(function=command_baseline)

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
