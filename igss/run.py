"""Command-line entry point, and the writer of the output files (SPEC §10, §11).

    python -m igss.run quick --model DR --scoring individual
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
    ("sweep", "run_id", "code_version", "run_seed", "scoring", "model", "assessment_rule")
    + tuple(name.lower() for name in config.BASE if name not in _ASSESSMENT_KEYS)
)
COUNT_COLUMNS = tuple("count_" + bits for bits in rules.ALL_ACTION_BITS)
GENERATION_COLUMNS = (
    PARAMETER_COLUMNS + ("generation",) + COUNT_COLUMNS
    + ("most_common", "most_common_name", "mean_size")
    + SUMMARY_COLUMNS
)
RUN_COLUMNS = GENERATION_COLUMNS + ("tied", "recovered")
RULE_COLUMNS = ("run_id", "generation", "rule_index", "expression", "simplified",
                "size", "height", "bit_string", "score", "selection_score")
LABEL_COLUMNS = ("model", "assessment_rule", "bit_string", "description", "literature_name",
                 "canonical")
ENVIRONMENT_COLUMNS = ("sweep", "code_version", "model", "assessment_rule",
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
        "scoring": cfg["SCORING"],
        "model": cfg["MODEL"],
        "assessment_rule": tt(rules.ASSESSMENT_BITS[config.assessment_rule(cfg)]),
    })
    return values


# --- One run, turned into rows ----------------------------------------------

def run_rows(job):
    """Run one configuration; return its row for runs.csv and its rows for the other two files."""
    cfg, version = job
    parameters = parameter_values(cfg, version)
    assessment_bits = rules.ASSESSMENT_BITS[config.assessment_rule(cfg)]
    generation_rows, rule_rows = [], []
    for record in search.run_evolution(cfg):
        generation = record["generation"]
        row = dict(parameters)
        row["generation"] = generation
        for bits, count in record["counts"].items():
            row["count_" + bits] = count
        row["most_common"] = "+".join(tt(bits) for bits in record["most_common"])
        row["most_common_name"] = "+".join(
            rules.literature_name(cfg["MODEL"], bits, assessment_bits)
            for bits in record["most_common"])
        row["mean_size"] = record["mean_size"]
        row.update(record["summary"])
        generation_rows.append(row)

        is_logged = (generation == 1 or generation == cfg["GENERATIONS"]
                     or generation % cfg["RULES_LOG_EVERY"] == 0)
        if is_logged:
            for rule in record["rules"]:
                small_enough = rule["size"] <= cfg["SIMPLIFY_MAX_SIZE"]
                rule_rows.append({
                    **rule,
                    "run_id": cfg["RUN_ID"],
                    "generation": generation,
                    "bit_string": tt(rule["bit_string"]),
                    "simplified": (rules.simplify(rule["expression"], cfg["MODEL"])
                                   if small_enough else None),
                })

    final = record
    run_row = dict(generation_rows[-1])
    run_row["tied"] = int(len(final["most_common"]) > 1)
    run_row["recovered"] = int(final["most_common"] == [rules.CANONICAL_ACTION_BITS])
    return run_row, generation_rows, rule_rows


# --- Writing -----------------------------------------------------------------

def new_output_directory(out_root, name):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = pathlib.Path(out_root) / f"{name}-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _writer(file, columns):
    writer = csv.DictWriter(file, fieldnames=columns)  # None is written as an empty cell
    writer.writeheader()
    return writer


def write_labels(configurations, directory):
    """labels.csv: what each bit string does and what the literature calls it (SPEC §16)."""
    seen = {(cfg["MODEL"], config.assessment_rule(cfg)) for cfg in configurations}
    with open(directory / "labels.csv", "w", newline="", encoding="utf-8") as file:
        writer = _writer(file, LABEL_COLUMNS)
        for model, assessment in sorted(seen):
            assessment_bits = rules.ASSESSMENT_BITS[assessment]
            for name, bits in rules.ACTION_BITS.items():
                writer.writerow({
                    "model": model,
                    "assessment_rule": tt(assessment_bits),
                    "bit_string": tt(bits),
                    "description": rules.ACTION_DESCRIPTIONS[name],
                    "literature_name": rules.literature_name(model, bits, assessment_bits),
                    "canonical": int(name == rules.CANONICAL_ACTION_RULE),
                })


def write_runs(runs, directory, workers):
    """Run every configuration and write runs.csv, generations.csv, rules.csv and labels.csv."""
    version = code_version()
    write_labels(runs, directory)
    jobs = [(cfg, version) for cfg in runs]
    with (open(directory / "runs.csv", "w", newline="", encoding="utf-8") as runs_file,
          open(directory / "generations.csv", "w", newline="", encoding="utf-8") as generations_file,
          open(directory / "rules.csv", "w", newline="", encoding="utf-8") as rules_file):
        runs_writer = _writer(runs_file, RUN_COLUMNS)
        generations_writer = _writer(generations_file, GENERATION_COLUMNS)
        rules_writer = _writer(rules_file, RULE_COLUMNS)

        if workers > 1:
            pool = multiprocessing.Pool(workers)
            results = pool.imap(run_rows, jobs)  # results come back in the order of `jobs`
        else:
            pool = None
            results = map(run_rows, jobs)
        for done, (run_row, generation_rows, rule_rows) in enumerate(results, start=1):
            runs_writer.writerow(run_row)
            generations_writer.writerows(generation_rows)
            rules_writer.writerows(rule_rows)
            print(f"  run {done} of {len(jobs)} done ({run_row['run_id']})", flush=True)
        if pool:
            pool.close()
            pool.join()


def baseline_environments(sweep_name):
    """The distinct environments of a sweep, whatever the scoring option or search parameters."""
    seen = {}
    for cfg in config.sweep_cells(sweep_name):
        key = (cfg["MODEL"], config.assessment_rule(cfg), cfg["N_IGSS"], cfg["N_C"], cfg["N_D"],
               cfg["B"], cfg["C"], cfg["E_ACTION"], cfg["ROUNDS"], cfg["BASELINE_SEEDS"])
        seen.setdefault(key, cfg)
    return list(seen.values())


def write_baseline(sweep_name, directory):
    version = code_version()
    environments = baseline_environments(sweep_name)
    write_labels(environments, directory)
    with open(directory / "baseline.csv", "w", newline="", encoding="utf-8") as file:
        writer = _writer(file, ENVIRONMENT_COLUMNS + baseline.BASELINE_COLUMNS)
        for done, cfg in enumerate(environments, start=1):
            environment_values = {
                "sweep": sweep_name,
                "code_version": version,
                "model": cfg["MODEL"],
                "assessment_rule": tt(rules.ASSESSMENT_BITS[config.assessment_rule(cfg)]),
                "n_igss": cfg["N_IGSS"], "n_c": cfg["N_C"], "n_d": cfg["N_D"],
                "b": cfg["B"], "c": cfg["C"], "e_action": cfg["E_ACTION"], "rounds": cfg["ROUNDS"],
            }
            env = search.environment_of(cfg)
            for row in baseline.baseline_rows(env, cfg["N_IGSS"], cfg["BASELINE_SEEDS"]):
                writer.writerow({**environment_values, **row})
            print(f"  environment {done} of {len(environments)} done", flush=True)


# --- Commands ----------------------------------------------------------------

def command_quick(args):
    cfg = config.validate({**config.BASE, **config.QUICK,
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


def command_sweep(args):
    runs = config.expand_sweep(args.sweep)
    directory = new_output_directory(args.out, args.sweep)
    print(f"Sweep {args.sweep!r}: {len(runs)} runs, writing to {directory}")
    write_runs(runs, directory, args.workers)


def command_baseline(args):
    directory = new_output_directory(args.out, f"baseline-{args.sweep}")
    print(f"Baseline for sweep {args.sweep!r}, writing to {directory}")
    write_baseline(args.sweep, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m igss.run", description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs", help="directory the output directories go in")
    commands = parser.add_subparsers(dest="command", required=True)

    quick = commands.add_parser("quick", help="one short run of the base configuration")
    quick.add_argument("--model", required=True, choices=rules.MODELS)
    quick.add_argument("--scoring", required=True, choices=config.SCORING_OPTIONS)
    quick.set_defaults(function=command_quick)

    dry_run = commands.add_parser("dry-run", help="list what a sweep would run")
    dry_run.add_argument("sweep", choices=list(config.SWEEPS))
    dry_run.set_defaults(function=command_dry_run)

    sweep = commands.add_parser("sweep", help="run a sweep")
    sweep.add_argument("sweep", choices=list(config.SWEEPS))
    sweep.add_argument("--workers", type=int, default=1, help="processes to run in parallel")
    sweep.set_defaults(function=command_sweep)

    base = commands.add_parser("baseline", help="run the baseline for the environments of a sweep")
    base.add_argument("sweep", choices=list(config.SWEEPS))
    base.set_defaults(function=command_baseline)

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
