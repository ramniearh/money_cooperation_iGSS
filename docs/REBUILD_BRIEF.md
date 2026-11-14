# Brief for Claude Code: rebuilding the iGSS code

<!-- AI-generated draft (Claude, Anthropic) for checking and amendment by the authors. -->

This brief is the task description for the rebuild. `CLAUDE.md` holds the standing rules and `docs/SPEC.md` holds the model. When this brief and SPEC.md disagree, SPEC.md wins: stop and report the conflict.

## What this is, and what it is not

This is a **rebuild against a specification**, not a refactor of the legacy code.

- The legacy code drifted from its intended design through choices made silently, and SPEC §17 lists known errors in it.
- Refactoring it would carry those choices forward under cleaner names.
- So the model core is written new, from SPEC.md.
- The legacy code is used for three things only:
  1. to learn what the repository contains, including files the authors have not yet reviewed in this process (utilities, logging, graph generation, R visualisation);
  2. to find choices the legacy code makes that SPEC.md does not, so that the authors can decide them explicitly;
  3. to identify peripheral code (plotting, log formatting) that may be ported later, once SPEC.md says what it must produce.

Work goes in phases. **Each phase ends with a stop: report, then wait for an author's approval before starting the next.** Do one phase per session unless told otherwise.

---

## Phase 0 — Set up the repository

Goal: freeze the legacy code and create the new layout, without changing any legacy file's content.

1. Work on a new branch, `rebuild`.
2. Propose a tag for the current state of the default branch, for example `legacy-2026-08`, and ask the author to confirm before creating it.
3. Move every existing code and script file into `legacy/` with `git mv`, so that history is kept. Change no file contents. Leave `CLAUDE.md` and `docs/` where they are: they belong to the rebuild.
   - If any file looks like data, output or configuration that should not move, list it and ask.
4. Create the layout in SPEC §11: `igss/`, `tests/` and `docs/`.
   - Add `outputs/` to `.gitignore`.
   - Do not create module files yet.
5. Write a short `README.md` for co-authors, labelled as an AI-generated draft. It says what the repository is and points to `docs/SPEC.md` as the single source of truth, `CLAUDE.md` as the assistant's instructions, and `legacy/` as frozen.

**Stop.** Report the moves, the proposed tag, and the README for review.

## Phase 1 — Inventory the legacy code (read-only)

Goal: an accurate map of what exists. Do not modify anything in `legacy/`, and do not run long jobs.

Write `docs/legacy_inventory.md`, labelled as an AI-generated draft. It covers every file in `legacy/`, including R scripts, notebooks and data files. For each file give:

- its purpose, in one or two sentences, from reading the code rather than its name or comments;
- its inputs and outputs: which files it reads and writes, with column names for CSVs;
- its dependencies: packages with versions where they are pinned, and other legacy files it imports or sources;
- how it is run (command, entry point), if that can be determined;
- whether it belongs to the **model core** (rules, episode or simulation, scoring or fitness, search or evolution, configuration) or is **peripheral** (logging, plotting, summaries, utilities).

Then add three short sections:

1. **Environment.** The Python and R versions and packages the legacy code needs. Propose a `requirements.txt` for the new code with only what SPEC.md needs (DEAP, SymPy, pytest; anything else must be argued for), but do not commit it yet.
2. **What the outputs look like.** The figures and tables the legacy plotting and logging code produces: what each shows, from which columns, with which grouping. This is raw material for specifying figures in SPEC §10.
3. **Unclear items.** Anything you could not determine from reading the code.

**Stop.** Report the inventory.

## Phase 2 — Compare legacy with the specification

Goal: every difference between the legacy code and SPEC.md made visible, and every choice the rebuild will need identified, before any new code exists.

Write `docs/rebuild_plan.md`, labelled as an AI-generated draft, with five sections.

1. **Silent choices in the legacy code.** List every place where the legacy code makes a choice that SPEC.md does not make. Examples: defaults, row orders, how partners are drawn, whether a donor can meet itself, when stored values update, how fitness is averaged, seeds, tie-breaking, constants typed into scripts. For each, give:
   - file and line;
   - what the legacy code does;
   - whether SPEC.md decides it (cite the section), contradicts it, or is silent.
2. **Known problems re-checked.** For each known problem in SPEC §17:
   - confirm it, citing file and line, or report that you could not reproduce it;
   - add any further mismatches you find, kept separate from the confirmed list.
3. **Questions for the decision register.** Every choice SPEC.md leaves open that mode 1 will need, written as proposed rows for SPEC §13. Each row gives topic, proposed choice, alternatives, and why it matters. Include the "proposed" rows already in §13 that you think need an author's attention first. **Do not resolve any of them yourself.**
4. **Peripheral code to port later.** For each peripheral legacy file worth keeping:
   - what it would take to adapt it to the new outputs (SPEC §10);
   - which figure or table in SPEC §10 it would serve, once that section is written.
   - Then propose text for a "Figures" subsection of SPEC §10, as a suggestion for the authors.
5. **Build sequence.** Confirm or amend the order of work in SPEC §11 for mode 1. Note any test you would add, but do not add tests the authors have not approved.

Put proposed changes to SPEC.md in the plan, as quoted text marked "proposed". Do not edit SPEC.md in this phase.

**Stop.** Report the plan. The authors will update SPEC.md (or ask you to apply agreed changes) before phase 3.

## Phase 3 — Build mode 1, one module at a time

Follow SPEC §11 and the working loop in `CLAUDE.md`. For each module, in order (`rules.py`, `episode.py`, `baseline.py`, `scoring.py`, `search.py`, `config.py`, `run.py`):

1. Plan the module against the SPEC sections it implements, and wait for approval.
2. Write the tests listed in SPEC §11 first, and show them failing.
3. Write the module, and show the tests passing.
4. List every assumption not stated in SPEC.md.
5. Commit with a message naming the SPEC section.
6. **Stop** for review. A fresh session may then review the change against SPEC.md (SPEC §12, item 7).

Smoke runs of a few generations are allowed, to check that the code runs. Their numbers are not results and are not written into any doc.

## Phase 4 — Outputs and figures

Start only after SPEC §10 has an agreed "Figures" subsection.

- Port or rewrite the peripheral code identified in phase 2 so that it reads the new CSV outputs.
- Keep the R scripts in R unless the authors decide otherwise.
- The same loop applies: plan, tests where they make sense (for example, that a figure script runs on a small fixture CSV), one change per commit, and a stop for review.

## Out of scope for this brief

- Modes 2 and 3, the combined DR+IR model, junk inputs and third-order IR (SPEC §14–15).
- Full sweeps. The authors run them on their own machines from committed code.
- Any edit inside `legacy/`.
