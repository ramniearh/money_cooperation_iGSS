# Instructions for Claude Code — iGSS rebuild

<!-- AI-generated draft (Claude, Anthropic) for checking and amendment by the authors. -->

This repository rebuilds the iGSS models of direct reciprocity (DR) and indirect reciprocity (IR) from scratch.

- The specification is `docs/SPEC.md`. It is the single source of truth: the code must match it, and any mismatch is a bug.
- This file holds standing instructions only. It does not describe the model.

## Before any task

- Read the sections of `docs/SPEC.md` that the task touches, and always §3 (vocabulary), §5 (truth tables) and §13 (decision register).
- For rebuild work, follow `docs/REBUILD_BRIEF.md` and do only the phase you were asked to do.
- For a new phase, a new mode or a change of design, start in plan mode and wait for approval. Routine work inside an agreed build does not need a plan first.

## Hard rules

1. **No silent choices.** Every choice the code needs that SPEC.md does not make is written into SPEC §13.
   - If it can change which rules evolve (scoring, error, assessment rules, selection pressure, what is swept), stop and ask, and write no code that depends on it until an author confirms.
   - If it is a DEAP mechanic or cannot change which rules evolve (ids, orderings, seed handling, column names, file formats), make an informed choice, record it in §13 with status "assistant", and report it at the end of the task. Explain DEAP behaviour when reporting it: the authors are learning DEAP.
   - When unsure which kind it is, ask.
   - Where the legacy code and SPEC.md disagree, follow SPEC.md and report it afterwards.
2. **Spec before code.** An agreed change in behaviour goes into SPEC.md, including its change log (§18), before the code or in the same commit.
3. **Bit strings are derived, never typed.** Define each named rule as a predicate on named inputs and derive its bit string. Row order: all-ones first, first input most significant. The only literal bit strings allowed are in tests, as expected values copied from SPEC.md.
4. **`legacy/` is read-only.** Read it for reference. Never edit it, import from it or "fix" it. Port nothing from it into the model core (`rules`, `episode`, `scoring`, `baseline`, `search`). Peripheral code (plotting, logging) may be ported only once SPEC.md specifies what it must produce.
5. **Vocabulary.** Use the terms in SPEC §3 in code, comments and docs: donor, partner, action rule, assessment rule, iGSS agent, background agent, episode, round, generation, run, environment, scoring option.
   - Never use "recipient", "institution", "world" or "target".
   - Literature names (Tit-for-Tat, Image Scoring, and so on) appear only in output labels derived from SPEC §16.
6. **Scoring is always explicit.**
   - `score_individual` and `score_aggregate` are separate named functions.
   - `SCORING` has no default, so a run without it must fail.
   - Every output row records the scoring option used.
7. **Every parameter in `config.py` is read by the code.** A parameter that does nothing is a bug.
8. **No findings.** Run tests and short smoke runs only; the authors run full sweeps. Never describe output as a result, and never write numbers into docs unless an author produced them with committed code.
9. **Writing for the authors.** Docs you write for the authors (plans, inventories, reviews):
   - open with a line labelling them as AI-generated drafts;
   - are written impersonally, never in an author's voice;
   - use British spelling, as SPEC.md does.

## Working loop

1. Agree the design points first (see hard rule 1).
2. Write the tests for the module first, using the tests listed in SPEC §11. Then write the code, run the tests and show the output.
3. Make one module or one change per commit. The commit message names the SPEC section it implements, for example `episode: implement one round (SPEC §4)`.
4. After each change, list every assumption you made that SPEC.md does not state. Do this even when the list seems empty.
5. Several modules may be built in one task when an author asks for it. Report at the end of the task, with a short smoke run the author can look at.

## Commands

- Install: `pip install -r requirements.txt`
- Tests: `python -m pytest`
- One quick run: `python -m igss.run quick --mode 1 --model DR --scoring aggregate`
- List a sweep without running it: `python -m igss.run dry-run first`
- Run one part of a sweep (for a cluster): `python -m igss.run sweep main --part 3/40 --workers 8`
- Merge the parts of a sweep: `python -m igss.run merge main <part directories>`
- Rerun and compare a sample of a sweep's runs: `python -m igss.run check <sweep directory> --sample 20`
- Figures: `Rscript R/figures.R <sweep directory> --baseline <baseline directory>`

## Gotchas

- DEAP's operators draw from Python's global `random` module. Episodes must use their own generator, following the seed scheme in SPEC §13.
- When DEAP's `staticLimit` rejects an over-deep child, it returns one of the parents. It does so only in the operator's return value: `children[i], = toolbox.mutate(children[i])`. Calling the operator and ignoring what it returns leaves the over-deep child in the population.
- Under aggregate scoring, all P episodes of a generation share one seed. That is why rules with the same bit string get identical scores. This is intended, not a bug.
- In modes 2 and 3, many unnamed strategies behave exactly like a named one (SPEC §10, "A caution on unnamed"). Check table T3 and the baseline before reading anything into the share of "unnamed".
- When no agent ever defects, `10` and `11` cannot be told apart (SPEC §6.3). Check the logged input counts before reading anything into their relative frequency.
