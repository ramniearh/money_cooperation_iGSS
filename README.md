# iGSS: direct and indirect reciprocity

*AI-generated draft (Claude, Anthropic), for checking and amendment by the authors.*

This repository rebuilds the iGSS (inverse generative social science) models of direct reciprocity (DR) and indirect reciprocity (IR) from scratch, against a written specification. Genetic programming searches for action rules, and the question is whether the search reliably finds the rules the literature calls canonical.

## Where things are

| Path | What it is |
| --- | --- |
| `docs/SPEC.md` | The specification. It is the single source of truth: the code must match it, and any mismatch is a bug. Changes in behaviour are made there first and recorded in its change log (§18). |
| `docs/REBUILD_BRIEF.md` | The phased task description for the rebuild. |
| `CLAUDE.md` | Standing instructions for the AI coding assistant (Claude Code). It does not describe the model. |
| `igss/` | The new code. Empty until mode 1 is built (SPEC §11). |
| `tests/` | One test file per module in `igss/`. |
| `legacy/` | The code as it stood before the rebuild, frozen at git tag `legacy-2026-11`. Read-only: it is kept for reference and is not edited, imported or fixed. Its original README is `legacy/README.md`. Known problems are listed in SPEC §17. |
| `outputs/` | Sweep results. Not committed. |

## Status

No new code exists yet. The rebuild proceeds in the phases set out in `docs/REBUILD_BRIEF.md`, each ending in a review by the authors.
