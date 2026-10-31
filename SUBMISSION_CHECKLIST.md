# GLEE entry and paper checklist

The live competition closes **August 29, 2026 AoE**. Competition papers are due
**September 5, 2026 AoE** in the IAB competition-paper track on OpenReview (extended
from September 1).

## Account actions (must be done by the account owner)

- [ ] Confirm every author already has an active OpenReview profile. The organizers
  warn that moderation can take up to two weeks, so this is urgent.
- [ ] Sign in at <https://glee-competition.com>, accept the Terms of Service, and use
  one team/account only.
- [x] Create the autonomous agent `Granola` and identify its policy as MUESLI in the paper.
- [x] Save its one-time `glee_...` key in the ignored, owner-readable `.env` file.
- [ ] Create the experimental `Buckwheat` agent under the same account (GLEE allows
  up to five), save its distinct key in ignored `.env.challenger`, and never create a
  second account.
- [x] Copy the public agent ID from **My Agents** into `paper/main.tex` and the
  OpenReview submission form.
- [ ] Fill the author affiliation and email `TODO`s in `paper/main.tex`.

## Safe launch sequence

```bash
cd /home/harrison/Documents/GitHub/glee-competition-entry
export GLEE_API_KEY=glee_your_real_key_here
.venv/bin/muesli-glee --stats-only
.venv/bin/muesli-glee --families bargaining --concurrency 1 --max-games 3
.venv/bin/muesli-glee --families negotiation --concurrency 1 --max-games 3
.venv/bin/muesli-glee --families persuasion --concurrency 1 --max-games 3
```

- [x] Verify no invalid agent moves or remaining active games after the first smoke run.
- [x] Inspect game histories for offers, counteroffers, and persuasion signals that
  match the behavior described in the paper.
- [ ] Only then raise concurrency (the repository default is 4).
- [ ] Keep a top-100 agent active enough to satisfy the platform's per-family activity
  and rating-decay rules.
- [ ] Give Buckwheat the preregistered 10-game-per-family sequential budget before
  changing any challenger module; keep Granola frozen for the comparison.

## Before the paper deadline

- [x] Export game histories and ratings after a meaningful number of games per role
  and family.
- [x] Replace the initial online-results placeholder with the 25-game telemetry cohort.
- [x] Re-run `pytest`, `ruff`, and `scripts/offline_audit.py` at the current candidate
  commit.
- [x] Build the paper and confirm the main text is at most four pages; references may
  continue beyond page four.
- [x] Read the PDF end to end and verify every number against its source artefact.
- [x] Review current NeurIPS research-ethics and generative-AI policies. The 2026
  author policy permits tools for preparing and writing a paper without disclosure;
  authors remain responsible for correctness, originality, citations, and the ban on
  review-manipulating prompt injections. No paper disclosure is required for editing
  assistance.
- [x] Replace the pre-close paper ratings with the account owner's final positions:
  35/198 overall, 10/198 bargaining, 62/198 negotiation, and 35/198 persuasion.
- [ ] Submit to the
  [IAB Competition Paper Track](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/IAB_Competition_Paper_Track)
  by September 5 AoE.
