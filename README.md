# MUESLI: a structural GLEE competition agent

MUESLI (Model-based Utility Estimation with Strategic Lookahead and Inference) is a
transparent, zero-LLM-cost entry for the
[GLEE Competition](https://glee-competition.com/). It
handles bargaining, bilateral negotiation, and repeated persuasion through one
`glee-sdk` process.

This repository is deliberately a baseline we can iterate on. The strategy has no
cross-game mutable state, reconstructs opponent evidence from each game's history,
and places API legality and payoff guards before strategic optimization.

## Quick start

Python 3.10 or newer is required.

```bash
cd /home/harrison/Documents/GitHub/glee-competition-entry
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Create an agent in the [GLEE dashboard](https://glee-competition.com/dashboard),
accept the competition terms, and save the one-time API key. Keep it in the shell,
not in this repository:

```bash
export GLEE_API_KEY=glee_your_real_key_here
.venv/bin/muesli-glee --stats-only
```

Start with a bounded, single-game smoke run:

```bash
.venv/bin/muesli-glee --concurrency 1 --max-games 3
```

Once the dashboard confirms valid completed games, a conservative continuous run is:

```bash
./run_granola.sh
```

The launcher loads `.env`, defaults to concurrency four, and forwards any extra CLI
arguments. For example, `./run_granola.sh --families persuasion --max-games 20` runs a
bounded persuasion-only experiment. Set `GLEE_CONCURRENCY=1` before the command to
override its default. The SDK's `--max-games` counter advances when a terminal result
arrives on our move; games ending on the opponent's move are recovered by telemetry
but can make the actual match count exceed that target. Use `--max-time` as the hard
queueing bound when an exact exposure limit matters.

Every live run is archived automatically in the ignored private database
`.local/glee-games.sqlite3`. Use Ctrl-C once and allow MUESLI to leave the queues and
drain in-flight games; another Ctrl-C while the drain is running forces an immediate
exit. The platform allows
60 requests per minute per agent; the default two-second polling interval and
concurrency of four are chosen to stay modest. A local process lock rejects an
accidental second live runner using the same archive.

Useful variants:

```bash
# Isolate one family while tuning
.venv/bin/muesli-glee --families bargaining --max-games 20

# Bound a run to roughly one hour (in-flight games are still drained)
.venv/bin/muesli-glee --max-time 3600
```

## Isolated challenger agent

GLEE permits up to five agents on one account and ranks the account by its best
agent. MUESLI includes an experimental challenger profile, provisionally named
**Buckwheat**, for a preregistered comparison with Granola. Create it under the same
GLEE account (never create a second account), then save its distinct one-time key:

```bash
cp .env.challenger.example .env.challenger
chmod 600 .env.challenger
# Edit .env.challenger, then verify the new key without queueing:
./run_challenger.sh --stats-only
```

The challenger has a separate policy fingerprint, retry-stable signaling key,
process identity, and private archive at `.local/challenger-games.sqlite3`. Its
launcher always selects `--profile challenger`; the ordinary `run_granola.sh` and
default CLI profile remain Granola.

Start with equal, sequential family budgets so configuration-relative outcomes can
be compared without simultaneous-run timing effects:

```bash
GLEE_CHALLENGER_CONCURRENCY=1 ./run_challenger.sh --families bargaining --max-games 10
GLEE_CHALLENGER_CONCURRENCY=1 ./run_challenger.sh --families negotiation --max-games 10
GLEE_CHALLENGER_CONCURRENCY=1 ./run_challenger.sh --families persuasion --max-games 10

.venv/bin/python scripts/analyze_telemetry.py \
  --database .local/challenger-games.sqlite3
```

Before launch, reproduce the challenger safety audit and same-observation archive
comparison with:

```bash
.venv/bin/python scripts/offline_audit.py --profile challenger
.venv/bin/python scripts/compare_profiles.py \
  --output .local/profile-comparison.json
```

The comparison reports action divergences only; it deliberately makes no
counterfactual-payoff claim because the opponent could respond differently.

The experiment is intentionally narrow: repeated-offer discount protection in
bargaining, a complete-information unlimited-horizon surplus floor in negotiation,
and template-aware/exact-indifference safeguards in persuasion. Treat rating as a
secondary measure; compare normalized payoff, agreement/trade rates, and buyer
outcomes within role and information strata before retaining any module.

## Strategy at a glance

- **Bargaining:** combines finite-horizon rollout or an unknown-horizon Rubinstein
  reference with a discrete posterior over the opponent's acceptance threshold.
  Rejections, counterclaims, and the pending offer update the posterior; conservative
  terminal floors limit brittle no-deal bets until live evidence supports them.
- **Negotiation:** searches offers in own-utility space against a discrete
  concession-threshold belief. Private-information games use the competition's
  valuation-factor grid, while rejection-conditioned continuation, monotone
  concessions, and final-round guards apply symmetrically to buyer and seller.
  In capped private-value games, opponent evidence activates the lower
  continuation-value multiplier validated by the isolated Millet profile. Terminal
  private offers leave a strict-positive receiver buffer at grid breakpoints, and a
  conservative loss-only patience bound prevents incompatible uncapped games from
  consuming a worker through the platform safety cap. The public 55/45 bracket also
  banks a positive offer after ten consecutive non-improving observations rather than
  repeating a rejected counter to the safety cap. A replay-stable opponent-specific
  guard offers the disclosed fairness punisher an even terminal split; all other
  terminal proposers retain the validated 99% surplus demand.
- **Persuasion:** privately mixes low-quality recommendations within a
  reputation-adjusted Bayesian-persuasion cap. The buyer combines all observed signal
  frequencies with coverage-shrunk outcomes, conservative text classification, and
  explicit known/unknown-horizon handling.

The dispatcher is in [`src/muesli_agent/strategy.py`](src/muesli_agent/strategy.py), with
the family policies in [`bargaining.py`](src/muesli_agent/bargaining.py),
[`negotiation.py`](src/muesli_agent/negotiation.py), and
[`persuasion.py`](src/muesli_agent/persuasion.py). Their exposed experiment controls are
`BargainingConfig`, `NegotiationConfig`, and `PersuasionConfig`.

## Verification

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/python scripts/offline_audit.py
```

For negotiation-only parameter screening against paired algorithmic opponent
models, run:

```bash
.venv/bin/python scripts/simulate_negotiation.py --scenarios 500
```

This simulator is a development stress test, not a leaderboard estimate. It
reports agreement rate and normalized own payoff for Boulware, conceding,
tit-for-tat, fixed-threshold, and walkaway-sensitive opponents.

The checked-in audit result at [`results/offline_audit.json`](results/offline_audit.json)
covers 12,000 seeded legal states. It is a policy safety and behavior audit, **not** a
leaderboard-performance estimate. The first bounded live run is recorded separately in
[`results/live_smoke_2026-08-23.json`](results/live_smoke_2026-08-23.json), the next
pre-telemetry iteration in
[`results/live_iteration_2026-08-23.json`](results/live_iteration_2026-08-23.json), and
the final-policy telemetry pilot in
[`results/live_telemetry_pilot_2026-08-23.json`](results/live_telemetry_pilot_2026-08-23.json),
and the full development cohort in
[`results/live_final_cohort_2026-08-23.json`](results/live_final_cohort_2026-08-23.json).
The subsequent 34-game run is recorded separately in
[`results/live_continuation_run_2026-08-23.json`](results/live_continuation_run_2026-08-23.json).
The live evidence, replay scope, synthetic tradeoff, and rollback gate for the
private-value negotiation promotion are recorded in
[`results/granola_negotiation_promotion_2026-08-28.json`](results/granola_negotiation_promotion_2026-08-28.json).
The terminal-proposer evidence and live gate for the higher-upside Deadline Edge
candidate are recorded in
[`results/granola_deadline_edge_candidate_2026-08-29.json`](results/granola_deadline_edge_candidate_2026-08-29.json).
The private terminal-buffer evidence and throughput guard are recorded in
[`results/granola_strict_gain_candidate_2026-08-29.json`](results/granola_strict_gain_candidate_2026-08-29.json).
The current suite contains 231 tests. The August 23 archive cohort persisted 683 valid
moves across 59 completed games with no invalid actions, strategy errors, transport
errors, or mixed-policy games.
The non-sensitive aggregate evidence frozen for the paper, including the pre-close
ratings and full-archive move-health counts, is recorded in
[`results/paper_evidence_snapshot_2026-08-30.json`](results/paper_evidence_snapshot_2026-08-30.json).
The account owner's final overall and family leaderboard positions used in the revised
paper are recorded separately in
[`results/final_leaderboard_positions_2026-08-30.json`](results/final_leaderboard_positions_2026-08-30.json).

The frozen challenger hypotheses, evaluation budget, stop conditions, and offline
verification are recorded before its first live game in
[`results/challenger_preregistration_2026-08-23.json`](results/challenger_preregistration_2026-08-23.json).
The resulting 55-game frozen-policy cohort, including family gates and adverse
findings, is recorded separately in
[`results/live_challenger_cohort_2026-08-23.json`](results/live_challenger_cohort_2026-08-23.json).

## Private live-game telemetry

MUESLI persists each visible game state, selected action, API response, opponent
disclosure, terminal result, configuration slice, run-level rating change, and a hash
of the exact policy source. Games that end on the opponent's move are reconciled through
the SDK's read-only game-state endpoint. Storage is initialized before matchmaking;
after that, actions are submitted before telemetry writes, so a full disk or damaged
archive cannot make the agent forfeit a turn.

Summarize the archive by policy, family, role, information condition, horizon, and
opponent type. The report also includes per-run family outcomes, active-game counts,
and move health; ratings for an unfinished run remain null until its terminal drain:

```bash
.venv/bin/python scripts/analyze_telemetry.py
```

Export self-contained JSON Lines for deeper analysis or paper figures:

```bash
.venv/bin/python -m muesli_agent.telemetry \
  --output .local/glee-games.jsonl \
  --summary
```

Set `MUESLI_POLICY_VERSION=experiment-name` to attach a human-readable experimental
label instead of the automatic source hash. Set `GLEE_GAME_DB=/private/path/games.db`
or pass `--telemetry-db` to move the archive. The database and exported transcripts can
contain opponent messages and are permission-restricted to the local user; keep them
private and do not commit `.local/`.

## Paper

The competition-paper draft is in [`paper/main.tex`](paper/main.tex), using the
official unmodified NeurIPS 2026 style. Build it with:

```bash
make paper
```

Before submission, fill the remaining author `TODO`s in the paper and follow
[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md). In particular, the author
affiliation and email require information that is not stored here.

## Reproducibility and secrets

- The agent never reads an LLM key or calls an external model.
- Low-quality persuasion pooling uses a SHA-256 draw derived from game ID and round,
  so identical inputs produce identical actions.
- `.env` and logs are ignored. Do not paste the GLEE key into code, issues, or the
  paper.
- `.local/` is ignored because it contains private game transcripts and experiment
  metadata.
- The official `neurips_2026.sty` is vendored unchanged from the
  [NeurIPS 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
