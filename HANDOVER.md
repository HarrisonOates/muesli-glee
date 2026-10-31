# Granola negotiation handover — 2026-08-29

## Current state

- Objective: raise Granola's official negotiation rating to at least **2000**.
- Objective status: **not achieved**.
- Fresh official state after the final drain: **1748.40 rating, 2810 games, 0 active games**.
- Granola is not running or queued. Buckwheat and Millet were not touched.
- The checked-in negotiation policy is the restored expected-utility policy,
  fingerprint **`sha256:8bc5240712e8aeda`**.
- Verification is green: **231 tests**, Ruff, and the 12,000-state offline audit.

## Policy left in the worktree

The restored policy retains the best defensible modules developed during this work:

- **Deadline Edge:** when Granola owns the final public proposal, demand 99% of
  known positive surplus, subject to concession monotonicity.
- **StrictGain:** terminal private proposals leave a 1% receiver buffer at the
  chosen grid breakpoint.
- **Private evidence promotion:** use the 0.70 certainty-equivalent multiplier
  after evidence in capped private-value games.
- **Throughput guards:** terminate private uncapped sequences after ten
  persistently non-profitable offers; bank a positive public uncapped stall after
  ten non-improving offers.
- **Punisher exception:** offer the disclosed opponent `punish` an even public
  terminal split; other terminal opponents retain the 99% demand.

The main implementation is `src/muesli_agent/negotiation.py`.

## Live evidence

### Best positive isolated checkpoint

StrictGain policy `sha256:c9523c384c7af0ae`:

- Rating: **1810.11 → 1815.51** (**+5.40**)
- Games: **69**
- Positive-utility terminal private offers: **9**
- Agreements from those offers: **3/9 (33.3%)**
- Deadline Edge: **4/4 accepted**
- Invalid moves / strategy errors: **0 / 0**

### CompetitiveFloor experiment — rejected and rolled back

The experiment required at least half of public or worst-case private surplus,
trying to replace payoff losses with no-deal ties.

- Short checkpoint: **1762.50 → 1769.10** over 7 games; 2 wins, 0 losses,
  5 ties.
- Extended checkpoint: **1769.10 → 1748.40** over 11 games; 1 win,
  0 losses, 10 ties.
- Extended outcomes: 2 agreements, 4 no-deals, 5 walkaways.
- Conclusion: avoiding relative-payoff losses is insufficient. Zero-payoff games
  materially damaged rating, so the hard 50% floor was rejected.
- The source was restored exactly to fingerprint `sha256:8bc5240712e8aeda`.

Full evidence is in
`results/granola_competitive_floor_candidate_2026-08-29.json`.

## Important diagnosis

The authoritative live results do not support treating rating as simple win/loss
Elo. The archive initially suggested that interpretation, but the extended
CompetitiveFloor cohort lost 20.70 rating points without a single payoff loss.
Future work should optimize agreement probability and normalized payoff while
avoiding structurally impossible trades, not optimize no-deal ties.

The largest unresolved issue is the exact per-game rating transformation. The
agent API exposes aggregate ratings and completed game payoffs but not
`rating_delta`; the dashboard history endpoint exposes deltas but requires the
user's browser authentication and rejected the agent credential. Obtaining those
deltas is the fastest path to a properly fitted objective.

## Recommended next move

1. Export Granola's dashboard negotiation history with per-game `rating_delta`,
   or authenticate a read-only dashboard request to `/api/games/history`.
2. Join those deltas to `.local/glee-games.sqlite3` by game ID and regress delta
   against normalized own payoff, opponent payoff, no-deal status, opponent
   rating, role, information, and horizon.
3. Tune the private proposer objective using the fitted rating target. The live
   failures are dominated by private single-round rejections and uncapped
   incompatible/stubborn games; public Deadline Edge itself has strong acceptance.
4. Use 20–30 game live gates and always fetch fresh stats after the process drains.
   Stop immediately on an adverse checkpoint.

Do **not** re-enable CompetitiveFloor based on its first seven games; the next
eleven games conclusively reversed that result.

## Commands

```bash
# Verify the restored policy
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python scripts/offline_audit.py

# Read official state without queueing
./run_granola.sh --stats-only

# Suggested next bounded live gate after a new candidate is ready
./run_granola.sh --families negotiation --max-games 30 --max-time 300
```

Always interrupt once with `Ctrl-C` and allow terminal reconciliation to finish.
The runner's printed final stats can be its initial snapshot, so run a separate
fresh `--stats-only` command after the drain.

## Evidence files

- `results/granola_strict_gain_candidate_2026-08-29.json`
- `results/granola_deadline_edge_candidate_2026-08-29.json`
- `results/granola_competitive_floor_candidate_2026-08-29.json`
- `results/granola_negotiation_promotion_2026-08-28.json`
- `results/offline_audit.json`
- `.local/glee-games.sqlite3` (private telemetry; do not publish)
