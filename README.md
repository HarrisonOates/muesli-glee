# MUESLI: Symbolic Reasoning for Language-Based Economic Games

Code for **MUESLI** (*Model-based Utility Estimation with Strategic Lookahead and
Inference*), a symbolic-first entry to the [GLEE Competition](https://glee-competition.com/)
described in our paper at the NeurIPS 2026 Workshop on Interpreting Agent Behavior (IAB).

MUESLI plays all three GLEE families (bargaining, bilateral negotiation, and repeated
persuasion) **without an LLM at decision time**. Each turn is a deterministic function
of the visible game state and history: explicit beliefs over opponent thresholds,
valuations, or product quality are rebuilt from scratch, candidate actions are scored by
expected utility, and hard legality and individual-rationality guards are applied before
an optional templated message is attached.

## Competition agents

MUESLI was entered under one GLEE account as three near-identical agents:

| Agent | Profile | Description |
| --- | --- | --- |
| **Granola** | `--profile granola` (default) | The primary MUESLI policy. |
| **Buckwheat** | `--profile buckwheat` | A preregistered challenger that changes three narrow cells and otherwise delegates to Granola. |
| **Millet** | `--profile millet` | An isolated challenger for one negotiation change (accepting profitable offers more readily after opponent evidence in capped private-value games); since promoted into Granola. |

Buckwheat's changes are *DiscountLatch* (bargaining), *NashFloorBracket* (negotiation),
and *TemplateBayesExactGuard* (persuasion); see
[`results/challenger_preregistration_2026-08-23.json`](results/challenger_preregistration_2026-08-23.json).
Millet's negotiation change now lives in Granola (`_certainty_equivalent_discount` in
`negotiation.py`; evidence in
[`results/granola_negotiation_promotion_2026-08-28.json`](results/granola_negotiation_promotion_2026-08-28.json)),
so in this release the `millet` profile runs the same code as Buckwheat. `challenger` is
an older alias.

Among the 198 agents on the final leaderboard, the team finished **35th overall**,
**10th in bargaining**, **35th in persuasion**, and **62nd in negotiation**.

## Repository layout

```
src/muesli_agent/
  strategy.py      family dispatch (Granola)
  bargaining.py    Rubinstein / backward-induction reference + threshold posterior
  negotiation.py   latent-valuation threshold belief + expected-payoff offer search
  persuasion.py    reputation-budgeted pooling (seller) and Bayesian buying (buyer)
  challenger.py    Buckwheat overrides
  cli.py           live runner (glee-sdk)
  telemetry.py     private local game archive
scripts/           offline audit, telemetry analysis, profile comparison, simulators
tests/             unit and behavioral-invariant tests
results/           aggregate evidence used in the paper (no game transcripts)
```

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Verification

```bash
make test     # pytest
make lint     # ruff
make audit    # 12,000 randomized states checked for legality and payoff invariants
```

`scripts/offline_audit.py --profile challenger` runs the same audit for Buckwheat.

## Running an agent

Create an agent in the [GLEE dashboard](https://glee-competition.com/dashboard), accept
the competition terms, and put its one-time key in an ignored env file:

```bash
cp .env.example .env && chmod 600 .env   # then set GLEE_API_KEY
./run_granola.sh --stats-only            # check the key without queueing
./run_granola.sh --concurrency 1 --max-games 3
```

`./run_buckwheat.sh` and `./run_millet.sh` do the same using `.env.buckwheat` and `.env.millet`. Extra arguments
are forwarded to `muesli-glee`, for example `--families bargaining` or `--max-time 3600`.
Press Ctrl-C once to stop queueing and let in-flight games drain.

## Telemetry and privacy

Live runs record every observation, action, and result in a local SQLite archive under
`.local/` so decisions can be replayed exactly. These archives contain other
participants' messages and are deliberately **not** included in this repository; only
aggregate, de-identified summaries appear in `results/`. Analyse your own archive with:

```bash
.venv/bin/python scripts/analyze_telemetry.py
```

## Paper

The paper is available from [harrisonoates.com](https://harrisonoates.com/papers/iabneurips2026muesli/).

```bibtex
@inproceedings{oates2026muesli,
  title     = {{MUESLI}: Symbolic Reasoning for Language-Based Economic Games},
  author    = {Oates, Harrison},
  booktitle = {NeurIPS 2026 Workshop on Interpreting Agent Behavior (IAB)},
  year      = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
