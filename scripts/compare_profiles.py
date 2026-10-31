#!/usr/bin/env python3
"""Replay archived observations through Granola and the isolated challenger.

The report counts action divergences only. Opponent responses to counterfactual
actions are unobserved, so this script deliberately makes no payoff-improvement
claim.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from muesli_agent.challenger import challenger_strategy
from muesli_agent.strategy import strategy as granola_strategy
from muesli_agent.telemetry import SOURCE_FILES, default_database_path, policy_fingerprint


def _label(action: dict[str, Any]) -> str:
    decision = action.get("decision")
    if decision is not None:
        return str(decision)
    if "product_price" in action:
        return "offer"
    if "alice_gain" in action or "player_1_gain" in action:
        return "offer"
    if "message" in action:
        return "message"
    return "other"


def build_report(database: Path) -> dict[str, Any]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT g.family, m.phase, m.observation_json
            FROM moves AS m
            JOIN games AS g ON g.game_id = m.game_id
            ORDER BY m.move_id
            """
        ).fetchall()

    family_totals: Counter[str] = Counter()
    family_divergences: Counter[str] = Counter()
    phase_totals: Counter[str] = Counter()
    phase_divergences: Counter[str] = Counter()
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()

    for row in rows:
        family = str(row["family"])
        phase = str(row["phase"] or "unknown")
        family_totals[family] += 1
        phase_totals[f"{family}:{phase}"] += 1
        try:
            observation = json.loads(row["observation_json"])
            granola = granola_strategy(observation)
            challenger = challenger_strategy(observation)
        except Exception as error:  # noqa: BLE001 - aggregate replay diagnostics
            errors[f"{family}:{type(error).__name__}"] += 1
            continue
        if granola == challenger:
            continue
        family_divergences[family] += 1
        phase_divergences[f"{family}:{phase}"] += 1
        transitions[family][f"{_label(granola)}->{_label(challenger)}"] += 1

    return {
        "description": (
            "Same-observation policy replay; divergences are not counterfactual payoff claims."
        ),
        "database": str(database.resolve()),
        "granola_policy": policy_fingerprint(),
        "challenger_policy": policy_fingerprint((*SOURCE_FILES, "challenger.py")),
        "moves": len(rows),
        "replay_errors": sum(errors.values()),
        "errors": dict(sorted(errors.items())),
        "families": {
            family: {
                "moves": family_totals[family],
                "divergences": family_divergences[family],
            }
            for family in sorted(family_totals)
        },
        "phases": {
            phase: {
                "moves": phase_totals[phase],
                "divergences": phase_divergences[phase],
            }
            for phase in sorted(phase_totals)
        },
        "action_transitions": {
            family: dict(sorted(counts.items())) for family, counts in sorted(transitions.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_database_path())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.database)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        args.output.chmod(0o600)
        print(f"Wrote profile comparison to {args.output}")
    else:
        print(rendered)
    return int(report["replay_errors"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
