#!/usr/bin/env python3
"""Summarize MUESLI's private live-game archive without external dependencies."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from muesli_agent.telemetry import default_database_path


def _load(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _rating_changes(row: sqlite3.Row) -> dict[str, Any]:
    initial = _load(row["initial_stats_json"]) or {}
    final = _load(row["final_stats_json"]) or {}
    initial_scores = initial.get("scores", {})
    final_scores = final.get("scores", {})
    changes = {}
    for family in sorted(set(initial_scores) | set(final_scores)):
        before = initial_scores.get(family, {})
        after = final_scores.get(family, {})
        changes[family] = {
            "rating_before": before.get("rating"),
            "rating_after": after.get("rating"),
            "rating_change": _difference(after.get("rating"), before.get("rating")),
            "games_before": before.get("games_played"),
            "games_after": after.get("games_played"),
            "games_added": _difference(after.get("games_played"), before.get("games_played")),
        }
    return changes


def _difference(after: Any, before: Any) -> float | None:
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None


def _normal_payoff(row: sqlite3.Row, configuration: dict[str, Any]) -> float | None:
    payoff = row["own_payoff"]
    if payoff is None:
        return None
    family = row["family"]
    if family == "bargaining":
        scale = configuration.get("money_to_divide")
    elif family == "negotiation":
        seller = configuration.get("player_1_value")
        buyer = configuration.get("player_2_value")
        try:
            scale = float(buyer) - float(seller)
        except (TypeError, ValueError):
            return None
    elif family == "persuasion":
        rounds = configuration.get("total_rounds")
        if row["role"] == "seller":
            unit_scale = configuration.get("product_price")
        else:
            try:
                unit_scale = float(configuration.get("v")) - float(configuration.get("u"))
            except (TypeError, ValueError):
                return None
        try:
            scale = float(rounds) * float(unit_scale)
        except (TypeError, ValueError):
            return None
    else:
        return None
    try:
        scale_number = float(scale)
    except (TypeError, ValueError):
        return None
    if scale_number <= 0:
        return None
    return float(payoff) / scale_number


def build_report(database: Path) -> dict[str, Any]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        runs = connection.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        games = connection.execute("SELECT * FROM games ORDER BY first_seen_at").fetchall()
        move_health = connection.execute(
            """
            SELECT COUNT(*) AS moves,
                   SUM(CASE WHEN response_json IS NOT NULL
                                 AND json_extract(response_json, '$.valid') = 0
                            THEN 1 ELSE 0 END) AS invalid_moves,
                   SUM(CASE WHEN error_stage = 'strategy' THEN 1 ELSE 0 END)
                       AS strategy_errors,
                   SUM(CASE WHEN error_stage = 'transport' THEN 1 ELSE 0 END)
                       AS transport_errors,
                   AVG(CASE WHEN responded_at IS NOT NULL
                            THEN (julianday(responded_at) - julianday(observed_at)) * 86400.0
                       END) AS mean_response_seconds,
                   MAX(CASE WHEN responded_at IS NOT NULL
                            THEN (julianday(responded_at) - julianday(observed_at)) * 86400.0
                       END) AS max_response_seconds
            FROM moves
            """
        ).fetchone()
        run_game_rows = connection.execute(
            """
            SELECT run_id, family,
                   COUNT(*) AS games,
                   SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_games,
                   COALESCE(outcome, status) AS outcome,
                   COUNT(*) AS outcome_games
            FROM games
            GROUP BY run_id, family, COALESCE(outcome, status)
            ORDER BY run_id, family, outcome
            """
        ).fetchall()
        run_move_rows = connection.execute(
            """
            SELECT run_id,
                   COUNT(*) AS moves,
                   SUM(CASE WHEN response_json IS NOT NULL
                                 AND json_extract(response_json, '$.valid') = 0
                            THEN 1 ELSE 0 END) AS invalid_moves,
                   SUM(CASE WHEN error_stage = 'strategy' THEN 1 ELSE 0 END)
                       AS strategy_errors,
                   SUM(CASE WHEN error_stage = 'transport' THEN 1 ELSE 0 END)
                       AS transport_errors,
                   AVG(CASE WHEN responded_at IS NOT NULL
                            THEN (julianday(responded_at) - julianday(observed_at)) * 86400.0
                       END) AS mean_response_seconds,
                   MAX(CASE WHEN responded_at IS NOT NULL
                            THEN (julianday(responded_at) - julianday(observed_at)) * 86400.0
                       END) AS max_response_seconds
            FROM moves
            GROUP BY run_id
            ORDER BY run_id
            """
        ).fetchall()
        policy_rows = connection.execute(
            """
            SELECT game_id, policy_version
            FROM moves
            GROUP BY game_id, policy_version
            ORDER BY game_id, policy_version
            """
        ).fetchall()

    policies_by_game: dict[str, list[str]] = defaultdict(list)
    for row in policy_rows:
        policies_by_game[str(row["game_id"])].append(str(row["policy_version"]))

    run_families: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in run_game_rows:
        run_id = str(row["run_id"])
        family = str(row["family"])
        bucket = run_families[run_id].setdefault(
            family,
            {"games": 0, "active_games": 0, "outcomes": {}},
        )
        bucket["games"] += int(row["outcome_games"])
        bucket["active_games"] += int(row["active_games"] or 0)
        bucket["outcomes"][str(row["outcome"])] = int(row["outcome_games"])
    run_moves = {str(row["run_id"]): dict(row) for row in run_move_rows}

    grouped: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "games": 0,
            "completed": 0,
            "unresolved": 0,
            "outcomes": defaultdict(int),
            "payoffs": [],
            "normalized_payoffs": [],
            "mixed_policy_games": 0,
        }
    )
    for game in games:
        configuration = _load(game["configuration_json"]) or {}
        key = (
            game["policy_version"],
            game["family"],
            game["role"],
            configuration.get("complete_information"),
            configuration.get("horizon_known"),
            game["opponent_type"],
        )
        bucket = grouped[key]
        bucket["games"] += 1
        if len(policies_by_game.get(str(game["game_id"]), [])) > 1:
            bucket["mixed_policy_games"] += 1
        if game["status"] == "active":
            bucket["unresolved"] += 1
        else:
            bucket["completed"] += 1
        bucket["outcomes"][game["outcome"] or game["status"]] += 1
        if game["own_payoff"] is not None:
            bucket["payoffs"].append(float(game["own_payoff"]))
        normalized = _normal_payoff(game, configuration)
        if normalized is not None:
            bucket["normalized_payoffs"].append(normalized)

    slices = []
    for key, bucket in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        version, family, role, complete, known_horizon, opponent_type = key
        slices.append(
            {
                "policy_version": version,
                "family": family,
                "role": role,
                "complete_information": complete,
                "horizon_known": known_horizon,
                "opponent_type": opponent_type,
                "games": bucket["games"],
                "completed": bucket["completed"],
                "unresolved": bucket["unresolved"],
                "mixed_policy_games": bucket["mixed_policy_games"],
                "outcomes": dict(sorted(bucket["outcomes"].items())),
                "mean_payoff": _mean_or_none(bucket["payoffs"]),
                "mean_normalized_payoff": _mean_or_none(bucket["normalized_payoffs"]),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": str(database.resolve()),
        "run_count": len(runs),
        "game_count": len(games),
        "mixed_policy_game_count": sum(
            len(policies_by_game.get(str(game["game_id"]), [])) > 1 for game in games
        ),
        "move_health": dict(move_health) if move_health is not None else {},
        "runs": [
            {
                "run_id": row["run_id"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "policy_version": row["policy_version"],
                "exit_reason": row["exit_reason"],
                "game_count": sum(
                    family["games"] for family in run_families.get(str(row["run_id"]), {}).values()
                ),
                "active_game_count": sum(
                    family["active_games"]
                    for family in run_families.get(str(row["run_id"]), {}).values()
                ),
                "family_outcomes": run_families.get(str(row["run_id"]), {}),
                "move_health": run_moves.get(str(row["run_id"]), {}),
                "rating_changes": _rating_changes(row),
            }
            for row in runs
        ],
        "slices": slices,
    }


def _mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


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
        print(f"Wrote telemetry analysis to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
