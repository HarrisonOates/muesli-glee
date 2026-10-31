from __future__ import annotations

import importlib.util
from pathlib import Path

from muesli_agent.telemetry import GameRecorder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_telemetry.py"
SPEC = importlib.util.spec_from_file_location("analyze_telemetry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_groups_role_configuration_and_rating_change(tmp_path):
    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        run_id = recorder.start_run(
            policy_version="policy-a",
            options={"game_families": ["negotiation"]},
            initial_stats={"scores": {"negotiation": {"rating": 1000.0, "games_played": 0}}},
        )
        game = {
            "game_id": "n-1",
            "game_family": "negotiation",
            "your_player": "player_1",
            "phase": "decision",
            "opponent": {"type": "hidden", "name": None},
            "game_state": {
                "phase": "decision",
                "round": 1,
                "horizon_known": True,
                "max_rounds": 3,
                "complete_information": True,
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 40,
                "player_2_value": 100,
                "last_offer": {"price": 70},
                "history": [],
            },
            "valid_actions": {"type": "decision", "fields": {}},
        }
        move_id = recorder.record_move(
            game,
            {"decision": "AcceptOffer"},
            run_id=run_id,
            policy_version="policy-a",
        )
        recorder.record_response(
            move_id,
            {
                "valid": True,
                "game_over": True,
                "result": {
                    "outcome": "agreement",
                    "player_1_payoff": 30,
                    "player_2_payoff": 30,
                },
            },
        )
        recorder.finish_run(
            run_id,
            final_stats={"scores": {"negotiation": {"rating": 1012.5, "games_played": 1}}},
            exit_reason="completed",
        )

    report = MODULE.build_report(database)
    assert report["game_count"] == 1
    assert report["mixed_policy_game_count"] == 0
    assert report["move_health"]["invalid_moves"] == 0
    assert report["move_health"]["mean_response_seconds"] is not None
    run = report["runs"][0]
    assert run["rating_changes"]["negotiation"]["rating_change"] == 12.5
    assert run["game_count"] == 1
    assert run["active_game_count"] == 0
    assert run["family_outcomes"]["negotiation"] == {
        "games": 1,
        "active_games": 0,
        "outcomes": {"agreement": 1},
    }
    assert run["move_health"]["moves"] == 1
    assert run["move_health"]["invalid_moves"] == 0
    assert report["slices"][0]["family"] == "negotiation"
    assert report["slices"][0]["role"] == "seller"
    assert report["slices"][0]["mixed_policy_games"] == 0
    assert report["slices"][0]["mean_normalized_payoff"] == 0.5


def test_active_run_does_not_invent_zero_final_stats(tmp_path):
    database = tmp_path / "active.sqlite3"
    with GameRecorder(database) as recorder:
        recorder.start_run(
            policy_version="policy-live",
            options={"game_families": ["bargaining"]},
            initial_stats={"scores": {"bargaining": {"rating": 1040.0, "games_played": 12}}},
        )

    report = MODULE.build_report(database)
    run = report["runs"][0]
    assert run["ended_at"] is None
    assert run["game_count"] == 0
    assert run["active_game_count"] == 0
    assert run["rating_changes"]["bargaining"] == {
        "rating_before": 1040.0,
        "rating_after": None,
        "rating_change": None,
        "games_before": 12,
        "games_after": None,
        "games_added": None,
    }
