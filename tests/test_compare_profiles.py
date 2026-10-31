from __future__ import annotations

import importlib.util
from pathlib import Path

from muesli_agent.telemetry import GameRecorder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_profiles.py"
SPEC = importlib.util.spec_from_file_location("compare_profiles", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_counts_discount_latch_divergence(tmp_path):
    database = tmp_path / "games.sqlite3"
    observation = {
        "game_id": "repeated-offer",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "decision",
        "opponent": {"type": "hidden", "name": None},
        "game_state": {
            "phase": "decision",
            "current_player": "player_1",
            "round": 3,
            "horizon_known": False,
            "complete_information": False,
            "money_to_divide": 100,
            "delta_1": 0.8,
            "last_offer": {
                "player_1_gain": 20,
                "player_2_gain": 80,
                "proposer": "player_2",
                "round": 3,
            },
            "history": [
                {
                    "round": 1,
                    "proposer": "player_2",
                    "offer": {"player_1_gain": 20, "player_2_gain": 80},
                    "decision": "reject",
                }
            ],
        },
        "valid_actions": {"type": "decision", "fields": {}},
    }
    with GameRecorder(database) as recorder:
        run_id = recorder.start_run(
            policy_version="granola",
            options={"game_families": ["bargaining"]},
            initial_stats={},
        )
        recorder.record_move(
            observation,
            {"decision": "reject"},
            run_id=run_id,
            policy_version="granola",
        )

    report = MODULE.build_report(database)
    assert report["moves"] == 1
    assert report["replay_errors"] == 0
    assert report["families"]["bargaining"] == {"moves": 1, "divergences": 1}
    assert report["action_transitions"]["bargaining"] == {"reject->accept": 1}
