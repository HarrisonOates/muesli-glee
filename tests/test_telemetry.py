from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from muesli_agent.telemetry import (
    SOURCE_FILES,
    AgentRunLock,
    ConcurrentRunError,
    GameRecorder,
    RecordingGleeClient,
    policy_fingerprint,
)


def game(game_id: str = "game-1", *, phase: str = "offer") -> dict:
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": phase,
        "opponent": {"type": "agent", "name": "Opponent"},
        "game_state": {
            "phase": phase,
            "round": 1,
            "max_rounds": 5,
            "horizon_known": True,
            "complete_information": True,
            "messages_allowed": True,
            "money_to_divide": 1000,
            "delta_1": 0.9,
            "delta_2": 0.8,
            "history": [],
        },
        "valid_actions": {"type": phase, "fields": {}},
    }


def start(recorder: GameRecorder) -> tuple[str, str]:
    version = "test-policy"
    run_id = recorder.start_run(
        policy_version=version,
        options={"game_families": ["bargaining"]},
        initial_stats={"agent_id": "agent-1", "agent_name": "Granola"},
    )
    return run_id, version


def query(database: Path, sql: str):
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(sql).fetchall()


def test_records_complete_move_result_and_analysis_fields(tmp_path):
    database = tmp_path / "private" / "games.sqlite3"
    with GameRecorder(database) as recorder:
        sidecars = [Path(f"{database}{suffix}") for suffix in ("-wal", "-shm")]
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in sidecars if path.exists())
        run_id, version = start(recorder)
        move_id = recorder.record_move(
            game(),
            {"alice_gain": 600, "bob_gain": 400},
            run_id=run_id,
            policy_version=version,
        )
        recorder.record_response(
            move_id,
            {
                "valid": True,
                "game_over": True,
                "result": {
                    "outcome": "agreement",
                    "player_1_payoff": 600,
                    "player_2_payoff": 400,
                },
            },
        )
        recorder.finish_run(run_id, final_stats={"active_games": 0}, exit_reason="completed")
        summary = recorder.summary()

    stored_game = dict(query(database, "SELECT * FROM games")[0])
    stored_move = dict(query(database, "SELECT * FROM moves")[0])
    assert stored_game["status"] == "completed"
    assert stored_game["role"] == "player_1"
    assert stored_game["opponent_type"] == "agent"
    assert stored_game["own_payoff"] == 600
    assert stored_game["opponent_payoff"] == 400
    assert json.loads(stored_move["action_json"])["alice_gain"] == 600
    assert json.loads(stored_move["observation_json"])["game_id"] == "game-1"
    assert summary == [
        {
            "family": "bargaining",
            "role": "player_1",
            "games": 1,
            "completed": 1,
            "unresolved": 0,
            "mean_payoff": 600.0,
            "agreements": 1,
        }
    ]
    assert database.stat().st_mode & 0o777 == 0o600


def test_reconciles_game_that_ended_on_opponents_move(tmp_path):
    class FinishedClient:
        def game_state(self, game_id):
            assert game_id == "opponent-ended"
            return {
                "status": "completed",
                "game_state": {"history": [{"round": 1}]},
                "result": {
                    "outcome": "agreement",
                    "player_1_payoff": 450,
                    "player_2_payoff": 550,
                },
            }

    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        run_id, version = start(recorder)
        recorder.record_move(
            game("opponent-ended"),
            {"alice_gain": 450, "bob_gain": 550},
            run_id=run_id,
            policy_version=version,
        )
        assert recorder.reconcile_all(FinishedClient()) == (1, 0)

    stored = dict(query(database, "SELECT * FROM games")[0])
    assert stored["status"] == "completed"
    assert stored["own_payoff"] == 450
    assert json.loads(stored["terminal_state_json"])["status"] == "completed"


def test_recording_client_preserves_action_and_response(tmp_path, monkeypatch):
    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        run_id, version = start(recorder)
        client = RecordingGleeClient(
            "not-a-real-key",
            recorder=recorder,
            run_id=run_id,
            policy_version=version,
        )
        submitted = []

        def move(game_id, action):
            submitted.append((game_id, action))
            return {"valid": True, "game_over": False, "result": None}

        monkeypatch.setattr(client, "move", move)
        completed = client._handle_game(lambda _: {"alice_gain": 550, "bob_gain": 450}, game())

    assert completed is False
    assert submitted == [("game-1", {"alice_gain": 550, "bob_gain": 450})]
    stored = dict(query(database, "SELECT * FROM moves")[0])
    assert json.loads(stored["action_json"])["bob_gain"] == 450
    assert json.loads(stored["response_json"])["valid"] is True


def test_live_submission_precedes_any_telemetry_write(tmp_path, monkeypatch):
    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        run_id, version = start(recorder)
        client = RecordingGleeClient(
            "not-a-real-key",
            recorder=recorder,
            run_id=run_id,
            policy_version=version,
        )
        events = []
        original_record = recorder.record_exchange

        def record(*args, **kwargs):
            events.append("record")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(recorder, "record_exchange", record)
        monkeypatch.setattr(
            client,
            "move",
            lambda *_: (
                events.append("submit") or {"valid": True, "game_over": False, "result": None}
            ),
        )
        client._handle_game(lambda _: {"alice_gain": 500, "bob_gain": 500}, game())

    assert events == ["submit", "record"]


def test_telemetry_failure_never_blocks_a_legal_move(tmp_path, monkeypatch):
    with GameRecorder(tmp_path / "games.sqlite3") as recorder:
        run_id, version = start(recorder)
        client = RecordingGleeClient(
            "not-a-real-key",
            recorder=recorder,
            run_id=run_id,
            policy_version=version,
        )
        submitted = []
        monkeypatch.setattr(
            recorder,
            "record_exchange",
            lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")),
        )
        monkeypatch.setattr(
            client,
            "move",
            lambda game_id, action: (
                submitted.append((game_id, action)) or {"valid": True, "game_over": False}
            ),
        )
        client._handle_game(lambda _: {"alice_gain": 500, "bob_gain": 500}, game())

    assert submitted == [("game-1", {"alice_gain": 500, "bob_gain": 500})]


def test_strategy_exception_is_persisted_without_submitting(tmp_path, monkeypatch):
    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        run_id, version = start(recorder)
        client = RecordingGleeClient(
            "not-a-real-key",
            recorder=recorder,
            run_id=run_id,
            policy_version=version,
        )
        monkeypatch.setattr(
            client,
            "move",
            lambda *_: pytest.fail("an action should not be submitted after a strategy error"),
        )

        def broken_strategy(_):
            raise RuntimeError("broken policy")

        assert client._handle_game(broken_strategy, game()) is False

    stored = dict(query(database, "SELECT * FROM moves")[0])
    assert stored["action_json"] is None
    assert stored["error_stage"] == "strategy"
    assert "broken policy" in stored["error_text"]


def test_drain_services_pending_games_until_active_count_is_zero(tmp_path, monkeypatch):
    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        run_id, version = start(recorder)
        client = RecordingGleeClient(
            "not-a-real-key",
            recorder=recorder,
            run_id=run_id,
            policy_version=version,
        )
        pending = iter(([game()], []))
        submitted = []
        monkeypatch.setattr("glee_sdk.GleeClient.stats", lambda _: {"active_games": 0})
        monkeypatch.setattr(client, "leave_queue", lambda family=None: {})
        monkeypatch.setattr(client, "pending_games", lambda: next(pending))
        monkeypatch.setattr(
            client,
            "move",
            lambda game_id, action: (
                submitted.append((game_id, action))
                or {"valid": True, "game_over": True, "result": {"outcome": "agreement"}}
            ),
        )

        client.drain(
            lambda _: {"alice_gain": 500, "bob_gain": 500},
            poll_interval=0,
            concurrency=1,
        )

    assert submitted == [("game-1", {"alice_gain": 500, "bob_gain": 500})]
    assert dict(query(database, "SELECT * FROM games")[0])["status"] == "completed"


def test_exports_self_contained_jsonl(tmp_path):
    database = tmp_path / "games.sqlite3"
    export = tmp_path / "export" / "games.jsonl"
    with GameRecorder(database) as recorder:
        run_id, version = start(recorder)
        recorder.record_move(
            game(),
            {"alice_gain": 500, "bob_gain": 500},
            run_id=run_id,
            policy_version=version,
        )
        assert recorder.export_jsonl(export) == 1

    record = json.loads(export.read_text())
    assert record["game_id"] == "game-1"
    assert record["configuration"]["money_to_divide"] == 1000
    assert record["moves"][0]["observation"]["game_family"] == "bargaining"
    assert record["moves"][0]["action"]["alice_gain"] == 500


def test_policy_fingerprint_is_stable(monkeypatch):
    monkeypatch.delenv("MUESLI_POLICY_VERSION", raising=False)
    assert policy_fingerprint() == policy_fingerprint()
    assert policy_fingerprint().startswith("sha256:")


def test_challenger_fingerprint_is_stable_and_distinct(monkeypatch):
    monkeypatch.delenv("MUESLI_POLICY_VERSION", raising=False)
    source_files = (*SOURCE_FILES, "challenger.py")
    challenger = policy_fingerprint(source_files)
    assert challenger == policy_fingerprint(source_files)
    assert challenger.startswith("sha256:")
    assert challenger != policy_fingerprint()


def test_policy_fingerprint_accepts_explicit_experiment_label(monkeypatch):
    monkeypatch.setenv("MUESLI_POLICY_VERSION", "negotiation-search-v2")
    assert policy_fingerprint() == "negotiation-search-v2"


def test_second_live_runner_is_rejected_without_destroying_lock(tmp_path):
    lock_directory = tmp_path / "private"
    with AgentRunLock("same-agent", lock_directory=lock_directory) as first_lock:
        assert first_lock.path.stat().st_mode & 0o777 == 0o600
        with (
            pytest.raises(ConcurrentRunError, match="already active"),
            AgentRunLock("same-agent", lock_directory=lock_directory),
        ):
            pytest.fail("a second runner must not acquire the live lock")

    with AgentRunLock("same-agent", lock_directory=lock_directory):
        pass


def test_failed_exchange_is_buffered_and_recovered(tmp_path, monkeypatch):
    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        run_id, version = start(recorder)
        client = RecordingGleeClient(
            "not-a-real-key",
            recorder=recorder,
            run_id=run_id,
            policy_version=version,
        )
        original_record = recorder.record_exchange
        attempts = 0

        def flaky_record(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sqlite3.OperationalError("temporary lock")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(recorder, "record_exchange", flaky_record)
        monkeypatch.setattr(
            client,
            "move",
            lambda *_: {
                "valid": True,
                "game_over": True,
                "result": {
                    "outcome": "agreement",
                    "player_1_payoff": 500,
                    "player_2_payoff": 500,
                },
            },
        )
        client._handle_game(lambda _: {"alice_gain": 500, "bob_gain": 500}, game())
        assert client.flush_pending_telemetry() == 1

    stored = dict(query(database, "SELECT * FROM games")[0])
    assert stored["status"] == "completed"
    assert stored["own_payoff"] == 500


def test_resumed_game_uses_latest_run_and_policy_for_outcome_attribution(tmp_path):
    database = tmp_path / "games.sqlite3"
    with GameRecorder(database) as recorder:
        first_run, _ = start(recorder)
        recorder.record_move(
            game("resumed"),
            {"alice_gain": 500, "bob_gain": 500},
            run_id=first_run,
            policy_version="old-policy",
        )
        second_run = recorder.start_run(
            policy_version="new-policy",
            options={},
            initial_stats={"agent_id": "agent-1"},
        )
        recorder.record_move(
            game("resumed"),
            {"alice_gain": 550, "bob_gain": 450},
            run_id=second_run,
            policy_version="new-policy",
        )

    stored = dict(query(database, "SELECT * FROM games")[0])
    assert stored["run_id"] == second_run
    assert stored["policy_version"] == "new-policy"
