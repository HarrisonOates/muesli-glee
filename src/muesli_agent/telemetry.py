"""Durable, local telemetry for live GLEE games.

The public SDK intentionally keeps no game archive.  This module records every
state shown to MUESLI, the action it chose, the API response, and the terminal
state when it can be reconciled.  It never records the API key.

SQLite is used instead of process memory so an interrupted run can resume and
finish reconciling games that ended while the opponent had the move.  All writes
are short, serialized transactions; a telemetry failure is logged but never
changes the strategic action submitted to GLEE.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import requests
from glee_sdk import CompetitionClosedError, GleeAPIError, GleeClient

logger = logging.getLogger("muesli_agent.telemetry")

TERMINAL_STATUSES = {"completed", "no_deal", "cancelled", "timed_out", "abandoned"}
SOURCE_FILES = ("strategy.py", "bargaining.py", "negotiation.py", "persuasion.py")


class ConcurrentRunError(RuntimeError):
    """Raised when another live process already owns the agent run lock."""


class AgentRunLock:
    """Prevent two local runners from servicing the same agent simultaneously."""

    def __init__(self, agent_identity: str, *, lock_directory: str | Path | None = None):
        identity_hash = hashlib.sha256(agent_identity.encode()).hexdigest()[:20]
        directory = (
            Path(lock_directory).expanduser().resolve()
            if lock_directory is not None
            else Path(tempfile.gettempdir())
        )
        user_id = os.getuid() if hasattr(os, "getuid") else os.getpid()
        self.path = directory / f"muesli-glee-{user_id}-{identity_hash}.lock"
        self._file: Any = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(descriptor, 0o600)
        lock_file = os.fdopen(descriptor, "r+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise ConcurrentRunError(
                f"Another MUESLI live run is already active (lock: {self.path})."
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} started_at={_utc_now()}\n")
        lock_file.flush()
        self._file = lock_file
        return self

    def __exit__(self, *_: object) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def default_database_path() -> Path:
    """Return the ignored, project-local telemetry database path."""

    configured = os.environ.get("GLEE_GAME_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / ".local" / "glee-games.sqlite3"


def policy_fingerprint(source_files: Sequence[str] = SOURCE_FILES) -> str:
    """Identify the exact family-policy source used for a run."""

    configured = os.environ.get("MUESLI_POLICY_VERSION")
    if configured:
        return configured
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for filename in source_files:
        path = package_dir / filename
        digest.update(filename.encode())
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _visible_configuration(game: Mapping[str, Any]) -> dict[str, Any]:
    """Extract stable, visible fields useful for role/configuration analysis."""

    state = _mapping(game.get("game_state"))
    family = game.get("game_family")
    common = (
        "horizon_known",
        "max_rounds",
        "complete_information",
        "messages_allowed",
    )
    by_family = {
        "bargaining": ("money_to_divide", "delta_1", "delta_2"),
        "negotiation": (
            "player_1_role",
            "player_2_role",
            "player_1_value",
            "player_2_value",
        ),
        "persuasion": (
            "p",
            "v",
            "u",
            "product_price",
            "total_rounds",
            "seller_message_type",
            "is_seller_know_cv",
        ),
    }
    keys = common + by_family.get(str(family), ())
    return {key: state[key] for key in keys if key in state}


def _role(game: Mapping[str, Any]) -> str | None:
    state = _mapping(game.get("game_state"))
    player = game.get("your_player")
    explicit = state.get(f"{player}_role")
    if explicit is not None:
        return str(explicit)
    if game.get("game_family") == "persuasion":
        return "seller" if player == "player_1" else "buyer"
    return str(player) if player is not None else None


class GameRecorder:
    """Thread-safe SQLite store for runs, games, and move-level snapshots."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            # All in-process writes already use a lock and the agent-wide run lock
            # prevents a second runner.  A short external-lock timeout protects the
            # next game poll if an analyst happens to hold the archive open.
            self._connection.execute("PRAGMA busy_timeout=250")
            self._create_schema()
        self._restrict_database_files()

    def _restrict_database_files(self) -> None:
        """Keep the database and SQLite sidecars private on shared machines."""

        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
            Path(f"{self.path}-journal"),
        ):
            if path.exists():
                path.chmod(0o600)

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO metadata(key, value) VALUES ('schema_version', '1');

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                policy_version TEXT NOT NULL,
                agent_id TEXT,
                agent_name TEXT,
                options_json TEXT NOT NULL,
                initial_stats_json TEXT,
                final_stats_json TEXT,
                exit_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                family TEXT NOT NULL,
                your_player TEXT,
                role TEXT,
                opponent_type TEXT,
                opponent_name TEXT,
                configuration_key TEXT NOT NULL,
                configuration_json TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_reconciled_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                outcome TEXT,
                own_payoff REAL,
                opponent_payoff REAL,
                result_json TEXT,
                terminal_state_json TEXT,
                last_observation_json TEXT NOT NULL,
                reconcile_error TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS moves (
                move_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                responded_at TEXT,
                phase TEXT,
                round_number INTEGER,
                observation_json TEXT NOT NULL,
                action_json TEXT,
                response_json TEXT,
                error_stage TEXT,
                error_text TEXT,
                FOREIGN KEY(game_id) REFERENCES games(game_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS games_status_idx
                ON games(status, last_reconciled_at);
            CREATE INDEX IF NOT EXISTS games_analysis_idx
                ON games(family, role, configuration_key, status);
            CREATE INDEX IF NOT EXISTS moves_game_idx ON moves(game_id, move_id);
            """
        )
        self._connection.commit()

    def start_run(
        self,
        *,
        policy_version: str,
        options: Mapping[str, Any],
        initial_stats: Mapping[str, Any] | None,
    ) -> str:
        run_id = str(uuid.uuid4())
        stats = _mapping(initial_stats)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO runs(
                    run_id, started_at, policy_version, agent_id, agent_name,
                    options_json, initial_stats_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _utc_now(),
                    policy_version,
                    stats.get("agent_id"),
                    stats.get("agent_name"),
                    _json(dict(options)),
                    _json(dict(stats)) if stats else None,
                ),
            )
            self._connection.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        final_stats: Mapping[str, Any] | None,
        exit_reason: str,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE runs
                SET ended_at = ?, final_stats_json = ?, exit_reason = ?
                WHERE run_id = ?
                """,
                (
                    _utc_now(),
                    _json(dict(final_stats)) if final_stats else None,
                    exit_reason,
                    run_id,
                ),
            )
            self._connection.commit()

    def _upsert_game_locked(
        self,
        game: Mapping[str, Any],
        *,
        run_id: str,
        policy_version: str,
    ) -> None:
        game_id = str(game["game_id"])
        family = str(game.get("game_family", "unknown"))
        player = game.get("your_player")
        opponent = _mapping(game.get("opponent"))
        configuration = _visible_configuration(game)
        configuration_json = _json(configuration)
        configuration_key = hashlib.sha256(configuration_json.encode()).hexdigest()[:16]
        now = _utc_now()
        observation_json = _json(dict(game))
        self._connection.execute(
            """
            INSERT INTO games(
                game_id, run_id, policy_version, family, your_player, role,
                opponent_type, opponent_name, configuration_key,
                configuration_json, first_seen_at, last_seen_at,
                last_observation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                run_id = excluded.run_id,
                policy_version = excluded.policy_version,
                last_seen_at = excluded.last_seen_at,
                last_observation_json = excluded.last_observation_json,
                role = COALESCE(excluded.role, games.role),
                opponent_type = COALESCE(excluded.opponent_type, games.opponent_type),
                opponent_name = COALESCE(excluded.opponent_name, games.opponent_name)
            """,
            (
                game_id,
                run_id,
                policy_version,
                family,
                player,
                _role(game),
                opponent.get("type"),
                opponent.get("name"),
                configuration_key,
                configuration_json,
                now,
                now,
                observation_json,
            ),
        )

    def record_move(
        self,
        game: Mapping[str, Any],
        action: Mapping[str, Any] | None,
        *,
        run_id: str,
        policy_version: str,
        error_stage: str | None = None,
        error_text: str | None = None,
    ) -> int:
        state = _mapping(game.get("game_state"))
        with self._lock:
            self._upsert_game_locked(game, run_id=run_id, policy_version=policy_version)
            cursor = self._connection.execute(
                """
                INSERT INTO moves(
                    game_id, run_id, policy_version, observed_at, phase,
                    round_number, observation_json, action_json,
                    error_stage, error_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(game["game_id"]),
                    run_id,
                    policy_version,
                    _utc_now(),
                    game.get("phase", state.get("phase")),
                    _integer_or_none(state.get("round")),
                    _json(dict(game)),
                    _json(dict(action)) if action is not None else None,
                    error_stage,
                    error_text,
                ),
            )
            self._connection.commit()
            return int(cursor.lastrowid)

    def record_response(self, move_id: int, response: Mapping[str, Any]) -> None:
        response_dict = dict(response)
        with self._lock:
            self._connection.execute(
                """
                UPDATE moves SET responded_at = ?, response_json = ? WHERE move_id = ?
                """,
                (_utc_now(), _json(response_dict), move_id),
            )
            if response_dict.get("game_over"):
                row = self._connection.execute(
                    "SELECT game_id FROM moves WHERE move_id = ?", (move_id,)
                ).fetchone()
                if row is not None:
                    self._mark_terminal_locked(
                        str(row["game_id"]),
                        response_dict.get("result"),
                        terminal_state=None,
                        status=None,
                    )
            self._connection.commit()

    def record_exchange(
        self,
        game: Mapping[str, Any],
        action: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        run_id: str,
        policy_version: str,
        observed_at: str | None = None,
    ) -> int:
        """Persist an observation, action, and API response in one transaction."""

        state = _mapping(game.get("game_state"))
        response_dict = dict(response)
        now = _utc_now()
        with self._lock:
            self._upsert_game_locked(game, run_id=run_id, policy_version=policy_version)
            cursor = self._connection.execute(
                """
                INSERT INTO moves(
                    game_id, run_id, policy_version, observed_at, responded_at,
                    phase, round_number, observation_json, action_json, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(game["game_id"]),
                    run_id,
                    policy_version,
                    observed_at or now,
                    now,
                    game.get("phase", state.get("phase")),
                    _integer_or_none(state.get("round")),
                    _json(dict(game)),
                    _json(dict(action)),
                    _json(response_dict),
                ),
            )
            if response_dict.get("game_over"):
                self._mark_terminal_locked(
                    str(game["game_id"]),
                    response_dict.get("result"),
                    terminal_state=None,
                    status=None,
                )
            self._connection.commit()
            return int(cursor.lastrowid)

    def record_move_error(self, move_id: int, stage: str, error: BaseException) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE moves
                SET responded_at = ?, error_stage = ?, error_text = ?
                WHERE move_id = ?
                """,
                (_utc_now(), stage, f"{type(error).__name__}: {error}", move_id),
            )
            self._connection.commit()

    def record_reconciliation(self, game_id: str, payload: Mapping[str, Any]) -> bool:
        payload_dict = dict(payload)
        state = _mapping(payload_dict.get("game_state"))
        status = str(payload_dict.get("status", state.get("status", "active"))).lower()
        result = payload_dict.get("result", state.get("result"))
        terminal = status in TERMINAL_STATUSES or result is not None
        with self._lock:
            if terminal:
                self._mark_terminal_locked(
                    game_id,
                    result,
                    terminal_state=payload_dict,
                    status=status,
                )
            else:
                self._connection.execute(
                    """
                    UPDATE games
                    SET last_reconciled_at = ?, reconcile_error = NULL
                    WHERE game_id = ?
                    """,
                    (_utc_now(), game_id),
                )
            self._connection.commit()
        return terminal

    def _mark_terminal_locked(
        self,
        game_id: str,
        result: Any,
        *,
        terminal_state: Mapping[str, Any] | None,
        status: str | None,
    ) -> None:
        result_map = _mapping(result)
        row = self._connection.execute(
            "SELECT your_player FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        player = str(row["your_player"]) if row is not None else ""
        other = "player_2" if player == "player_1" else "player_1"
        own_payoff = _number_or_none(result_map.get(f"{player}_payoff"))
        opponent_payoff = _number_or_none(result_map.get(f"{other}_payoff"))
        outcome = result_map.get("outcome")
        resolved_status = status if status in TERMINAL_STATUSES else "completed"
        if str(outcome).lower() == "no_deal":
            resolved_status = "no_deal"
        self._connection.execute(
            """
            UPDATE games
            SET status = ?, outcome = ?, own_payoff = ?,
                opponent_payoff = ?, result_json = ?, terminal_state_json = ?,
                last_reconciled_at = ?, reconcile_error = NULL
            WHERE game_id = ?
            """,
            (
                resolved_status,
                outcome,
                own_payoff,
                opponent_payoff,
                _json(dict(result_map)) if result_map else None,
                _json(dict(terminal_state)) if terminal_state is not None else None,
                _utc_now(),
                game_id,
            ),
        )

    def _candidate_for_reconciliation(self, minimum_age_seconds: float) -> str | None:
        cutoff = datetime.fromtimestamp(
            time.time() - minimum_age_seconds, tz=timezone.utc
        ).isoformat(timespec="milliseconds")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT game_id FROM games
                WHERE status = 'active'
                  AND (last_reconciled_at IS NULL OR last_reconciled_at < ?)
                ORDER BY COALESCE(last_reconciled_at, first_seen_at), first_seen_at
                LIMIT 1
                """,
                (cutoff,),
            ).fetchone()
            if row is None:
                return None
            game_id = str(row["game_id"])
            self._connection.execute(
                "UPDATE games SET last_reconciled_at = ? WHERE game_id = ?",
                (_utc_now(), game_id),
            )
            self._connection.commit()
            return game_id

    def reconcile_one(self, client: GleeClient, minimum_age_seconds: float = 15.0) -> bool:
        game_id = self._candidate_for_reconciliation(minimum_age_seconds)
        if game_id is None:
            return False
        try:
            payload = client.game_state(game_id)
            return self.record_reconciliation(game_id, _mapping(payload))
        except Exception as error:  # noqa: BLE001 - reconciliation must not stop play
            with self._lock:
                self._connection.execute(
                    "UPDATE games SET reconcile_error = ? WHERE game_id = ?",
                    (f"{type(error).__name__}: {error}", game_id),
                )
                self._connection.commit()
            return False

    def reconcile_all(self, client: GleeClient) -> tuple[int, int]:
        with self._lock:
            game_ids = [
                str(row["game_id"])
                for row in self._connection.execute(
                    "SELECT game_id FROM games WHERE status = 'active' ORDER BY first_seen_at"
                )
            ]
        completed = 0
        for game_id in game_ids:
            try:
                payload = client.game_state(game_id)
                completed += self.record_reconciliation(game_id, _mapping(payload))
            except Exception as error:  # noqa: BLE001 - preserve every other game
                with self._lock:
                    self._connection.execute(
                        "UPDATE games SET reconcile_error = ? WHERE game_id = ?",
                        (f"{type(error).__name__}: {error}", game_id),
                    )
                    self._connection.commit()
        return completed, len(game_ids) - completed

    def summary(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT family, role,
                       COUNT(*) AS games,
                       SUM(status <> 'active') AS completed,
                       SUM(status = 'active') AS unresolved,
                       AVG(own_payoff) AS mean_payoff,
                       SUM(CASE WHEN outcome = 'agreement' THEN 1 ELSE 0 END) AS agreements
                FROM games
                GROUP BY family, role
                ORDER BY family, role
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def export_jsonl(self, destination: str | Path) -> int:
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        descriptor = os.open(destination_path, open_flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with self._lock, os.fdopen(descriptor, "w", encoding="utf-8") as output:
            games = self._connection.execute(
                "SELECT * FROM games ORDER BY first_seen_at, game_id"
            ).fetchall()
            for game in games:
                record = dict(game)
                for key in (
                    "configuration_json",
                    "result_json",
                    "terminal_state_json",
                    "last_observation_json",
                ):
                    record[key.removesuffix("_json")] = _loads_or_none(record.pop(key))
                moves = self._connection.execute(
                    "SELECT * FROM moves WHERE game_id = ? ORDER BY move_id",
                    (record["game_id"],),
                ).fetchall()
                record["moves"] = []
                for move in moves:
                    move_record = dict(move)
                    for key in ("observation_json", "action_json", "response_json"):
                        move_record[key.removesuffix("_json")] = _loads_or_none(
                            move_record.pop(key)
                        )
                    record["moves"].append(move_record)
                output.write(_json(record) + "\n")
                count += 1
        destination_path.chmod(0o600)
        return count

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RecordingGleeClient(GleeClient):
    """GLEE client that preserves move records and can safely drain on exit."""

    def __init__(
        self,
        *args: Any,
        recorder: GameRecorder,
        run_id: str,
        policy_version: str,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self.run_id = run_id
        self.policy_version = policy_version
        self._reconcile_lock = threading.Lock()
        self._last_reconcile = -math.inf
        self._pending_lock = threading.Lock()
        self._pending_exchanges: list[tuple[dict, dict, dict, str]] = []

    def _record_quietly(self, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - telemetry is deliberately fail-open
            logger.warning("Telemetry write failed without affecting play: %s", error)
            return None

    def _persist_exchange(self, game: dict, action: dict, response: dict, observed_at: str) -> None:
        move_id = self._record_quietly(
            lambda: self.recorder.record_exchange(
                game,
                action,
                response,
                run_id=self.run_id,
                policy_version=self.policy_version,
                observed_at=observed_at,
            )
        )
        if not isinstance(move_id, int):
            with self._pending_lock:
                self._pending_exchanges.append(
                    (dict(game), dict(action), dict(response), observed_at)
                )
            return
        self.flush_pending_telemetry(limit=1)

    def flush_pending_telemetry(self, *, limit: int | None = None) -> int:
        """Retry exchanges kept in memory after a transient storage failure."""

        recovered = 0
        with self._pending_lock:
            while self._pending_exchanges and (limit is None or recovered < limit):
                game, action, response, observed_at = self._pending_exchanges[0]
                move_id = self._record_quietly(
                    lambda game=game, action=action, response=response, observed_at=observed_at: (
                        self.recorder.record_exchange(
                            game,
                            action,
                            response,
                            run_id=self.run_id,
                            policy_version=self.policy_version,
                            observed_at=observed_at,
                        )
                    )
                )
                if not isinstance(move_id, int):
                    break
                self._pending_exchanges.pop(0)
                recovered += 1
        return recovered

    def _handle_game(self, strategy: Callable[[dict], dict], game: dict) -> bool:
        observed_at = _utc_now()
        game_id = game["game_id"]
        family = game["game_family"]
        state = _mapping(game.get("game_state"))
        logger.info(
            "Game %s (%s) - round %s, phase: %s, your turn as %s",
            game_id,
            family,
            state.get("round", "?"),
            game.get("phase"),
            game.get("your_player"),
        )
        try:
            action = strategy(game)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            self._record_quietly(
                lambda error_text=error_text: self.recorder.record_move(
                    game,
                    None,
                    run_id=self.run_id,
                    policy_version=self.policy_version,
                    error_stage="strategy",
                    error_text=error_text,
                )
            )
            logger.exception("Strategy error for game %s", game_id)
            return False

        try:
            result = self.move(game_id, action)
        except Exception as error:
            move_id = self._record_quietly(
                lambda: self.recorder.record_move(
                    game,
                    action,
                    run_id=self.run_id,
                    policy_version=self.policy_version,
                )
            )
            if isinstance(move_id, int):
                captured_error = error
                self._record_quietly(
                    lambda captured_error=captured_error: self.recorder.record_move_error(
                        move_id, "transport", captured_error
                    )
                )
            raise

        # Persist only after the action has reached the API, so local storage can
        # never consume the current move's 120-second submission budget.
        self._persist_exchange(game, action, result, observed_at)
        if result.get("valid") is False:
            attempts = result.get("attempts_left")
            suffix = f" ({attempts} attempts left)" if attempts is not None else ""
            logger.warning(
                "Game %s: move rejected as invalid — %r%s",
                game_id,
                result.get("error"),
                suffix,
            )
        else:
            logger.info("Move result: %s", result)
        if result.get("game_over"):
            logger.info("Game %s finished! Result: %s", game_id, result.get("result"))
            return True
        return False

    def stats(self) -> dict:
        stats = super().stats()
        now = time.monotonic()
        if now - self._last_reconcile >= 60.0 and self._reconcile_lock.acquire(blocking=False):
            try:
                self._last_reconcile = now
                self._record_quietly(lambda: self.recorder.reconcile_one(self))
            finally:
                self._reconcile_lock.release()
        return stats

    def drain(
        self,
        strategy: Callable[[dict], dict],
        *,
        poll_interval: float,
        concurrency: int,
    ) -> None:
        """Leave all queues and finish games that the SDK returned before draining."""

        self._leave_queue_quietly()
        # GleeClient.run() consumes the first Ctrl-C before returning.  A Ctrl-C
        # here is therefore the explicit force-exit requested by the operator.
        interrupted_once = True
        last_status_check = -math.inf
        status_interval = max(15.0, poll_interval * 5.0)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while True:
                try:
                    games = self.pending_games()
                    futures = [pool.submit(self._handle_game, strategy, game) for game in games]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except (GleeAPIError, requests.RequestException) as error:
                            logger.warning("Game error while draining: %s", error)
                    now = time.monotonic()
                    if not games and now - last_status_check >= status_interval:
                        last_status_check = now
                        # Bypass the opportunistic reconciliation in stats(); final
                        # reconciliation runs once active games reach zero.  This keeps
                        # drain traffic below the platform's 60-request/minute limit.
                        stats = super().stats()
                        if int(stats.get("active_games", 0)) == 0:
                            return
                    time.sleep(poll_interval)
                except CompetitionClosedError:
                    return
                except KeyboardInterrupt:
                    if interrupted_once:
                        logger.warning("Second interrupt: leaving active games undrained.")
                        return
                    interrupted_once = True
                    logger.info("Still draining active games; press Ctrl-C again to force exit.")
                except (GleeAPIError, requests.RequestException) as error:
                    logger.warning("API error while draining: %s", error)
                    time.sleep(max(5.0, poll_interval))


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _loads_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def export_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export MUESLI's private GLEE telemetry")
    parser.add_argument("--database", type=Path, default=default_database_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    with GameRecorder(args.database) as recorder:
        count = recorder.export_jsonl(args.output)
        if args.summary:
            print(json.dumps(recorder.summary(), indent=2, default=str))
    print(f"Exported {count} games to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(export_main())
