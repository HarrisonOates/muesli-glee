"""Command-line runner for MUESLI."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .strategy import strategy
from .telemetry import (
    SOURCE_FILES,
    AgentRunLock,
    ConcurrentRunError,
    GameRecorder,
    RecordingGleeClient,
    default_database_path,
    policy_fingerprint,
)

FAMILIES = ("bargaining", "negotiation", "persuasion")
PROFILES = ("granola", "buckwheat", "millet", "challenger")
StrategyCallback = Callable[[dict[str, Any]], dict[str, Any]]


def _profile_strategy(profile: str) -> tuple[StrategyCallback, tuple[str, ...]]:
    """Resolve a policy callback and the source files that identify it."""

    if profile == "granola":
        return strategy, SOURCE_FILES
    if profile == "challenger":
        from .challenger import challenger_strategy

        return challenger_strategy, (*SOURCE_FILES, "challenger.py")
    if profile == "buckwheat":
        from .challenger import buckwheat_strategy

        return buckwheat_strategy, (*SOURCE_FILES, "challenger.py")
    if profile == "millet":
        from .challenger import millet_strategy

        return millet_strategy, (*SOURCE_FILES, "challenger.py")
    raise ValueError(f"Unknown policy profile: {profile!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MUESLI GLEE competition agent")
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="granola",
        help="policy profile to run (default: %(default)s)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="game families to queue (default: all)",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--max-games",
        type=int,
        help=(
            "SDK completion target; games ending on the opponent's move may make the "
            "reconciled match count larger"
        ),
    )
    parser.add_argument("--max-time", type=float, help="maximum queueing time in seconds")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="verify credentials and print agent stats without queueing",
    )
    parser.add_argument(
        "--telemetry-db",
        type=Path,
        default=default_database_path(),
        help="private SQLite game archive (default: %(default)s)",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")

    api_key = os.environ.get("GLEE_API_KEY")
    if not api_key:
        raise SystemExit(
            "GLEE_API_KEY is not set. Create an agent at "
            "https://glee-competition.com/dashboard and export its one-time key."
        )

    from glee_sdk import GleeClient

    base_url = os.environ.get("GLEE_API_URL")
    initial_client = (
        GleeClient(api_key=api_key, base_url=base_url) if base_url else GleeClient(api_key)
    )
    initial_stats = initial_client.stats()
    print(f"Agent stats: {initial_stats}")
    if args.stats_only:
        return 0

    run_options = {
        "game_families": args.families,
        "concurrency": args.concurrency,
        "poll_interval": args.poll_interval,
    }
    if args.max_games is not None:
        run_options["max_games"] = args.max_games
    if args.max_time is not None:
        run_options["max_time"] = args.max_time

    strategy_callback, source_files = _profile_strategy(args.profile)
    version = policy_fingerprint(source_files)
    lock_identity = str(initial_stats.get("agent_id") or version)
    try:
        with AgentRunLock(lock_identity), GameRecorder(args.telemetry_db) as recorder:
            run_id = recorder.start_run(
                policy_version=version,
                options={**run_options, "profile": args.profile},
                initial_stats=initial_stats,
            )
            client_options = {
                "api_key": api_key,
                "recorder": recorder,
                "run_id": run_id,
                "policy_version": version,
            }
            if base_url:
                client_options["base_url"] = base_url
            client = RecordingGleeClient(**client_options)
            logging.getLogger(__name__).info(
                "Persisting private game telemetry to %s (policy %s)",
                args.telemetry_db,
                version,
            )

            exit_reason = "sdk-returned-and-drained"
            final_stats = None
            try:
                client.run(strategy_callback, **run_options)
                # The current SDK returns immediately on Ctrl-C.  This no-op on normal
                # bounded completion keeps servicing any games still active after an
                # interrupt, protecting both the rating and terminal telemetry.
                client.drain(
                    strategy_callback,
                    poll_interval=args.poll_interval,
                    concurrency=args.concurrency,
                )
            except BaseException as error:
                exit_reason = f"{type(error).__name__}: {error}"
                raise
            finally:
                try:
                    recovered = client.flush_pending_telemetry()
                    if recovered:
                        logging.getLogger(__name__).info(
                            "Recovered %s buffered telemetry exchanges", recovered
                        )
                    final_stats = client.stats()
                    if int(final_stats.get("active_games", 0)) == 0:
                        completed, unresolved = recorder.reconcile_all(client)
                        logging.getLogger(__name__).info(
                            "Terminal reconciliation: %s completed, %s unresolved",
                            completed,
                            unresolved,
                        )
                except Exception as error:  # noqa: BLE001 - preserve original outcome
                    logging.getLogger(__name__).warning("Could not finalize telemetry: %s", error)
                recorder.finish_run(
                    run_id,
                    final_stats=final_stats,
                    exit_reason=exit_reason,
                )
    except ConcurrentRunError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
