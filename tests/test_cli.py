from __future__ import annotations

from muesli_agent.cli import _profile_strategy, build_parser
from muesli_agent.strategy import strategy
from muesli_agent.telemetry import SOURCE_FILES


def test_parser_defaults_to_granola_profile():
    args = build_parser().parse_args([])
    assert args.profile == "granola"


def test_granola_profile_preserves_existing_callback_and_fingerprint_sources():
    callback, source_files = _profile_strategy("granola")
    assert callback is strategy
    assert source_files == SOURCE_FILES


def test_challenger_profile_has_its_own_callback_and_source():
    from muesli_agent.challenger import challenger_strategy

    callback, source_files = _profile_strategy("challenger")
    assert callback is challenger_strategy
    assert source_files == (*SOURCE_FILES, "challenger.py")
