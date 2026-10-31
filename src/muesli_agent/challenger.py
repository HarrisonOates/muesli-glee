"""Adversarial challenger policies layered over the deployed MUESLI agent.

The challenger deliberately changes only a few evidence-backed cells.  Every
other payload is passed to the deployed family policy verbatim.  Like the base
agent, it reconstructs all state from the payload: there is no process-global
game memory, so retries and concurrent games remain independent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .bargaining import bargaining_strategy
from .negotiation import negotiation_strategy
from .persuasion import _classify_signal, persuasion_strategy

Number = int | float


@dataclass(frozen=True)
class ChallengerConfig:
    """Frozen constants for challenger ablations."""

    negotiation_initial_share: float = 0.55
    negotiation_acceptance_floor: float = 0.45
    persuasion_required_negative_exposures: int = 2
    persuasion_high_template: str = (
        "This is a high-grade product. I recommend buying it at the posted price."
    )
    persuasion_pooled_low_template: str = (
        "This one is serviceable — nothing fancy, but it does the job. Worth it."
    )
    persuasion_pass_template: str = "I recommend passing on this product."


DEFAULT_CHALLENGER_CONFIG = ChallengerConfig()

_CANONICAL_POSITIVE = "I recommend buying this product; it is high quality."
_CANONICAL_NEGATIVE = "I recommend passing on this product."


def challenger_strategy(
    game: Mapping[str, Any],
    config: ChallengerConfig = DEFAULT_CHALLENGER_CONFIG,
) -> dict[str, Any]:
    """Compatibility alias for the Millet challenger profile."""

    return millet_strategy(game, config)


def buckwheat_strategy(
    game: Mapping[str, Any],
    config: ChallengerConfig = DEFAULT_CHALLENGER_CONFIG,
) -> dict[str, Any]:
    """Run the frozen Buckwheat challenger policy."""

    family = game.get("game_family")
    if family == "bargaining":
        return _discount_latch(game)
    if family == "negotiation":
        return _nash_floor_bracket(game, config)
    if family == "persuasion":
        return _template_bayes_exact_guard(game, config)
    raise ValueError(f"Unknown game family: {family!r}")


def millet_strategy(
    game: Mapping[str, Any],
    config: ChallengerConfig = DEFAULT_CHALLENGER_CONFIG,
) -> dict[str, Any]:
    """Compatibility profile for the experiment now promoted to Granola."""

    return buckwheat_strategy(game, config)


def _discount_latch(game: Mapping[str, Any]) -> dict[str, Any]:
    """Accept a positive, repeated opponent offer when waiting strictly costs.

    GLEE bargaining histories normally store an offer under ``record['offer']``
    with the proposer on either the record or offer.  The parser also accepts a
    direct offer record, player gain aliases, and missing history/proposer fields.
    Ambiguous records are ignored rather than changing the base decision.
    """

    base = bargaining_strategy(game)
    state = _state(game)
    if _action_type(game) != "decision" or _bargaining_final(state):
        return base

    me = _player(game, state)
    delta_raw = state.get("delta_1" if me == "player_1" else "delta_2")
    try:
        delta = float(delta_raw)
    except (TypeError, ValueError):
        return base
    if not math.isfinite(delta) or delta >= 1.0:
        return base

    current = state.get("last_offer")
    if not isinstance(current, Mapping):
        return base
    other = _other(me)
    if _bargaining_proposer(current) != other:
        return base
    current_signature = _bargaining_offer_signature(current)
    if current_signature is None:
        return base
    current_gain = current_signature[0 if me == "player_1" else 1]
    if current_gain is None or current_gain <= 0.0:
        return base

    current_round = _round_number(current.get("round", state.get("round")))
    history = _history(state)
    for position, record in enumerate(history, start=1):
        offer = record.get("offer")
        if not isinstance(offer, Mapping):
            offer = record if _looks_like_bargaining_offer(record) else None
        if offer is None:
            continue
        proposer = _bargaining_history_proposer(record, offer, me)
        if proposer != other:
            continue
        record_round = _round_number(record.get("round", offer.get("round")))
        # Some SDK payloads duplicate the current offer in history.  A matching
        # contextual round is the current observation, not earlier evidence.
        if current_round is not None and record_round is not None and record_round >= current_round:
            continue
        # With no round metadata, a completed decision marks a genuinely earlier
        # observation.  Otherwise identical trailing records are ambiguous.
        if (
            current_round is None
            and record_round is None
            and "decision" not in record
            and position == len(history)
            and _same_bargaining_offer(offer, current)
        ):
            continue
        if _bargaining_offer_signature(offer) == current_signature:
            return {"decision": "accept"}
    return base


def _nash_floor_bracket(game: Mapping[str, Any], config: ChallengerConfig) -> dict[str, Any]:
    """Use a 55/45 Nash-surplus bracket in uncapped complete information."""

    base = negotiation_strategy(game)
    state = _state(game)
    cell = _negotiation_cell(game, state)
    if cell is None:
        return base
    me, role, own_value, surplus = cell
    action_type = _action_type(game)

    latest_demand = _latest_rejected_own_share(state, me, role, own_value, surplus)
    incoming_share = _incoming_surplus_share(state, me, role, own_value, surplus)

    if action_type == "decision":
        if incoming_share is None:
            return base
        if incoming_share + 1e-12 >= config.negotiation_acceptance_floor:
            return {"decision": "AcceptOffer"}
        if latest_demand is None:
            target_share = config.negotiation_initial_share
        else:
            target_share = _bracket_share(latest_demand, incoming_share, config)
        if target_share is None:  # Inconsistent external history below the floor.
            return base
        return _negotiation_counter(state, role, own_value, surplus, target_share)

    if action_type != "offer":
        return base
    if latest_demand is None:
        target_share = config.negotiation_initial_share
    elif incoming_share is None:
        if latest_demand + 1e-12 < config.negotiation_acceptance_floor:
            return base
        target_share = min(latest_demand, config.negotiation_initial_share)
    else:
        target_share = _bracket_share(latest_demand, incoming_share, config)
        if target_share is None:
            return base
    return _negotiation_offer(state, role, own_value, surplus, target_share)


def _template_bayes_exact_guard(
    game: Mapping[str, Any], config: ChallengerConfig
) -> dict[str, Any]:
    """Canonicalize live text templates and guard exact buyer indifference."""

    state = _state(game)
    action_type = _action_type(game)
    text_mode = action_type == "seller_message" or (
        action_type == "buyer_decision" and not _binary_persuasion(state)
    )
    transformed = _canonicalized_persuasion_game(game) if text_mode else game

    if action_type == "seller_message":
        quality = str(state.get("current_quality", "")).strip().lower()
        if quality == "high":
            # Call the base first so invalid payloads retain base validation.
            persuasion_strategy(game)
            return {"message": config.persuasion_high_template}
        if quality != "low":
            return persuasion_strategy(game)
        # Pooling is Granola's private, retry-stable draw.  Consult it on the
        # untouched payload; canonicalization is a buyer-side interpretation and
        # must not alter the seller's reputation budget.
        base = persuasion_strategy(game)
        if _classify_signal(base.get("message")) == "positive":
            return {"message": config.persuasion_pooled_low_template}
        return {"message": config.persuasion_pass_template}

    base = persuasion_strategy(transformed)
    if action_type != "buyer_decision" or base.get("decision") != "yes":
        return base
    transformed_state = _state(transformed)
    if not _at_public_prior_break_even(transformed_state):
        return base
    if _classify_signal(transformed_state.get("seller_message")) != "positive":
        return {"decision": "no"}

    history = _history(transformed_state)
    negative_exposures = sum(
        _classify_signal(record.get("seller_message")) == "negative" for record in history
    )
    if negative_exposures < config.persuasion_required_negative_exposures:
        return {"decision": "no"}
    if any(_purchased_positive_low(record) for record in history):
        return {"decision": "no"}
    return base


def _negotiation_cell(
    game: Mapping[str, Any], state: Mapping[str, Any]
) -> tuple[str, str, float, float] | None:
    # Explicitly unknown and uncapped is the only ablation cell.  A cap present
    # despite ``horizon_known=False`` is treated as known-cap and delegated.
    if state.get("horizon_known") is not False:
        return None
    if state.get("max_rounds") is not None or state.get("total_rounds") is not None:
        return None
    if not bool(state.get("complete_information", False)):
        return None
    me = _player(game, state)
    other = _other(me)
    role = str(state.get(f"{me}_role", "")).strip().lower()
    other_role = str(state.get(f"{other}_role", "")).strip().lower()
    if {role, other_role} != {"seller", "buyer"}:
        return None
    try:
        own_value = float(state[f"{me}_value"])
        other_value = float(state[f"{other}_value"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(own_value) or not math.isfinite(other_value):
        return None
    seller_value = own_value if role == "seller" else other_value
    buyer_value = other_value if role == "seller" else own_value
    surplus = buyer_value - seller_value
    if surplus <= 0.0:
        return None
    return me, role, own_value, surplus


def _latest_rejected_own_share(
    state: Mapping[str, Any], me: str, role: str, own_value: float, surplus: float
) -> float | None:
    latest: tuple[int, int, float] | None = None
    for position, record in enumerate(_history(state), start=1):
        decision = _decision_name(record.get("decision"))
        record_round = _round_number(record.get("round")) or position
        candidates: list[tuple[Mapping[str, Any], Any, int]] = []
        primary = record.get("offer")
        if isinstance(primary, Mapping) and decision in {"reject", "rejectoffer", "rejected"}:
            candidates.append((primary, record.get("proposer"), record_round))
        elif decision in {"reject", "rejectoffer", "rejected"} and (
            "price" in record or "product_price" in record
        ):
            candidates.append((record, record.get("proposer"), record_round))

        # GLEE can serialize the rejecting player's counter inside the same
        # history record.  Once a later opponent offer is current, that prior
        # counter is precisely the latest rejected own demand used by the bracket.
        counter = record.get("counteroffer")
        counter_player = record.get("decided_by", record.get("current_player"))
        if isinstance(counter, Mapping):
            candidates.append((counter, counter_player, record_round + 1))
        elif counter is not None:
            candidates.append(({"price": counter}, counter_player, record_round + 1))

        for offer, fallback_player, order in candidates:
            proposer = offer.get("from_player", offer.get("proposer", fallback_player))
            if proposer != me:
                continue
            share = _negotiation_offer_share(offer, role, own_value, surplus)
            if share is None:
                continue
            if latest is None or (order, position) > latest[:2]:
                latest = (order, position, share)
    return None if latest is None else latest[2]


def _incoming_surplus_share(
    state: Mapping[str, Any], me: str, role: str, own_value: float, surplus: float
) -> float | None:
    other = _other(me)
    current = state.get("last_offer")
    if isinstance(current, Mapping):
        proposer = current.get("from_player", current.get("proposer"))
        if proposer in {None, other}:
            share = _negotiation_offer_share(current, role, own_value, surplus)
            if share is not None:
                return share
    latest: tuple[int, int, float] | None = None
    for position, record in enumerate(_history(state), start=1):
        candidates: list[Mapping[str, Any]] = []
        offer = record.get("offer")
        if isinstance(offer, Mapping):
            candidates.append(offer)
        counter = record.get("counteroffer")
        if isinstance(counter, Mapping):
            candidates.append(counter)
        elif counter is not None:
            candidates.append(
                {
                    "price": counter,
                    "from_player": record.get("decided_by", record.get("current_player")),
                }
            )
        for candidate in candidates:
            proposer = candidate.get(
                "from_player", candidate.get("proposer", record.get("proposer"))
            )
            if proposer != other:
                continue
            share = _negotiation_offer_share(candidate, role, own_value, surplus)
            if share is None:
                continue
            order = _round_number(candidate.get("round", record.get("round"))) or position
            if latest is None or (order, position) > latest[:2]:
                latest = (order, position, share)
    return None if latest is None else latest[2]


def _bracket_share(
    latest_demand: float | None,
    incoming_share: float | None,
    config: ChallengerConfig,
) -> float | None:
    if latest_demand is None or incoming_share is None:
        return None
    # An externally supplied history below our floor cannot satisfy both the
    # floor and monotonicity.  Delegate that inconsistent cell to the base.
    if latest_demand + 1e-12 < config.negotiation_acceptance_floor:
        return None
    midpoint = (latest_demand + incoming_share) / 2.0
    return max(config.negotiation_acceptance_floor, min(latest_demand, midpoint))


def _negotiation_counter(
    state: Mapping[str, Any], role: str, value: float, surplus: float, share: float
) -> dict[str, Any]:
    price = _price_for_share(value, role, surplus, share)
    action: dict[str, Any] = {
        "decision": "RejectOffer",
        "product_price": _clean_number(price),
    }
    if state.get("messages_allowed"):
        action["message"] = f"Let's split the available gains; my counter is {price:.2f}."
    return action


def _negotiation_offer(
    state: Mapping[str, Any], role: str, value: float, surplus: float, share: float
) -> dict[str, Any]:
    price = _price_for_share(value, role, surplus, share)
    action: dict[str, Any] = {"product_price": _clean_number(price)}
    if state.get("messages_allowed"):
        action["message"] = f"I propose a balanced price of {price:.2f}."
    return action


def _price_for_share(value: float, role: str, surplus: float, share: float) -> float:
    return value + share * surplus if role == "seller" else value - share * surplus


def _negotiation_offer_share(
    offer: Mapping[str, Any], role: str, own_value: float, surplus: float
) -> float | None:
    raw = offer.get("price", offer.get("product_price"))
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price):
        return None
    utility = price - own_value if role == "seller" else own_value - price
    return utility / surplus


def _canonicalized_persuasion_game(game: Mapping[str, Any]) -> Mapping[str, Any]:
    state = _state(game)
    transformed_state = dict(state)
    transformed_state["seller_message"] = _canonicalize_template(state.get("seller_message"))
    history = state.get("history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        transformed_history: list[Any] = []
        for raw in history:
            if not isinstance(raw, Mapping):
                transformed_history.append(raw)
                continue
            record = dict(raw)
            if "seller_message" in record:
                record["seller_message"] = _canonicalize_template(record["seller_message"])
            transformed_history.append(record)
        transformed_state["history"] = transformed_history
    transformed_game = dict(game)
    transformed_game["game_state"] = transformed_state
    return transformed_game


def _canonicalize_template(message: Any) -> Any:
    text_source: Any = message
    if isinstance(message, Mapping):
        text_source = message.get("message", message.get("text", message.get("content", "")))
    normalized = str(text_source or "").lower().replace("’", "'")
    normalized = re.sub(r"[‐‑‒–—−]", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    high_grade = re.search(r"\bhigh(?:\s|-)*grade\b", normalized) is not None
    low_template = "serviceable" in normalized or "nothing fancy" in normalized
    if high_grade and not low_template:
        return _CANONICAL_POSITIVE
    if low_template and not high_grade:
        return _CANONICAL_NEGATIVE
    return message


def _at_public_prior_break_even(state: Mapping[str, Any]) -> bool:
    try:
        p = float(state.get("p"))
        high = float(state.get("v"))
        low = float(state.get("u"))
        price = float(state.get("product_price"))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (p, high, low, price)):
        return False
    expected = p * high + (1.0 - p) * low
    scale = max(1.0, abs(expected), abs(price), abs(high), abs(low))
    return abs(expected - price) <= 1e-12 * scale


def _purchased_positive_low(record: Mapping[str, Any]) -> bool:
    if _classify_signal(record.get("seller_message")) != "positive":
        return False
    if str(record.get("quality", "")).strip().lower() != "low":
        return False
    if "bought" in record:
        bought = record["bought"]
    else:
        bought = record.get("buyer_decision")
    if isinstance(bought, bool):
        return bought
    return str(bought or "").strip().lower() in {"yes", "buy", "bought", "true", "1"}


def _binary_persuasion(state: Mapping[str, Any]) -> bool:
    return str(state.get("seller_message_type", "")).strip().lower() in {
        "binary",
        "recommendation",
    }


def _bargaining_final(state: Mapping[str, Any]) -> bool:
    known = bool(state.get("horizon_known", state.get("max_rounds") is not None))
    maximum = state.get("max_rounds")
    return known and maximum is not None and int(state.get("round", 1)) >= int(maximum)


def _bargaining_gain(offer: Mapping[str, Any], player: str) -> float | None:
    keys = (
        ("player_1_gain", "alice_gain")
        if player == "player_1"
        else (
            "player_2_gain",
            "bob_gain",
        )
    )
    for key in keys:
        if key not in offer:
            continue
        try:
            value = float(offer[key])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    return None


def _bargaining_offer_signature(offer: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return a complete allocation only when its aliases agree."""

    gains: list[float] = []
    for aliases in (("player_1_gain", "alice_gain"), ("player_2_gain", "bob_gain")):
        values: list[float] = []
        for key in aliases:
            if key not in offer:
                continue
            try:
                value = float(offer[key])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            values.append(value)
        if not values or any(value != values[0] for value in values[1:]):
            return None
        gains.append(values[0])
    return gains[0], gains[1]


def _bargaining_proposer(offer: Mapping[str, Any]) -> str | None:
    values = [offer[key] for key in ("proposer", "from_player") if key in offer]
    if not values or any(value != values[0] for value in values[1:]):
        return None
    return values[0] if values[0] in {"player_1", "player_2"} else None


def _bargaining_history_proposer(
    record: Mapping[str, Any], offer: Mapping[str, Any], me: str
) -> str | None:
    """Read a history proposer without guessing through conflicting metadata."""

    values = [
        source[key]
        for source in (record, offer)
        for key in ("proposer", "from_player")
        if key in source
    ]
    if values:
        if any(value != values[0] for value in values[1:]):
            return None
        return values[0] if values[0] in {"player_1", "player_2"} else None
    if record.get("decided_by") == me:
        return _other(me)
    return None


def _looks_like_bargaining_offer(record: Mapping[str, Any]) -> bool:
    return any(
        key in record for key in ("player_1_gain", "player_2_gain", "alice_gain", "bob_gain")
    )


def _same_bargaining_offer(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return all(
        _bargaining_gain(first, player) == _bargaining_gain(second, player)
        for player in ("player_1", "player_2")
    )


def _decision_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("decision")
    return str(value or "").strip().lower()


def _history(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = state.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return []
    return [record for record in history if isinstance(record, Mapping)]


def _round_number(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_number(value: float) -> Number:
    rounded = round(float(value), 6)
    return int(rounded) if rounded.is_integer() else rounded


def _state(game: Mapping[str, Any]) -> Mapping[str, Any]:
    state = game.get("game_state")
    if not isinstance(state, Mapping):
        raise TypeError("game_state must be a mapping")
    return state


def _action_type(game: Mapping[str, Any]) -> str:
    valid_actions = game.get("valid_actions")
    if not isinstance(valid_actions, Mapping) or not isinstance(valid_actions.get("type"), str):
        raise TypeError("valid_actions.type must be present")
    return str(valid_actions["type"])


def _player(game: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    player = game.get("your_player", state.get("current_player"))
    if player not in {"player_1", "player_2"}:
        raise ValueError(f"Unknown player identifier: {player!r}")
    return str(player)


def _other(player: str) -> str:
    return "player_2" if player == "player_1" else "player_1"
