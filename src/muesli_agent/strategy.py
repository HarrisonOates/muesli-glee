"""Transparent strategies for the three GLEE game families.

The policy is deliberately deterministic and dependency-free.  All adaptation is
reconstructed from ``game_state["history"]``, so concurrent games cannot leak state
into one another and restarting the process does not change an in-flight policy.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from statistics import median
from typing import Any

from .bargaining import bargaining_strategy as _adaptive_bargaining_strategy
from .negotiation import negotiation_strategy as _adaptive_negotiation_strategy
from .persuasion import persuasion_strategy as _adaptive_persuasion_strategy

Number = int | float


@dataclass(frozen=True)
class StrategyConfig:
    """The small set of intentionally exposed policy constants."""

    bargaining_min_opponent_share: float = 0.35
    bargaining_fair_share: float = 0.50
    bargaining_max_opponent_share: float = 0.52
    unknown_horizon_scale: float = 8.0
    negotiation_initial_surplus_claim: float = 0.58
    persuasion_prior_strength: float = 4.0
    persuasion_pooling_fraction: float = 0.50


DEFAULT_CONFIG = StrategyConfig()


def strategy(game: dict[str, Any]) -> dict[str, Any]:
    """Dispatch an SDK game payload to its family policy."""

    family = game.get("game_family")
    if family == "bargaining":
        return bargaining_strategy(game)
    if family == "negotiation":
        return negotiation_strategy(game)
    if family == "persuasion":
        return persuasion_strategy(game)
    raise ValueError(f"Unknown game family: {family!r}")


def _legacy_bargaining_strategy(
    game: Mapping[str, Any], config: StrategyConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Equilibrium-inspired offers with a deadline and observed-claim correction."""

    state = _state(game)
    action_type = _action_type(game)
    me = _me(game, state)
    money = float(state["money_to_divide"])
    if not math.isfinite(money) or money <= 0:
        raise ValueError("money_to_divide must be a positive finite number")

    if action_type == "offer":
        opponent_share = _bargaining_opponent_share(state, me, config)
        opponent_amount = _clean_number(money * opponent_share)
        own_amount = _clean_number(money - float(opponent_amount))

        if me == "player_1":
            action: dict[str, Any] = {
                "alice_gain": own_amount,
                "bob_gain": opponent_amount,
            }
        else:
            action = {
                "alice_gain": opponent_amount,
                "bob_gain": own_amount,
            }
        if _messages_allowed(state):
            action["message"] = (
                f"I propose an efficient agreement: {opponent_share:.0%} for you "
                f"and {1.0 - opponent_share:.0%} for me."
            )
        return action

    if action_type == "decision":
        offer = state.get("last_offer") or {}
        offered = _gain_for_player(offer, me)
        offered_share = offered / money

        # On a known last round every positive agreement strictly dominates no deal.
        if _is_final_round(state):
            return {"decision": "accept" if offered > 0 else "reject"}

        delta = _own_discount(state, me)
        next_state = dict(state)
        next_state["round"] = int(state.get("round", 1)) + 1
        next_own_share = 1.0 - _bargaining_opponent_share(next_state, me, config)
        reservation = _clamp(delta * next_own_share, 0.25, 0.50)
        return {"decision": "accept" if offered_share + 1e-12 >= reservation else "reject"}

    raise ValueError(f"Unsupported bargaining action type: {action_type!r}")


def _legacy_negotiation_strategy(
    game: Mapping[str, Any], config: StrategyConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Surplus-aware bilateral trade with history-based concession forecasting."""

    state = _state(game)
    action_type = _action_type(game)
    me = _me(game, state)
    role = str(state[f"{me}_role"]).lower()
    if role not in {"seller", "buyer"}:
        raise ValueError(f"Unknown negotiation role: {role!r}")
    own_value = float(state[f"{me}_value"])
    target = _negotiation_target_price(state, me, role, own_value, config)

    if action_type == "offer":
        return _negotiation_offer(target, role, state)

    if action_type == "decision":
        last_offer = state.get("last_offer") or {}
        price = float(last_offer["price"])
        profitable = price + 1e-12 >= own_value if role == "seller" else price <= own_value + 1e-12

        if profitable and (
            _is_final_round(state)
            or (role == "seller" and price + 1e-12 >= target)
            or (role == "buyer" and price <= target + 1e-12)
        ):
            return {"decision": "AcceptOffer"}

        if _is_final_round(state):
            # The GLEE API forbids a counteroffer on the final capped round.
            return {"decision": "RejectOffer"}

        action: dict[str, Any] = {
            "decision": "RejectOffer",
            "product_price": _clean_number(target),
        }
        if _messages_allowed(state):
            direction = "minimum" if role == "seller" else "maximum"
            action["message"] = f"My current {direction} workable price is {target:.2f}."
        return action

    raise ValueError(f"Unsupported negotiation action type: {action_type!r}")


def _legacy_persuasion_strategy(
    game: Mapping[str, Any], config: StrategyConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Reputation-bounded signaling for sellers and Bayesian calibration for buyers."""

    state = _state(game)
    action_type = _action_type(game)

    if action_type in {"seller_message", "seller_recommendation"}:
        quality = str(state.get("current_quality", "")).lower()
        if quality not in {"high", "low"}:
            raise ValueError("Seller must observe current_quality as 'high' or 'low'")
        recommend = quality == "high" or _pool_low_quality(game, state, config)

        if action_type == "seller_recommendation":
            return {"decision": "yes" if recommend else "no"}
        if recommend and quality == "high":
            return {"message": "I recommend buying this unit; it is high quality."}
        if recommend:
            # Strategic pooling is legal, but the text avoids a false quality claim.
            return {"message": "I recommend buying this unit at the posted price."}
        return {"message": "I do not recommend buying this unit."}

    if action_type == "buyer_decision":
        price = float(state["product_price"])
        high_value = float(state["v"])
        low_value = float(state["u"])
        if price <= low_value:
            return {"decision": "yes"}
        if price >= high_value:
            return {"decision": "no"}

        signal = _classify_message(state.get("seller_message"))
        posterior = _quality_posterior(state, signal, config.persuasion_prior_strength)
        expected_value = posterior * high_value + (1.0 - posterior) * low_value
        return {"decision": "yes" if expected_value > price + 1e-12 else "no"}

    raise ValueError(f"Unsupported persuasion action type: {action_type!r}")


def _bargaining_opponent_share(state: Mapping[str, Any], me: str, config: StrategyConfig) -> float:
    other = _other_player(me)
    delta_me = _own_discount(state, me)
    delta_other = _discount_for(state, other, fallback=0.95)

    # Rubinstein receiver share when the current proposer is ``me``.
    denominator = max(1e-9, 1.0 - delta_me * delta_other)
    equilibrium_receiver = delta_other * (1.0 - delta_me) / denominator
    equilibrium_receiver = _clamp(
        equilibrium_receiver,
        config.bargaining_min_opponent_share,
        config.bargaining_fair_share,
    )

    progress = _round_progress(state, config.unknown_horizon_scale)
    target = equilibrium_receiver + progress * (config.bargaining_fair_share - equilibrium_receiver)

    observed_claims = list(_bargaining_claims(state, other))
    if observed_claims:
        # The opponent's smallest own claim is the clearest revealed concession.
        revealed_floor = _clamp(
            min(observed_claims) - 0.03,
            config.bargaining_min_opponent_share,
            config.bargaining_max_opponent_share,
        )
        evidence_weight = 0.20 + 0.45 * progress
        target = (1.0 - evidence_weight) * target + evidence_weight * revealed_floor

    return _clamp(
        target,
        config.bargaining_min_opponent_share,
        config.bargaining_max_opponent_share,
    )


def _negotiation_target_price(
    state: Mapping[str, Any],
    me: str,
    role: str,
    own_value: float,
    config: StrategyConfig,
) -> float:
    progress = _round_progress(state, config.unknown_horizon_scale)
    other = _other_player(me)
    other_value_raw = state.get(f"{other}_value")

    if other_value_raw is not None:
        other_value = float(other_value_raw)
        seller_value = own_value if role == "seller" else other_value
        buyer_value = other_value if role == "seller" else own_value
        if buyer_value <= seller_value:
            return own_value
        surplus = buyer_value - seller_value
        claim = config.negotiation_initial_surplus_claim - 0.08 * progress
        if role == "seller":
            return seller_value + claim * surplus
        return buyer_value - claim * surplus

    opponent_prices = list(_negotiation_prices(state, from_player=other))
    if role == "seller":
        base = own_value * 1.30
        if opponent_prices:
            best = max(opponent_prices)
            forecast = _forecast_price(opponent_prices, role)
            evidence = max(best, forecast)
            target = base + (0.35 + 0.50 * progress) * (evidence - base)
        else:
            target = base
        target = max(own_value, target)
    else:
        base = own_value * 0.70
        if opponent_prices:
            best = min(opponent_prices)
            forecast = _forecast_price(opponent_prices, role)
            evidence = min(best, forecast)
            target = base + (0.35 + 0.50 * progress) * (evidence - base)
        else:
            target = base
        target = min(own_value, target)

    # Never reverse our own concession sequence.
    own_prices = list(_negotiation_prices(state, from_player=me))
    if own_prices:
        if role == "seller":
            target = min(target, own_prices[-1])
            target = max(target, own_value)
        else:
            target = max(target, own_prices[-1])
            target = min(target, own_value)
    return target


def _negotiation_offer(target: float, role: str, state: Mapping[str, Any]) -> dict[str, Any]:
    action: dict[str, Any] = {"product_price": _clean_number(target)}
    if _messages_allowed(state):
        if role == "seller":
            action["message"] = f"I can offer the product for {target:.2f}."
        else:
            action["message"] = f"I can offer {target:.2f} for the product."
    return action


def _pool_low_quality(
    game: Mapping[str, Any], state: Mapping[str, Any], config: StrategyConfig
) -> bool:
    """Use a conservative fraction of the Bayesian-persuasion pooling cap."""

    required = ("p", "v", "u", "product_price")
    if any(state.get(key) is None for key in required):
        return False
    p = _clamp(float(state["p"]), 0.0, 1.0)
    high_value = float(state["v"])
    low_value = float(state["u"])
    price = float(state["product_price"])
    if not (0.0 < p < 1.0 and low_value < price < high_value):
        return False

    denominator = (1.0 - p) * (price - low_value)
    persuasion_cap = min(1.0, p * (high_value - price) / max(denominator, 1e-12))
    progress = _round_progress(state, config.unknown_horizon_scale)
    reputation_multiplier = 0.25 + 0.75 * progress
    pooling_probability = (
        config.persuasion_pooling_fraction * reputation_multiplier * persuasion_cap
    )

    game_id = str(game.get("game_id", "offline"))
    round_number = int(state.get("round", 1))
    digest = hashlib.sha256(f"prism-v1:{game_id}:{round_number}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return draw < pooling_probability


def _quality_posterior(state: Mapping[str, Any], signal: str, prior_strength: float) -> float:
    p = _clamp(float(state["p"]), 0.0, 1.0)
    if signal == "negative":
        base = p * 0.20
    elif signal == "positive":
        base = min(0.99, p + 0.50 * (1.0 - p))
    else:
        base = p

    alpha = max(1e-6, prior_strength * base)
    beta = max(1e-6, prior_strength * (1.0 - base))
    for record in _history(state):
        if not bool(record.get("bought", False)):
            continue
        record_signal = _classify_message(record.get("seller_message"))
        if record_signal != signal:
            continue
        quality = str(record.get("quality", "")).lower()
        if quality == "high":
            alpha += 1.0
        elif quality == "low":
            beta += 1.0
    return alpha / (alpha + beta)


def _classify_message(message: Any) -> str:
    if isinstance(message, Mapping):
        message = message.get("message", message.get("decision", ""))
    text = str(message or "").strip().lower()
    if text in {"no", "false", "0"}:
        return "negative"
    if text in {"yes", "true", "1"}:
        return "positive"

    negative_phrases = (
        "do not recommend",
        "don't recommend",
        "not recommend",
        "low quality",
        "do not buy",
        "don't buy",
        "avoid",
        "pass",
    )
    if any(phrase in text for phrase in negative_phrases):
        return "negative"
    positive_phrases = ("recommend", "high quality", "worth", "great", "buy")
    if any(phrase in text for phrase in positive_phrases):
        return "positive"
    return "unknown"


def _forecast_price(prices: list[float], role: str) -> float:
    if len(prices) < 2:
        return prices[-1]
    changes = [later - earlier for earlier, later in pairwise(prices)]
    trend = median(changes[-3:])
    if role == "seller":
        trend = max(0.0, trend)  # Buyer offers should rise.
    else:
        trend = min(0.0, trend)  # Seller asks should fall.
    return prices[-1] + trend


def _bargaining_claims(state: Mapping[str, Any], player: str) -> Iterable[float]:
    money = float(state["money_to_divide"])
    for record in _history(state):
        offer = record.get("offer") or {}
        proposer = record.get("proposer", offer.get("proposer"))
        if proposer != player:
            continue
        try:
            yield _gain_for_player(offer, player) / money
        except (KeyError, TypeError, ValueError):
            continue


def _negotiation_prices(state: Mapping[str, Any], from_player: str) -> Iterable[float]:
    for record in _history(state):
        offer = record.get("offer") or {}
        proposer = offer.get("from_player", record.get("from_player"))
        if proposer != from_player:
            continue
        try:
            yield float(offer["price"])
        except (KeyError, TypeError, ValueError):
            continue


def _gain_for_player(offer: Mapping[str, Any], player: str) -> float:
    aliases = {
        "player_1": ("player_1_gain", "alice_gain"),
        "player_2": ("player_2_gain", "bob_gain"),
    }
    for key in aliases[player]:
        if key in offer:
            return float(offer[key])
    raise KeyError(f"Offer has no gain field for {player}")


def _history(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = state.get("history") or []
    return [record for record in history if isinstance(record, Mapping)]


def _round_progress(state: Mapping[str, Any], unknown_scale: float) -> float:
    round_number = max(1, int(state.get("round", 1)))
    if bool(state.get("horizon_known", state.get("max_rounds") is not None)):
        maximum = int(state.get("max_rounds", state.get("total_rounds", round_number)))
        if maximum <= 1:
            return 1.0
        return _clamp((round_number - 1) / (maximum - 1), 0.0, 1.0)
    return 1.0 - math.exp(-(round_number - 1) / unknown_scale)


def _is_final_round(state: Mapping[str, Any]) -> bool:
    if not bool(state.get("horizon_known", state.get("max_rounds") is not None)):
        return False
    maximum = state.get("max_rounds")
    if maximum is None:
        return False
    return int(state.get("round", 1)) >= int(maximum)


def _own_discount(state: Mapping[str, Any], me: str) -> float:
    return _discount_for(state, me, fallback=0.95)


def _discount_for(state: Mapping[str, Any], player: str, fallback: float) -> float:
    key = "delta_1" if player == "player_1" else "delta_2"
    value = state.get(key, fallback)
    return _clamp(float(value), 0.0, 1.0)


def _state(game: Mapping[str, Any]) -> Mapping[str, Any]:
    state = game.get("game_state")
    if not isinstance(state, Mapping):
        raise TypeError("game_state must be a mapping")
    return state


def _action_type(game: Mapping[str, Any]) -> str:
    actions = game.get("valid_actions")
    if not isinstance(actions, Mapping) or not isinstance(actions.get("type"), str):
        raise TypeError("valid_actions.type must be present")
    return str(actions["type"])


def _me(game: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    player = game.get("your_player", state.get("current_player"))
    if player not in {"player_1", "player_2"}:
        raise ValueError(f"Unknown player identifier: {player!r}")
    return str(player)


def _other_player(player: str) -> str:
    return "player_2" if player == "player_1" else "player_1"


def _messages_allowed(state: Mapping[str, Any]) -> bool:
    return bool(state.get("messages_allowed", False))


def _clean_number(value: float) -> Number:
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


# Keep the original import surface while routing both the dispatcher and direct users
# to the independently tested family modules.  The legacy implementations above remain
# temporarily as a readable record for the paper's baseline comparison.
bargaining_strategy = _adaptive_bargaining_strategy
negotiation_strategy = _adaptive_negotiation_strategy
persuasion_strategy = _adaptive_persuasion_strategy
