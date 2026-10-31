"""Retry-stable mixed policies for GLEE's repeated persuasion game.

The buyer deliberately separates two kinds of evidence.  Every recommendation is
observed, so the frequency of positive signals identifies a conservative
``high-always-positive`` signaling model.  Product quality, however, is observed
only after a purchase.  Those selected labels calibrate (but do not replace) the
frequency model, with shrinkage for incomplete coverage and recent failures.

The seller uses a reputation budget.  It always recommends high quality, phases
toward complete pooling when purchase is profitable under the public prior, and
otherwise privately mixes low-quality recommendations with a bounded probability.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersuasionConfig:
    """Small, explicit set of policy constants."""

    exposure_prior_strength: float = 0.1
    exposure_uncertainty_weight: float = 0.25
    signal_evidence_scale: float = 2.0
    outcome_prior_strength: float = 2.0
    outcome_uncertainty_weight: float = 0.35
    low_quality_failure_weight: float = 2.0
    outcome_half_life: float = 6.0
    monotone_grace_failures: int = 3
    buyer_endgame_fraction: float = 0.25
    buyer_information_value_weight: float = 0.10
    buyer_exploration_fraction: float = 0.25
    seller_response_adaptation_weight: float = 0.40
    seller_endgame_fraction: float = 0.25
    seller_known_cap_fraction: float = 0.85
    seller_unknown_pooling_target: float = 0.20
    seller_tie_pooling_target: float = 0.20
    unknown_horizon_scale: float = 12.0
    unknown_progress_cap: float = 0.65


DEFAULT_PERSUASION_CONFIG = PersuasionConfig()


def persuasion_strategy(
    game: Mapping[str, Any],
    config: PersuasionConfig = DEFAULT_PERSUASION_CONFIG,
) -> dict[str, str]:
    """Return a legal seller signal or buyer decision for a persuasion state."""

    state = _state(game)
    action_type = _action_type(game)

    if action_type in {"seller_message", "seller_recommendation"}:
        quality = str(state.get("current_quality", "")).strip().lower()
        if quality not in {"high", "low"}:
            raise ValueError("Seller must observe current_quality as 'high' or 'low'")
        positive = quality == "high" or _pool_low_quality(game, state, config)
        if action_type == "seller_recommendation":
            return {"decision": "yes" if positive else "no"}
        if positive:
            # Use the identical signal for high and strategically pooled low
            # quality.  Different positive phrasings would leak the private state.
            return {"message": "I highly recommend buying this product at the posted price."}
        return {"message": "I recommend passing on this product."}

    if action_type == "buyer_decision":
        return {"decision": "yes" if _buyer_should_buy(state, config) else "no"}

    raise ValueError(f"Unsupported persuasion action type: {action_type!r}")


def _buyer_should_buy(state: Mapping[str, Any], config: PersuasionConfig) -> bool:
    price = _finite_number(state.get("product_price"), "product_price")
    high_value = _finite_number(state.get("v"), "v")
    low_value = _finite_number(state.get("u"), "u")
    if high_value <= low_value:
        raise ValueError("v must be greater than u")

    # These guards are independent of any strategic model.
    if price <= low_value:
        return True
    if price >= high_value:
        return False

    p = _probability(state.get("p"), "p")
    threshold = (price - low_value) / (high_value - low_value)
    signal = _classify_signal(_seller_signal(state))
    if signal == "negative":
        return False
    if signal == "unknown":
        return p > threshold + 1e-12

    posterior = _positive_quality_posterior(state, p, config)
    if posterior > threshold + 1e-12:
        return True

    # A purchase is also an experiment: the buyer learns the realized quality
    # and can update the seller model on later rounds.  Spend a small amount on
    # that information only early in a sufficiently long game, and only when
    # the immediate loss is small.  This is deliberately a bonus rather than a
    # relaxed posterior, so clear negative deals and endgame decisions remain
    # unchanged.
    immediate_value = posterior * high_value + (1.0 - posterior) * low_value - price
    scale = max(1.0, abs(price), abs(high_value), abs(low_value))
    if immediate_value >= -1e-12 * scale:
        # Preserve strict behavior at public-prior indifference.  Cheap talk
        # alone must not manufacture positive value.
        return 0.0
    information_bonus = _buyer_information_bonus(
        state, posterior, high_value, low_value, price, config
    )
    return immediate_value + information_bonus > 1e-12


def _buyer_information_bonus(
    state: Mapping[str, Any],
    posterior: float,
    high_value: float,
    low_value: float,
    price: float,
    config: PersuasionConfig,
) -> float:
    """Bound the early value of learning a seller's signaling behavior."""

    current_round = max(1, int(state.get("round", 1)))
    total_rounds = _known_total_rounds(state, current_round)
    if total_rounds is None or total_rounds <= 1:
        return 0.0
    early_limit = max(1, math.floor(config.buyer_exploration_fraction * total_rounds))
    if current_round > early_limit:
        return 0.0
    remaining = max(0, total_rounds - current_round)
    if remaining < 2:
        return 0.0

    spread = high_value - low_value
    immediate_value = posterior * high_value + (1.0 - posterior) * low_value - price
    # Do not turn a clearly bad purchase into an experiment.  The tolerance is
    # scale-free and grows with the product's value range.
    loss_limit = 0.05 * spread
    if immediate_value < -loss_limit:
        return 0.0

    uncertainty = 4.0 * posterior * (1.0 - posterior)
    future_weight = min(1.0, remaining / max(1.0, total_rounds * 0.5))
    # A quality observation can at most resolve the high/low value spread.  The
    # small coefficient prevents exploration from dominating current payoff.
    return (
        config.buyer_information_value_weight
        * uncertainty
        * future_weight
        * spread
        * 0.05
    )


def _positive_quality_posterior(
    state: Mapping[str, Any], p: float, config: PersuasionConfig
) -> float:
    """Conservatively calibrate a positive signal from exposures and outcomes.

    Under the canonical monotone policy, high quality is always recommended and
    low quality is recommended with some pooling rate.  Hence
    ``P(high | positive) = p / P(positive)``.  Unlike purchase outcomes, signal
    frequency is observed without selection.  Purchased labels then update that
    estimate, while passes remain missing rather than being imputed as successes.
    """

    history = _history(state)
    current_round = max(1, int(state.get("round", len(history) + 1)))
    signals = [_classify_signal(_seller_signal(record)) for record in history]
    signals.append(_classify_signal(_seller_signal(state)))
    # An unfamiliar text is not evidence of a negative recommendation.  Exclude
    # it from this binary frequency model and let the public prior carry it.
    classified_signals = [signal for signal in signals if signal != "unknown"]
    positive_exposures = sum(signal == "positive" for signal in classified_signals)
    all_exposures = len(classified_signals)

    # Start at complete pooling (positive rate one), not at an invented boost for
    # persuasive language.  Observed negative signals are evidence of separation.
    strength = max(1e-6, config.exposure_prior_strength)
    alpha_rate = strength + positive_exposures
    beta_rate = 1e-6 + all_exposures - positive_exposures
    rate_mean = alpha_rate / (alpha_rate + beta_rate)
    rate_variance = (
        alpha_rate * beta_rate / ((alpha_rate + beta_rate) ** 2 * (alpha_rate + beta_rate + 1.0))
    )
    # Early uncertainty is valuable protection against a pooling seller.  After
    # several classified exposures, especially repeated negative signals, the
    # upper-confidence correction should shrink: continued separation is now
    # evidence about the seller's policy rather than a one-round accident.
    effective_uncertainty_weight = config.exposure_uncertainty_weight
    if all_exposures >= 4 and (all_exposures - positive_exposures) >= 2:
        effective_uncertainty_weight *= max(0.25, 4.0 / all_exposures)
    conservative_positive_rate = _clamp(
        rate_mean + effective_uncertainty_weight * math.sqrt(rate_variance),
        p,
        1.0,
    )
    frequency_posterior = _clamp(p / conservative_positive_rate, p, 1.0)
    negative_exposures = all_exposures - positive_exposures
    signal_reliability = 1.0 - math.exp(
        -negative_exposures / max(1e-6, config.signal_evidence_scale)
    )
    # A single negative signal suggests separation but cannot establish a
    # perfectly revealing sender.  Certainty grows only with repeated exposures.
    frequency_posterior = min(
        frequency_posterior,
        p + signal_reliability * (1.0 - p),
    )

    weighted_high = 0.0
    weighted_low = 0.0
    labelled_positive = 0
    revealed_low_count = 0
    for index, record in enumerate(history, start=1):
        if _classify_signal(_seller_signal(record)) != "positive":
            continue
        if not _was_bought(record):
            continue
        quality = str(record.get("quality", "")).strip().lower()
        if quality not in {"high", "low"}:
            continue
        labelled_positive += 1
        record_round = _record_round(record, index)
        age = max(0, current_round - record_round)
        recency = 0.5 ** (age / max(1e-6, config.outcome_half_life))
        if quality == "high":
            weighted_high += recency
        else:
            revealed_low_count += 1
            weighted_low += recency * config.low_quality_failure_weight

    if labelled_positive:
        # Only a selected subset has labels.  Lower coverage reduces its effective
        # sample size; it never turns an unobserved pass into a favourable outcome.
        coverage = labelled_positive / max(1, positive_exposures)
        coverage_weight = math.sqrt(_clamp(coverage, 0.0, 1.0))
        alpha_quality = (
            config.outcome_prior_strength * frequency_posterior + coverage_weight * weighted_high
        )
        beta_quality = (
            config.outcome_prior_strength * (1.0 - frequency_posterior)
            + coverage_weight * weighted_low
        )
        total = max(1e-6, alpha_quality + beta_quality)
        outcome_mean = alpha_quality / total
        outcome_variance = alpha_quality * beta_quality / (total**2 * (total + 1.0))
        calibrated = outcome_mean - config.outcome_uncertainty_weight * math.sqrt(
            max(0.0, outcome_variance)
        )
    else:
        calibrated = frequency_posterior

    # The public prior is an exact i.i.d. quality probability.  One or two low
    # draws are ordinary under a monotone pooling policy, not enough to conclude
    # that the seller perversely recommends low while hiding high.  Three recent
    # failures can reject that structural assumption; this retains protection
    # against adversarial/reversed signaling and preserves the existing bad-run
    # stop.  At a break-even prior, the floor still results in a strict pass.
    if revealed_low_count < config.monotone_grace_failures:
        calibrated = max(p, calibrated)

    # A finite-horizon seller has less future reputation to protect near the end.
    # Collapse learned signal value toward the public prior over the last portion.
    total_rounds = _known_total_rounds(state, current_round)
    if total_rounds is None:
        # Unknown continuation is not a deadline.  Reputation remains valuable.
        reputation_weight = 1.0
    else:
        remaining = max(0, total_rounds - current_round)
        endgame_span = max(1.0, config.buyer_endgame_fraction * max(1, total_rounds - 1))
        reputation_weight = _clamp(remaining / endgame_span, 0.0, 1.0)
    return _clamp(p + reputation_weight * (calibrated - p), 0.0, 1.0)


def _pool_low_quality(
    game: Mapping[str, Any], state: Mapping[str, Any], config: PersuasionConfig
) -> bool:
    probability = _low_pooling_probability(state, config)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return _private_draw(game, state) < probability


def _low_pooling_probability(state: Mapping[str, Any], config: PersuasionConfig) -> float:
    p = _probability(state.get("p"), "p")
    history = _history(state)
    current_round = max(1, int(state.get("round", len(history) + 1)))
    total_rounds = _known_total_rounds(state, current_round)

    price_raw = state.get("product_price")
    high_raw = state.get("v")
    low_raw = state.get("u")
    values_known = price_raw is not None and high_raw is not None and low_raw is not None
    pooled_prior_profitable = False
    if values_known:
        price = _finite_number(price_raw, "product_price")
        high_value = _finite_number(high_raw, "v")
        low_value = _finite_number(low_raw, "u")
        if high_value <= low_value:
            values_known = False

    if values_known:
        prior_value = p * high_value + (1.0 - p) * low_value
        scale = max(1.0, abs(price), abs(high_value), abs(low_value))
        margin = prior_value - price
        # When pooling itself gives the buyer strictly positive expected value,
        # the sustainable pooling cap is one.  Still phase it in: observed GLEE
        # buyers condition on recommendations even in these configurations.
        if margin > 1e-12 * scale:
            pooled_prior_profitable = True
            base_target = 1.0
        elif not (0.0 < p < 1.0 and low_value < price < high_value):
            base_target = 0.0
        else:
            denominator = (1.0 - p) * (price - low_value)
            persuasion_cap = _clamp(p * (high_value - price) / max(1e-12, denominator), 0.0, 1.0)
            if abs(margin) <= 1e-12 * scale:
                # At exact indifference, complete pooling conveys no actionable
                # information to strict buyers.  Build reputation before spending it.
                base_target = min(persuasion_cap, config.seller_tie_pooling_target)
            else:
                base_target = config.seller_known_cap_fraction * persuasion_cap
    else:
        # Missing buyer values are normal under incomplete information.  A small
        # reputation budget plus the endgame rule needs no guessed valuation.
        base_target = config.seller_unknown_pooling_target

    if total_rounds is None:
        pseudo_progress = 1.0 - math.exp(
            -(current_round - 1) / max(1e-6, config.unknown_horizon_scale)
        )
        progress = min(config.unknown_progress_cap, pseudo_progress)
        urgency = 0.0
    else:
        if pooled_prior_profitable and current_round >= total_rounds - 2:
            return 1.0
        if total_rounds <= 1 or current_round >= total_rounds:
            return 1.0
        progress = _clamp((current_round - 1) / max(1, total_rounds - 1), 0.0, 1.0)
        endgame_start = 1.0 - config.seller_endgame_fraction
        urgency = _clamp(
            (progress - endgame_start) / max(1e-6, config.seller_endgame_fraction),
            0.0,
            1.0,
        )

    positive_records = [
        record for record in history if _classify_signal(_seller_signal(record)) == "positive"
    ]
    negative_records = [
        record for record in history if _classify_signal(_seller_signal(record)) == "negative"
    ]
    if any(_was_bought(record) for record in negative_records):
        # A buyer who purchased after our negative signal has demonstrated that
        # even a known-low recommendation does not deter them.  Pool thereafter.
        base_target = 1.0
    positive_buys = sum(_was_bought(record) for record in positive_records)
    responsiveness = (positive_buys + 1.0) / (len(positive_records) + 2.0)
    if len(positive_records) >= 3:
        base_target *= 0.5 + 0.5 * responsiveness

    base_target *= _seller_trust_multiplier(positive_records)

    # Explicitly classify the buyer's response to our signals.  A buyer who
    # keeps purchasing after negative recommendations is not giving us much
    # reason to preserve separation; spend the pooling budget faster.  A buyer
    # who buys positives and passes negatives remains worth persuading, so the
    # existing conservative target is retained.
    buyer_responsiveness, response_observations = _buyer_signal_responsiveness(history)
    if response_observations >= 4:
        ignoring_fraction = 1.0 - buyer_responsiveness
        base_target += (
            config.seller_response_adaptation_weight
            * ignoring_fraction
            * (1.0 - base_target)
        )
        urgency = 1.0 - (1.0 - urgency) * (
            1.0 - 0.50 * config.seller_response_adaptation_weight * ignoring_fraction
        )
    base_target = _clamp(base_target, 0.0, 1.0)

    estimated_lows, estimated_pooled_lows = _estimate_seller_low_history(history, p)
    if estimated_lows >= 4.0 and 0.0 < base_target < 1.0:
        # HMAC mixing has the desired rate in expectation.  If a short realized
        # run overshoots it, reduce (but never deterministically force) the next
        # probability.  Do not chase an undershoot with a predictable catch-up.
        realized_rate = estimated_pooled_lows / estimated_lows
        if realized_rate > base_target:
            base_target *= _clamp(base_target / realized_rate, 0.35, 1.0)

    # A larger incentive-compatible budget can be spent more quickly.  When the
    # pooled prior itself is profitable (target one), rise from 25% to 95%; for
    # narrow/unknown budgets retain the slower reputation-building schedule.
    schedule_start = 0.15 + 0.10 * base_target
    schedule_finish = 0.80 + 0.15 * base_target
    reputation_schedule = schedule_start + (schedule_finish - schedule_start) * progress
    target = _clamp(base_target * reputation_schedule, 0.0, 1.0)
    target += (1.0 - target) * urgency
    return _clamp(target, 0.0, 1.0)


def _classify_signal(message: Any) -> str:
    if isinstance(message, Mapping):
        # Binary mode is documented as ``decision``.  The other short fields are
        # plausible serialized labels.  Prefer them over free text, but treat
        # disagreement between structured labels as unknown rather than guessing.
        structured = []
        for key in ("decision", "recommendation", "label", "signal", "value"):
            if key not in message:
                continue
            classification = _classify_structured_label(message[key])
            if classification != "unknown":
                structured.append(classification)
        if structured:
            return structured[0] if len(set(structured)) == 1 else "unknown"
        message = message.get("message", message.get("text", message.get("content", "")))

    structured_scalar = _classify_structured_label(message)
    if structured_scalar != "unknown":
        return structured_scalar

    text = _normalize_text(message)
    if not text:
        return "unknown"

    # Negated negative qualities ("not low quality") and double-negation idioms
    # are not safely recoverable with a compact deterministic parser.
    ambiguous_negations = (
        "not low quality",
        "not poor quality",
        "not bad quality",
        "not bad",
        "isn't low quality",
        "is not low quality",
    )
    if any(phrase in text for phrase in ambiguous_negations):
        return "unknown"

    directive_negative = (
        "do not recommend",
        "don't recommend",
        "not recommend",
        "cannot recommend",
        "can't recommend",
        "recommend passing",
        "do not buy",
        "don't buy",
        "should not buy",
        "shouldn't buy",
        "avoid",
        "skip",
        "refrain",
        "pass on",
        "pass today",
        "want to pass",
        "sit this round out",
        "sit out this round",
    )
    quality_negative = (
        "low quality",
        "poor quality",
        "bad quality",
        "not worth",
        "not the premium",
        "not premium",
        "not my best",
        "isn't my best",
        "is not my best",
        "not up to my usual standard",
        "not up to the usual standard",
    )
    strong_positive = (
        "high quality",
        "good quality",
        "excellent",
        "premium",
        "valuable",
        "worth it",
        "worthwhile",
        "great product",
    )
    directive_positive = (
        "recommend buying",
        "recommend purchasing",
        "recommend this",
        "should buy",
        "should purchase",
        "buy this",
        "purchase this",
        "highly recommend",
    )

    has_directive_negative = any(phrase in text for phrase in directive_negative)
    has_quality_negative = any(phrase in text for phrase in quality_negative)
    positive_scan_text = text
    for phrase in quality_negative:
        positive_scan_text = positive_scan_text.replace(phrase, " ")
    has_strong_positive = any(phrase in positive_scan_text for phrase in strong_positive)
    has_directive_positive = any(phrase in text for phrase in directive_positive)

    # Conflicting claims are deliberately unknown.  A clear negated directive
    # such as "do not recommend buying" remains negative even though it contains
    # the token "buy"; positive quality alongside that directive is a conflict.
    if has_directive_negative and has_strong_positive:
        return "unknown"
    if has_quality_negative and has_strong_positive:
        return "unknown"
    if has_quality_negative and has_directive_positive and not has_directive_negative:
        return "unknown"
    if has_directive_negative or has_quality_negative:
        return "negative"
    if has_strong_positive or has_directive_positive:
        return "positive"

    # Generic persuasion words are weak but useful only when no negation is
    # present.  An unmatched negation makes the whole message ambiguous.
    if re.search(r"\b(?:not|never|no|cannot|can't|don't|isn't|shouldn't)\b", text):
        return "unknown"
    if re.search(r"\b(?:recommend|buy|purchase|great|good|worth)\b", text):
        return "positive"
    return "unknown"


def _seller_signal(record: Mapping[str, Any]) -> Any:
    """Extract the seller's signal across text and structured payload shapes."""

    for key in ("seller_message", "seller_recommendation", "seller_signal"):
        value = record.get(key)
        if value is not None:
            return value
    return None


def _classify_structured_label(value: Any) -> str:
    if isinstance(value, bool):
        return "positive" if value else "negative"
    if isinstance(value, (int, float)) and value in {0, 1}:
        return "positive" if int(value) == 1 else "negative"
    normalized = _normalize_text(value)
    if normalized in {"yes", "true", "1", "positive", "recommend", "buy"}:
        return "positive"
    if normalized in {"no", "false", "0", "negative", "pass", "reject", "do not buy"}:
        return "negative"
    return "unknown"


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:!?\"'")


def _was_bought(record: Mapping[str, Any]) -> bool:
    if "bought" in record:
        bought = record["bought"]
        if isinstance(bought, bool):
            return bought
        return str(bought).strip().lower() in {"yes", "buy", "bought", "true", "1"}
    decision = str(record.get("buyer_decision", "")).strip().lower()
    return decision in {"yes", "buy", "bought", "true", "1"}


def _buyer_signal_responsiveness(history: list[Mapping[str, Any]]) -> tuple[float, int]:
    """Estimate whether the buyer changes behavior with the seller's signal."""

    positive_buys = positive_exposures = negative_buys = negative_exposures = 0
    for record in history:
        signal = _classify_signal(_seller_signal(record))
        if signal == "positive":
            positive_exposures += 1
            positive_buys += int(_was_bought(record))
        elif signal == "negative":
            negative_exposures += 1
            negative_buys += int(_was_bought(record))

    observations = positive_exposures + negative_exposures
    if not positive_exposures or not negative_exposures:
        return 1.0, observations

    positive_buy_rate = positive_buys / positive_exposures
    negative_buy_rate = negative_buys / negative_exposures
    # 1 means perfectly signal-responsive; 0 means the observed direction is
    # reversed.  Equal behavior on both signals maps to the neutral midpoint.
    responsiveness = _clamp(0.5 + 0.5 * (positive_buy_rate - negative_buy_rate), 0.0, 1.0)
    return responsiveness, observations


def _seller_trust_multiplier(positive_records: list[Mapping[str, Any]]) -> float:
    """React to buyer behavior after a revealed lie, not to the lie alone."""

    revealed_lie_indices = [
        index
        for index, record in enumerate(positive_records)
        if _was_bought(record) and str(record.get("quality", "")).strip().lower() == "low"
    ]
    if not revealed_lie_indices:
        return 1.0

    subsequent = positive_records[revealed_lie_indices[-1] + 1 :]
    if not subsequent:
        # Initially wait for one response before another reputation shock.  Once
        # the buyer has repeatedly purchased revealed lows and later returned,
        # their demonstrated tolerance supports a larger (still reduced) budget.
        return 0.65 if len(revealed_lie_indices) >= 2 else 0.25
    post_lie_buy_rate = sum(_was_bought(record) for record in subsequent) / len(subsequent)
    return _clamp(0.15 + 0.85 * post_lie_buy_rate, 0.15, 1.0)


def _estimate_seller_low_history(history: list[Mapping[str, Any]], p: float) -> tuple[float, float]:
    """Estimate past low and pooled-low counts without double-counting missing data."""

    known_lows = 0.0
    known_pooled_lows = 0.0
    unresolved: list[Mapping[str, Any]] = []
    for record in history:
        quality = str(record.get("quality", "")).strip().lower()
        signal = _classify_signal(_seller_signal(record))
        if quality == "low":
            known_lows += 1.0
            known_pooled_lows += float(signal == "positive")
        elif quality == "high":
            continue
        elif signal == "negative":
            # Under our own monotone policy, a negative signal certifies low
            # quality even if the serialized history omits the quality label.
            known_lows += 1.0
        else:
            unresolved.append(record)

    expected_unresolved_lows = (1.0 - p) * len(unresolved)
    if unresolved:
        positive_fraction = sum(
            _classify_signal(_seller_signal(record)) == "positive" for record in unresolved
        ) / len(unresolved)
    else:
        positive_fraction = 0.0
    return (
        known_lows + expected_unresolved_lows,
        known_pooled_lows + expected_unresolved_lows * positive_fraction,
    )


def _private_draw(game: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    """Stable per game/round mixing draw, privately keyed in live execution."""

    secret = os.environ.get("GLEE_API_KEY", "")
    key = secret.encode("utf-8") if secret else b"prism-offline-deterministic-fallback-v3"
    game_id = str(game.get("game_id", "offline"))
    round_number = max(1, int(state.get("round", 1)))
    payload = f"prism-persuasion-v3\x1f{game_id}\x1f{round_number}".encode()
    digest = hmac.new(key, payload, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _record_round(record: Mapping[str, Any], fallback: int) -> int:
    try:
        return max(1, int(record.get("round", fallback)))
    except (TypeError, ValueError):
        return fallback


def _history(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = state.get("history") or []
    return [record for record in history if isinstance(record, Mapping)]


def _known_total_rounds(state: Mapping[str, Any], current_round: int) -> int | None:
    if state.get("horizon_known") is False:
        return None
    raw = state.get("total_rounds", state.get("max_rounds"))
    if raw is None:
        return None
    try:
        return max(current_round, int(raw))
    except (TypeError, ValueError):
        return None


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


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _probability(value: Any, name: str) -> float:
    probability = _finite_number(value, name)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return probability


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
