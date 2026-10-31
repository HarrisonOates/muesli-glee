"""Expected-utility bilateral negotiation policy for the GLEE competition.

The policy works in own-utility space so the seller and buyer use the same
decision rule.  A small discrete belief represents the maximum utility the
opponent is currently likely to concede.  Offers, counteroffers, and rejections
update that belief; candidate prices are scored by their expected payoff and
the value of continuing after rejection.  A Boulware schedule supplies the
complete-information prior, rather than directly dictating the price.

All state is reconstructed from the current payload.  Concurrent games remain
independent and restarts cannot erase learned in-game evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, NamedTuple

Number = int | float


@dataclass(frozen=True)
class NegotiationConfig:
    """Parameters exposed for controlled policy experiments."""

    complete_initial_surplus_share: float = 0.72
    complete_final_surplus_share: float = 0.50
    complete_terminal_own_surplus_share: float = 0.99
    complete_terminal_fairness_share: float = 0.50
    terminal_fairness_opponent_names: tuple[str, ...] = ("punish",)
    incomplete_initial_margin: float = 0.38
    incomplete_final_margin: float = 0.12
    concession_exponent: float = 3.0
    unknown_horizon_scale: float = 18.0
    private_surplus_span: float = 0.80
    valuation_factors: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5)
    valuation_scales: tuple[float, ...] = (100.0, 10_000.0, 1_000_000.0)
    grid_match_tolerance: float = 1e-8
    private_initial_threshold_fraction: float = 0.70
    candidate_count: int = 41
    complete_uniform_mixture: float = 0.80
    complete_threshold_std: float = 0.18
    action_evidence_reliability: float = 0.92
    opponent_concession_forecast_weight: float = 0.50
    rejection_evidence_reliability: float = 0.88
    rejection_drift_fraction: float = 0.15
    continuation_survival: float = 0.70
    certainty_equivalent_discount: float = 0.85
    private_evidence_certainty_discount: float = 0.70
    private_terminal_opponent_buffer_fraction: float = 0.01
    stalled_opponent_patience: int = 3
    uncapped_private_loss_patience: int = 10
    uncapped_complete_stall_patience: int = 10
    stalled_positive_acceptance_fraction: float = 0.85
    best_seen_acceptance_fraction: float = 0.90
    robust_forecast_window: int = 4
    reciprocal_concession_weight: float = 0.35


DEFAULT_CONFIG = NegotiationConfig()


class _Offer(NamedTuple):
    price: float
    player: str
    round: int


@dataclass(frozen=True)
class _Belief:
    """Discrete posterior over the opponent's concession threshold.

    A threshold ``a`` means the opponent accepts offers giving us utility at
    most ``a``.  This absorbs both their private valuation and their strategic
    willingness to concede at the current time.
    """

    thresholds: tuple[float, ...]
    ceilings: tuple[float, ...]
    probabilities: tuple[float, ...]
    grid_based: bool = False

    def acceptance_probability(self, utility: float) -> float:
        return sum(
            probability
            for threshold, probability in zip(self.thresholds, self.probabilities, strict=True)
            if math.isfinite(threshold) and threshold + 1e-12 >= utility
        )


class _OfferChoice(NamedTuple):
    utility: float
    expected_value: float
    acceptance_probability: float


def negotiation_strategy(
    game: Mapping[str, Any], config: NegotiationConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Return a legal, individually rational bilateral-negotiation action.

    Complete-information games use the known gains from trade.  Under private
    values, the policy marginalizes the documented ``F * M`` valuation grid,
    with a scale-free fallback for synthetic values outside that grid.  In both
    cases it updates on observed actions, searches candidate prices, and never
    reverses its own concession sequence.
    """

    state = _mapping(game.get("game_state"), "game_state")
    action_type = _action_type(game)
    me = _player(game.get("your_player", state.get("current_player")))
    other = _other_player(me)
    role = str(state.get(f"{me}_role", "")).lower()
    other_role = str(state.get(f"{other}_role", "")).lower()
    if role not in {"seller", "buyer"} or other_role not in {"seller", "buyer"}:
        raise ValueError("Negotiation roles must be 'seller' and 'buyer'")
    if role == other_role:
        raise ValueError("A negotiation requires one seller and one buyer")

    own_value = _finite_number(state.get(f"{me}_value"), f"{me}_value")
    offers = _offers(state)
    _validate_config(config)
    terminal_action = _complete_terminal_offer(
        state,
        action_type,
        me,
        other,
        role,
        own_value,
        offers,
        _opponent_name(game),
        config,
    )
    if terminal_action is not None:
        return terminal_action
    bracket_action = _complete_unknown_horizon_bracket(
        state, action_type, me, other, role, own_value, offers, config
    )
    if bracket_action is not None:
        return bracket_action
    choice = _choose_offer(state, action_type, me, other, role, own_value, offers, config)
    target_utility = _private_terminal_buffered_utility(
        state,
        action_type,
        other,
        choice.utility,
        config.private_terminal_opponent_buffer_fraction,
    )
    target_price = _price_for_utility(own_value, role, target_utility)

    if action_type == "decision":
        forecast = _robust_opponent_concession_forecast(
            offers,
            other,
            own_value,
            role,
            config.reciprocal_concession_weight,
            config.robust_forecast_window,
        )
        if forecast is not None:
            # Do not demand more than a conservative forecast of the next
            # concession once the opponent has demonstrated a trend.  This is
            # a bounded ANAC-style tit-for-tat response; it never makes an
            # individually irrational offer because forecast is based in our
            # utility space and the existing choice remains IR.
            target_price = _price_for_utility(own_value, role, min(choice.utility, forecast))

    if action_type == "offer":
        return _offer_action(target_price, role, state)

    if action_type != "decision":
        raise ValueError(f"Unsupported negotiation action type: {action_type!r}")

    last_offer = _mapping(state.get("last_offer"), "last_offer")
    incoming_price = _finite_number(
        last_offer.get("price", last_offer.get("product_price")), "last_offer.price"
    )
    incoming_utility = _utility(incoming_price, own_value, role)
    tolerance = 1e-12 * max(1.0, abs(own_value), abs(incoming_price))

    # Reaching the cap eliminates the counteroffer.  Any weakly profitable deal
    # dominates (or ties) the zero outside option, irrespective of aspiration.
    if _is_final_round(state):
        if incoming_utility >= -tolerance:
            return {"decision": "AcceptOffer"}
        return {"decision": "RejectOffer"}

    # Acceptance compares a certain payoff with the certainty-equivalent value
    # of countering, not with the counteroffer's face value.  This avoids the
    # common error of rejecting a strong sure offer because an ambitious price
    # might (with low probability) be accepted.
    certainty_discount = _certainty_equivalent_discount(state, other, config)
    continuation_value = certainty_discount * choice.expected_value
    best_seen_utility = max(
        (
            _utility(offer.price, own_value, role)
            for offer in offers
            if offer.player == other
        ),
        default=incoming_utility,
    )
    best_seen_anchor = (
        config.best_seen_acceptance_fraction * best_seen_utility
        if best_seen_utility > tolerance
        else 0.0
    )
    if incoming_utility >= max(0.0, continuation_value, best_seen_anchor) - tolerance:
        return {"decision": "AcceptOffer"}

    # A profitable offer that survives several rounds without improvement is
    # evidence against the optimistic continuation model.  Take the deal once
    # it clears a discounted fraction of that model, rather than repeatedly
    # paying for a counteroffer that the opponent has shown no willingness to
    # improve.  This is deliberately bounded below the normal acceptance rule
    # and does not turn a zero-payoff stall into an agreement.
    if _should_accept_stalled_profitable_offer(
        offers,
        other,
        own_value,
        role,
        incoming_utility,
        continuation_value,
        config.stalled_opponent_patience,
        config.stalled_positive_acceptance_fraction,
    ):
        return {"decision": "AcceptOffer"}

    # With known incompatible valuations there is no mutually individually
    # rational price.  End the game instead of exchanging futile counters.
    if _known_nonpositive_surplus(state, me, role, own_value):
        return {"decision": "WalkAway"}
    if _should_walk_away_from_stalled_opponent(
        offers, other, own_value, role, incoming_utility, config.stalled_opponent_patience
    ):
        return {"decision": "WalkAway"}
    if (
        state.get(f"{other}_value") is None
        and not _has_known_horizon(state)
        and _should_walk_away_after_persistent_losses(
            offers,
            other,
            own_value,
            role,
            config.uncapped_private_loss_patience,
        )
    ):
        return {"decision": "WalkAway"}
    if choice.expected_value <= tolerance and choice.acceptance_probability <= tolerance:
        return {"decision": "WalkAway"}

    action: dict[str, Any] = {
        "decision": "RejectOffer",
        "product_price": _clean_number(target_price),
    }
    if _messages_allowed(state):
        if role == "seller":
            action["message"] = f"I can continue at {target_price:.2f}."
        else:
            action["message"] = f"My counteroffer is {target_price:.2f}."
    return action


def _certainty_equivalent_discount(
    state: Mapping[str, Any], other: str, config: NegotiationConfig
) -> float:
    """Discount continuation after evidence in capped private-value games.

    Millet isolated this acceptance change over two live cohorts.  Granola now
    owns the winning cell while retaining its more patient default before any
    history is available and in public-value or uncapped negotiations.
    """

    history = state.get("history")
    private_capped_with_evidence = (
        state.get(f"{other}_value") is None
        and state.get("horizon_known") is True
        and state.get("max_rounds") is not None
        and isinstance(history, list)
        and any(isinstance(record, Mapping) for record in history)
    )
    if private_capped_with_evidence:
        return min(
            config.certainty_equivalent_discount,
            config.private_evidence_certainty_discount,
        )
    return config.certainty_equivalent_discount


def _opponent_name(game: Mapping[str, Any]) -> str | None:
    opponent = game.get("opponent")
    if not isinstance(opponent, Mapping):
        return None
    name = opponent.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def _private_terminal_buffered_utility(
    state: Mapping[str, Any],
    action_type: str,
    other: str,
    utility: float,
    buffer_fraction: float,
) -> float:
    """Make a marginal private-value receiver strictly better off.

    The private grid optimizer naturally chooses an exact latent valuation
    breakpoint.  At that price the marginal compatible opponent earns zero and
    may rationally reject.  Sacrificing one percent of our chosen utility makes
    that breakpoint strictly profitable at a one-percent conditional-payoff
    cost, without changing any nonterminal or public-value proposal.
    """

    if (
        action_type == "offer"
        and _is_final_round(state)
        and state.get(f"{other}_value") is None
        and utility > 0.0
    ):
        return (1.0 - buffer_fraction) * utility
    return utility


def _complete_terminal_offer(
    state: Mapping[str, Any],
    action_type: str,
    me: str,
    other: str,
    role: str,
    own_value: float,
    offers: list[_Offer],
    opponent_name: str | None,
    config: NegotiationConfig,
) -> dict[str, Any] | None:
    """Use final-proposer leverage when both reservation values are public.

    The receiver has no continuation action and any strictly profitable deal
    dominates no trade.  Leave one percent of the known surplus by default,
    while never reversing an earlier concession in a multi-round history.
    """

    if action_type != "offer" or not _is_final_round(state):
        return None
    other_value_raw = state.get(f"{other}_value")
    if other_value_raw is None:
        return None
    other_value = _finite_number(other_value_raw, f"{other}_value")
    seller_value = own_value if role == "seller" else other_value
    buyer_value = other_value if role == "seller" else own_value
    surplus = buyer_value - seller_value
    if surplus <= 0.0:
        return None

    target_share = config.complete_terminal_own_surplus_share
    fairness_names = {name.strip().casefold() for name in config.terminal_fairness_opponent_names}
    if opponent_name is not None and opponent_name.casefold() in fairness_names:
        target_share = min(target_share, config.complete_terminal_fairness_share)
    own_demands = [
        _utility(offer.price, own_value, role) / surplus
        for offer in offers
        if offer.player == me
    ]
    if own_demands:
        target_share = min(target_share, max(0.0, own_demands[-1]))
    price = _price_for_utility(own_value, role, target_share * surplus)
    return _offer_action(price, role, state)


def _complete_unknown_horizon_bracket(
    state: Mapping[str, Any],
    action_type: str,
    me: str,
    other: str,
    role: str,
    own_value: float,
    offers: list[_Offer],
    config: NegotiationConfig,
) -> dict[str, Any] | None:
    """Apply the Buckwheat ablation in the no-deadline public-value cell.

    In this cell there is no documented terminal round and both values are
    visible, so a fixed Nash-style floor is a useful hedge against opponents
    that stop conceding.  All capped or private-information games continue to
    use the adaptive policy below.
    """

    if (
        state.get("horizon_known") is not False
        or state.get("max_rounds") is not None
        or state.get("total_rounds") is not None
        or not bool(state.get("complete_information", False))
        or action_type not in {"offer", "decision"}
    ):
        return None

    other_value_raw = state.get(f"{other}_value")
    if other_value_raw is None:
        return None
    other_value = _finite_number(other_value_raw, f"{other}_value")
    seller_value = own_value if role == "seller" else other_value
    buyer_value = other_value if role == "seller" else own_value
    surplus = buyer_value - seller_value
    if surplus <= 0.0:
        return None

    def share(utility: float) -> float:
        return utility / surplus

    own_offers = [offer for offer in offers if offer.player == me]
    other_offers = [offer for offer in offers if offer.player == other]
    latest_demand = share(_utility(own_offers[-1].price, own_value, role)) if own_offers else None
    incoming_share = (
        share(_utility(other_offers[-1].price, own_value, role)) if other_offers else None
    )

    if action_type == "decision":
        last_offer = _mapping(state.get("last_offer"), "last_offer")
        incoming_price = _finite_number(
            last_offer.get("price", last_offer.get("product_price")), "last_offer.price"
        )
        incoming_share = share(_utility(incoming_price, own_value, role))
        if incoming_share + 1e-12 >= 0.45:
            return {"decision": "AcceptOffer"}
        if _should_accept_persistent_positive_stall(
            offers,
            other,
            own_value,
            role,
            config.uncapped_complete_stall_patience,
        ):
            return {"decision": "AcceptOffer"}
        if latest_demand is None:
            target_share = 0.55
        elif latest_demand + 1e-12 < 0.45:
            return None
        else:
            target_share = max(0.45, min(latest_demand, (latest_demand + incoming_share) / 2.0))
        action = _bracket_action(state, role, own_value, surplus, target_share)
        action["decision"] = "RejectOffer"
        return {"decision": action.pop("decision"), **action}

    if latest_demand is None:
        target_share = 0.55
    elif incoming_share is None:
        if latest_demand + 1e-12 < 0.45:
            return None
        target_share = min(latest_demand, 0.55)
    else:
        if latest_demand + 1e-12 < 0.45:
            return None
        target_share = max(0.45, min(latest_demand, (latest_demand + incoming_share) / 2.0))
    return _bracket_action(state, role, own_value, surplus, target_share)


def _bracket_action(
    state: Mapping[str, Any], role: str, own_value: float, surplus: float, share: float
) -> dict[str, Any]:
    price = _price_for_utility(own_value, role, share * surplus)
    action: dict[str, Any] = {"product_price": _clean_number(price)}
    if _messages_allowed(state):
        action["message"] = f"I propose a balanced price of {price:.2f}."
    return action


def _choose_offer(
    state: Mapping[str, Any],
    action_type: str,
    me: str,
    other: str,
    role: str,
    own_value: float,
    offers: list[_Offer],
    config: NegotiationConfig,
) -> _OfferChoice:
    _validate_config(config)
    progress = _progress(state, config.unknown_horizon_scale)
    concession = progress**config.concession_exponent

    other_value_raw = state.get(f"{other}_value")
    complete = other_value_raw is not None
    grid_prior: _Belief | None = None
    if complete:
        other_value = _finite_number(other_value_raw, f"{other}_value")
        seller_value = own_value if role == "seller" else other_value
        buyer_value = other_value if role == "seller" else own_value
        surplus = buyer_value - seller_value
        if surplus <= 0.0:
            return _OfferChoice(0.0, 0.0, 1.0)
        initial = config.complete_initial_surplus_share * surplus
        final = config.complete_final_surplus_share * surplus
        baseline = final + (initial - final) * (1.0 - concession)
        upper_bound = surplus
    else:
        # Competition valuations are F*M with documented finite factor and scale
        # grids. Infer every latent (F, M) pair consistent with our value and
        # marginalize the opponent's factor. The continuous prior is retained
        # only for genuinely out-of-grid payloads and synthetic audit states.
        scale = max(1.0, abs(own_value))
        initial = config.incomplete_initial_margin * scale
        final = config.incomplete_final_margin * scale
        baseline = final + (initial - final) * (1.0 - concession)
        grid_prior = _private_grid_prior(own_value, role, progress, config)
        if grid_prior is not None:
            finite_ceilings = [
                ceiling
                for ceiling in grid_prior.ceilings
                if math.isfinite(ceiling) and ceiling > 0.0
            ]
            upper_bound = max(finite_ceilings, default=0.0)
        else:
            opponent_best = max(
                (
                    _utility(offer.price, own_value, role)
                    for offer in offers
                    if offer.player == other
                ),
                default=0.0,
            )
            upper_bound = max(
                config.private_surplus_span * scale,
                1.20 * max(0.0, opponent_best),
                baseline,
            )

    belief = _threshold_belief(
        state,
        me,
        other,
        role,
        own_value,
        offers,
        baseline,
        upper_bound,
        complete,
        progress,
        grid_prior,
        config,
    )

    candidate_round = int(state.get("round", 1)) + (1 if action_type == "decision" else 0)
    candidate_state = _state_at_round(state, candidate_round)
    candidate_belief = _advance_belief(belief, state, candidate_state, config)
    candidate_progress = _progress(candidate_state, config.unknown_horizon_scale)

    # A concession is a non-increasing demand for our own utility.  Rebuilding
    # this constraint from history makes the strategy safe under concurrency.
    own_utilities = [
        _utility(offer.price, own_value, role) for offer in offers if offer.player == me
    ]
    maximum_candidate = upper_bound
    if own_utilities:
        maximum_candidate = min(maximum_candidate, max(0.0, own_utilities[-1]))

    candidates = _candidate_utilities(
        maximum_candidate,
        baseline,
        offers,
        other,
        role,
        own_value,
        config.candidate_count,
        candidate_belief,
    )

    choices: list[_OfferChoice] = []
    for utility in candidates:
        acceptance = candidate_belief.acceptance_probability(utility)
        continuation = _rejection_continuation_value(
            utility,
            candidate_belief,
            candidate_state,
            maximum_candidate,
            baseline,
            offers,
            other,
            role,
            own_value,
            action_type,
            config,
        )
        expected = acceptance * utility + (1.0 - acceptance) * continuation
        choices.append(_OfferChoice(utility, expected, acceptance))

    # Expected payoff is primary.  On ties, prefer higher acceptance late and a
    # more informative/high-utility probe early; baseline distance is the final
    # deterministic tie-breaker.
    def tie_break(choice: _OfferChoice) -> tuple[float, float, float, float]:
        if candidate_progress >= 0.75 or not _candidate_has_continuation(
            candidate_state, action_type
        ):
            secondary = choice.acceptance_probability
            tertiary = choice.utility
        else:
            secondary = choice.utility
            tertiary = choice.acceptance_probability
        return (
            choice.expected_value,
            secondary,
            tertiary,
            -abs(choice.utility - baseline),
        )

    return max(choices, key=tie_break)


def _private_grid_prior(
    own_value: float,
    role: str,
    progress: float,
    config: NegotiationConfig,
) -> _Belief | None:
    plausible: list[tuple[float, float]] = []
    for scale in config.valuation_scales:
        for own_factor in config.valuation_factors:
            expected = scale * own_factor
            tolerance = config.grid_match_tolerance * max(1.0, abs(own_value), abs(expected))
            if abs(own_value - expected) <= tolerance:
                plausible.append((own_factor, scale))
    if not plausible:
        return None

    fraction = _private_threshold_fraction(progress, config)
    thresholds: list[float] = []
    ceilings: list[float] = []
    for _, scale in plausible:
        for opponent_factor in config.valuation_factors:
            opponent_value = scale * opponent_factor
            surplus = opponent_value - own_value if role == "seller" else own_value - opponent_value
            tolerance = config.grid_match_tolerance * max(1.0, abs(own_value), abs(opponent_value))
            if surplus <= tolerance:
                # No mutually IR positive trade exists in this latent state.
                # -inf ensures it contributes no acceptance probability, even
                # for an offer at our own reservation value.
                thresholds.append(-math.inf)
                ceilings.append(-math.inf)
            else:
                thresholds.append(fraction * surplus)
                ceilings.append(surplus)

    probability = 1.0 / len(thresholds)
    return _Belief(
        tuple(thresholds),
        tuple(ceilings),
        tuple([probability] * len(thresholds)),
        grid_based=True,
    )


def _private_threshold_fraction(progress: float, config: NegotiationConfig) -> float:
    concession = progress**config.concession_exponent
    initial = config.private_initial_threshold_fraction
    return initial + (1.0 - initial) * concession


def _threshold_belief(
    state: Mapping[str, Any],
    me: str,
    other: str,
    role: str,
    own_value: float,
    offers: list[_Offer],
    baseline: float,
    upper_bound: float,
    complete: bool,
    progress: float,
    grid_prior: _Belief | None,
    config: NegotiationConfig,
) -> _Belief:
    count = config.candidate_count
    if grid_prior is not None:
        thresholds = grid_prior.thresholds
        ceilings = grid_prior.ceilings
        weights = list(grid_prior.probabilities)
        grid_based = True
    elif complete:
        thresholds = tuple(upper_bound * index / (count - 1) for index in range(count))
        ceilings = tuple([upper_bound] * count)
        grid_based = False
        # If both parties use the same time-dependent aspiration, their current
        # concession threshold for our utility is surplus minus that aspiration.
        # A uniform component prevents brittle self-play assumptions and covers
        # humans/simple agents that accept any profitable proposal.
        center = _clamp(upper_bound - baseline, 0.0, upper_bound)
        deviation = max(1e-12, config.complete_threshold_std * upper_bound)
        bell = [
            math.exp(-0.5 * ((threshold - center) / deviation) ** 2) for threshold in thresholds
        ]
        bell_total = sum(bell)
        if bell_total == 0.0:
            nearest = min(range(count), key=lambda index: abs(thresholds[index] - center))
            bell[nearest] = 1.0
            bell_total = 1.0
        weights = [
            config.complete_uniform_mixture / count
            + (1.0 - config.complete_uniform_mixture) * density / bell_total
            for density in bell
        ]
    else:
        # Continuous fallback for a genuinely out-of-grid value. Treat its
        # support as economic surplus ceilings and expose only the gradually
        # increasing current fraction, matching the finite-grid model.
        ceilings = tuple(upper_bound * index / (count - 1) for index in range(count))
        fraction = _private_threshold_fraction(progress, config)
        thresholds = tuple(fraction * ceiling for ceiling in ceilings)
        weights = [1.0 / count] * count
        grid_based = True

    opponent_utilities = [
        _utility(offer.price, own_value, role) for offer in offers if offer.player == other
    ]
    if opponent_utilities:
        # Proposing a price is strong (but not infallible) evidence that the
        # opponent would accept terms at least as favorable to them.  Update
        # both the valuation hypothesis and its current strategic threshold;
        # never stretch a hypothesis beyond its economic surplus ceiling.
        revealed_floor = max(0.0, max(opponent_utilities))
        if revealed_floor > 1e-12:
            compatible = tuple(
                math.isfinite(ceiling) and ceiling + 1e-12 >= revealed_floor for ceiling in ceilings
            )
            weights = [
                weight
                * (
                    config.action_evidence_reliability
                    if is_compatible
                    else 1.0 - config.action_evidence_reliability
                )
                for weight, is_compatible in zip(weights, compatible, strict=True)
            ]
            thresholds = tuple(
                max(threshold, revealed_floor) if is_compatible else threshold
                for threshold, is_compatible in zip(thresholds, compatible, strict=True)
            )

        # ANAC-style opponent modeling: a repeated improvement in the
        # opponent's offers is evidence of a concession strategy, not just a
        # static reservation value.  Forecast only a fraction of the observed
        # step, and cap the forecast at each latent state's economic surplus.
        # This keeps the model useful against time-dependent conceders without
        # treating one ambitious counteroffer as a promise to accept it.
        forecast = _robust_opponent_concession_forecast(
            offers,
            other,
            own_value,
            role,
            config.opponent_concession_forecast_weight,
            config.robust_forecast_window,
        )
        if forecast is not None:
            compatible = tuple(
                math.isfinite(ceiling) and ceiling + 1e-12 >= forecast
                for ceiling in ceilings
            )
            thresholds = tuple(
                max(threshold, forecast) if is_compatible else threshold
                for threshold, is_compatible in zip(thresholds, compatible, strict=True)
            )

    own_offers = [offer for offer in offers if offer.player == me]
    if own_offers:
        # In an active game our most recent proposal was rejected.  Allow its
        # latent threshold to drift upward with elapsed time, then treat the
        # rejection as an upper-censoring observation.  We use one rejection to
        # avoid counting correlated repetitions as independent evidence.
        rejected = own_offers[-1]
        rejected_utility = max(0.0, _utility(rejected.price, own_value, role))
        rejected_progress = _progress_at_round(state, rejected.round, config.unknown_horizon_scale)
        drift = (
            config.rejection_drift_fraction * upper_bound * max(0.0, progress - rejected_progress)
        )
        rejected_ceiling = min(upper_bound, rejected_utility + drift)
        weights = _apply_likelihood(
            thresholds,
            weights,
            lambda threshold: threshold <= rejected_ceiling + 1e-12,
            config.rejection_evidence_reliability,
        )

    probabilities = _normalize(weights)
    return _Belief(thresholds, ceilings, tuple(probabilities), grid_based)


def _advance_belief(
    belief: _Belief,
    from_state: Mapping[str, Any],
    to_state: Mapping[str, Any],
    config: NegotiationConfig,
) -> _Belief:
    from_progress = _progress(from_state, config.unknown_horizon_scale)
    to_progress = _progress(to_state, config.unknown_horizon_scale)
    if to_progress <= from_progress:
        return belief

    if belief.grid_based:
        fraction = _private_threshold_fraction(to_progress, config)
        thresholds = tuple(
            fraction * ceiling if math.isfinite(ceiling) else -math.inf
            for ceiling in belief.ceilings
        )
    else:
        # Complete-information behavioral thresholds can relax with time, but
        # never exceed the known economic surplus.
        drift = min(
            1.0,
            config.rejection_drift_fraction * (to_progress - from_progress),
        )
        thresholds = tuple(
            threshold + drift * (ceiling - threshold)
            for threshold, ceiling in zip(belief.thresholds, belief.ceilings, strict=True)
        )
    return _Belief(
        thresholds,
        belief.ceilings,
        belief.probabilities,
        belief.grid_based,
    )


def _rejection_continuation_value(
    utility: float,
    belief: _Belief,
    candidate_state: Mapping[str, Any],
    maximum_candidate: float,
    baseline: float,
    offers: list[_Offer],
    other: str,
    role: str,
    own_value: float,
    action_type: str,
    config: NegotiationConfig,
) -> float:
    if not _candidate_has_continuation(candidate_state, action_type):
        return 0.0

    censored_weights = _apply_likelihood(
        belief.thresholds,
        list(belief.probabilities),
        lambda threshold: threshold < utility - 1e-12,
        config.rejection_evidence_reliability,
    )
    censored = _Belief(
        belief.thresholds,
        belief.ceilings,
        tuple(_normalize(censored_weights)),
        belief.grid_based,
    )
    next_state = _state_at_round(candidate_state, int(candidate_state.get("round", 1)) + 1)
    next_belief = _advance_belief(censored, candidate_state, next_state, config)
    next_maximum = min(maximum_candidate, max(0.0, utility))
    next_candidates = _candidate_utilities(
        next_maximum,
        baseline,
        offers,
        other,
        role,
        own_value,
        config.candidate_count,
        next_belief,
    )
    future_immediate = max(
        (
            candidate * next_belief.acceptance_probability(candidate)
            for candidate in next_candidates
        ),
        default=0.0,
    )
    return config.continuation_survival * future_immediate


def _candidate_has_continuation(candidate_state: Mapping[str, Any], action_type: str) -> bool:
    if action_type not in {"offer", "decision"}:
        return False
    if not _has_known_horizon(candidate_state):
        return True
    return int(candidate_state.get("round", 1)) < int(candidate_state["max_rounds"])


def _should_walk_away_from_stalled_opponent(
    offers: list[_Offer],
    opponent: str,
    own_value: float,
    role: str,
    incoming_utility: float,
    patience: int,
) -> bool:
    """Stop an endless loop when the opponent will not improve a zero deal."""

    if patience < 1 or incoming_utility > 1e-12:
        return False
    opponent_offers = [offer for offer in offers if offer.player == opponent]
    if len(opponent_offers) < patience:
        return False
    utilities = [_utility(offer.price, own_value, role) for offer in opponent_offers[-patience:]]
    tolerance = 1e-12 * max(1.0, abs(own_value))
    return all(
        current <= previous + tolerance
        for previous, current in pairwise(utilities)
    )


def _should_walk_away_after_persistent_losses(
    offers: list[_Offer],
    opponent: str,
    own_value: float,
    role: str,
    patience: int,
) -> bool:
    """Bound an uncapped private-value game that never reaches our reserve.

    Some opponents make asymptotic concessions for the platform's full
    100-round safety cap even when the latent values are incompatible.  Ten
    consecutive opponent offers provide a conservative throughput boundary:
    the archived Granola games contain no positive agreement after remaining
    unprofitable for this long.
    """

    opponent_offers = [offer for offer in offers if offer.player == opponent]
    if len(opponent_offers) < patience:
        return False
    tolerance = 1e-12 * max(1.0, abs(own_value))
    return all(
        _utility(offer.price, own_value, role) <= tolerance
        for offer in opponent_offers[-patience:]
    )


def _should_accept_persistent_positive_stall(
    offers: list[_Offer],
    opponent: str,
    own_value: float,
    role: str,
    patience: int,
) -> bool:
    """Bank an uncapped public offer after a long non-improving sequence."""

    opponent_offers = [offer for offer in offers if offer.player == opponent]
    if len(opponent_offers) < patience:
        return False
    utilities = [
        _utility(offer.price, own_value, role) for offer in opponent_offers[-patience:]
    ]
    tolerance = 1e-12 * max(1.0, abs(own_value), *(abs(value) for value in utilities))
    return utilities[-1] > tolerance and all(
        current <= previous + tolerance
        for previous, current in pairwise(utilities)
    )


def _should_accept_stalled_profitable_offer(
    offers: list[_Offer],
    opponent: str,
    own_value: float,
    role: str,
    incoming_utility: float,
    continuation_value: float,
    patience: int,
    acceptance_fraction: float,
) -> bool:
    """Accept a good but stagnant offer after the opponent stops conceding."""

    if patience < 1 or not 0.0 < acceptance_fraction <= 1.0:
        return False
    if incoming_utility <= 1e-12 or continuation_value <= 1e-12:
        return False
    opponent_offers = [offer for offer in offers if offer.player == opponent]
    if len(opponent_offers) < patience:
        return False
    utilities = [_utility(offer.price, own_value, role) for offer in opponent_offers[-patience:]]
    tolerance = 1e-12 * max(1.0, abs(own_value), abs(incoming_utility))
    stagnant = all(current <= previous + tolerance for previous, current in pairwise(utilities))
    return stagnant and incoming_utility >= acceptance_fraction * continuation_value - tolerance


def _opponent_concession_forecast(
    offers: list[_Offer],
    opponent: str,
    own_value: float,
    role: str,
    weight: float,
) -> float | None:
    """Forecast the opponent's next concession in our utility space.

    An offer is informative only when it improves our utility relative to the
    opponent's previous offer.  We use the latest two distinct observations so
    duplicated ``last_offer`` records cannot manufacture a trend.
    """

    opponent_offers = [offer for offer in offers if offer.player == opponent]
    if len(opponent_offers) < 2 or weight <= 0.0:
        return None
    previous, latest = opponent_offers[-2:]
    previous_utility = _utility(previous.price, own_value, role)
    latest_utility = _utility(latest.price, own_value, role)
    step = latest_utility - previous_utility
    tolerance = 1e-12 * max(1.0, abs(previous_utility), abs(latest_utility))
    if not math.isfinite(step) or step <= tolerance:
        return None
    return max(0.0, latest_utility + weight * step)


def _robust_opponent_concession_forecast(
    offers: list[_Offer],
    opponent: str,
    own_value: float,
    role: str,
    weight: float,
    window: int,
) -> float | None:
    """Forecast a concession from a recent, confidence-weighted slope median."""

    if weight <= 0.0 or window < 2:
        return None
    opponent_offers = [offer for offer in offers if offer.player == opponent][-window:]
    if len(opponent_offers) < 2:
        return None
    utilities = [_utility(offer.price, own_value, role) for offer in opponent_offers]
    tolerance = 1e-12 * max(1.0, abs(own_value), *(abs(value) for value in utilities))
    positive_steps = [
        current - previous
        for previous, current in pairwise(utilities)
        if current - previous > tolerance
    ]
    if not positive_steps:
        return None
    ordered = sorted(positive_steps)
    midpoint = len(ordered) // 2
    median_step = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    # One observed step is a weak signal; additional consistent observations
    # increase confidence, but the forecast never exceeds the latest offer plus
    # the configured fraction of the robust step.
    confidence = min(1.0, len(positive_steps) / 2.0)
    latest_utility = utilities[-1]
    return max(0.0, latest_utility + weight * confidence * median_step)


def _apply_likelihood(
    thresholds: tuple[float, ...],
    weights: list[float],
    predicate: Callable[[float], bool],
    reliability: float,
) -> list[float]:
    return [
        weight * (reliability if predicate(threshold) else 1.0 - reliability)
        for threshold, weight in zip(thresholds, weights, strict=True)
    ]


def _normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [weight / total for weight in weights]


def _candidate_utilities(
    maximum: float,
    baseline: float,
    offers: list[_Offer],
    other: str,
    role: str,
    own_value: float,
    count: int,
    belief: _Belief,
) -> list[float]:
    maximum = max(0.0, maximum)
    candidates = {maximum * index / (count - 1) for index in range(count)}
    candidates.add(_clamp(baseline, 0.0, maximum))
    for offer in offers:
        if offer.player == other:
            candidates.add(_clamp(_utility(offer.price, own_value, role), 0.0, maximum))
    # Exact surplus breakpoints are where acceptance probability changes, so a
    # reference-grid optimum can never be missed by the continuous mesh.
    for breakpoint in (*belief.thresholds, *belief.ceilings):
        if math.isfinite(breakpoint):
            candidates.add(_clamp(breakpoint, 0.0, maximum))
    return sorted(candidates)


def _validate_config(config: NegotiationConfig) -> None:
    if not isinstance(config.candidate_count, int) or isinstance(config.candidate_count, bool):
        raise TypeError("candidate_count must be an integer")
    numeric_values = {
        "complete_initial_surplus_share": config.complete_initial_surplus_share,
        "complete_final_surplus_share": config.complete_final_surplus_share,
        "complete_terminal_own_surplus_share": config.complete_terminal_own_surplus_share,
        "complete_terminal_fairness_share": config.complete_terminal_fairness_share,
        "incomplete_initial_margin": config.incomplete_initial_margin,
        "incomplete_final_margin": config.incomplete_final_margin,
        "concession_exponent": config.concession_exponent,
        "unknown_horizon_scale": config.unknown_horizon_scale,
        "private_surplus_span": config.private_surplus_span,
        "grid_match_tolerance": config.grid_match_tolerance,
        "private_initial_threshold_fraction": config.private_initial_threshold_fraction,
        "complete_uniform_mixture": config.complete_uniform_mixture,
        "complete_threshold_std": config.complete_threshold_std,
        "action_evidence_reliability": config.action_evidence_reliability,
        "opponent_concession_forecast_weight": config.opponent_concession_forecast_weight,
        "rejection_evidence_reliability": config.rejection_evidence_reliability,
        "rejection_drift_fraction": config.rejection_drift_fraction,
        "continuation_survival": config.continuation_survival,
        "certainty_equivalent_discount": config.certainty_equivalent_discount,
        "private_evidence_certainty_discount": config.private_evidence_certainty_discount,
        "private_terminal_opponent_buffer_fraction": (
            config.private_terminal_opponent_buffer_fraction
        ),
        "stalled_positive_acceptance_fraction": config.stalled_positive_acceptance_fraction,
        "best_seen_acceptance_fraction": config.best_seen_acceptance_fraction,
        "reciprocal_concession_weight": config.reciprocal_concession_weight,
    }
    for name, value in numeric_values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    for name, values in (
        ("valuation_factors", config.valuation_factors),
        ("valuation_scales", config.valuation_scales),
    ):
        if not isinstance(values, tuple) or not values:
            raise TypeError(f"{name} must be a nonempty tuple")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0.0
            for value in values
        ):
            raise ValueError(f"{name} must contain positive finite numbers")
        if len({float(value) for value in values}) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    if (
        not isinstance(config.terminal_fairness_opponent_names, tuple)
        or any(
            not isinstance(name, str) or not name.strip()
            for name in config.terminal_fairness_opponent_names
        )
    ):
        raise TypeError("terminal_fairness_opponent_names must be a tuple of nonempty strings")
    shares = (
        config.complete_initial_surplus_share,
        config.complete_final_surplus_share,
        config.complete_terminal_own_surplus_share,
        config.complete_terminal_fairness_share,
    )
    if any(not 0.0 <= share <= 1.0 for share in shares):
        raise ValueError("complete surplus shares must lie in [0, 1]")
    if config.complete_initial_surplus_share < config.complete_final_surplus_share:
        raise ValueError("initial complete-information share must be at least final share")
    if config.incomplete_initial_margin < config.incomplete_final_margin:
        raise ValueError("incomplete margins must be nonnegative and nonincreasing")
    if config.incomplete_initial_margin < 0.0 or config.incomplete_final_margin < 0.0:
        raise ValueError("incomplete margins must be nonnegative")
    if config.concession_exponent <= 0.0:
        raise ValueError("concession_exponent must be positive")
    if config.unknown_horizon_scale <= 0.0:
        raise ValueError("unknown_horizon_scale must be positive")
    if config.private_surplus_span <= 0.0:
        raise ValueError("private_surplus_span must be positive")
    if config.grid_match_tolerance <= 0.0:
        raise ValueError("grid_match_tolerance must be positive")
    if not 0.0 <= config.private_initial_threshold_fraction <= 1.0:
        raise ValueError("private_initial_threshold_fraction must lie in [0, 1]")
    if not 0.0 <= config.private_terminal_opponent_buffer_fraction <= 1.0:
        raise ValueError("private_terminal_opponent_buffer_fraction must lie in [0, 1]")
    if config.candidate_count < 3:
        raise ValueError("candidate_count must be at least 3")
    if not isinstance(config.stalled_opponent_patience, int) or isinstance(
        config.stalled_opponent_patience, bool
    ):
        raise TypeError("stalled_opponent_patience must be an integer")
    if config.stalled_opponent_patience < 1:
        raise ValueError("stalled_opponent_patience must be positive")
    if not isinstance(config.uncapped_private_loss_patience, int) or isinstance(
        config.uncapped_private_loss_patience, bool
    ):
        raise TypeError("uncapped_private_loss_patience must be an integer")
    if config.uncapped_private_loss_patience < 1:
        raise ValueError("uncapped_private_loss_patience must be positive")
    if not isinstance(config.uncapped_complete_stall_patience, int) or isinstance(
        config.uncapped_complete_stall_patience, bool
    ):
        raise TypeError("uncapped_complete_stall_patience must be an integer")
    if config.uncapped_complete_stall_patience < 1:
        raise ValueError("uncapped_complete_stall_patience must be positive")
    if not 0.0 <= config.reciprocal_concession_weight <= 1.0:
        raise ValueError("reciprocal_concession_weight must lie in [0, 1]")
    if not 0.0 < config.stalled_positive_acceptance_fraction <= 1.0:
        raise ValueError("stalled_positive_acceptance_fraction must lie in (0, 1]")
    if not 0.0 < config.best_seen_acceptance_fraction <= 1.0:
        raise ValueError("best_seen_acceptance_fraction must lie in (0, 1]")
    if not isinstance(config.robust_forecast_window, int) or isinstance(
        config.robust_forecast_window, bool
    ):
        raise TypeError("robust_forecast_window must be an integer")
    if config.robust_forecast_window < 2:
        raise ValueError("robust_forecast_window must be at least 2")
    if not 0.0 <= config.complete_uniform_mixture <= 1.0:
        raise ValueError("complete_uniform_mixture must lie in [0, 1]")
    if config.complete_threshold_std <= 0.0:
        raise ValueError("complete_threshold_std must be positive")
    for name, reliability in (
        ("action_evidence_reliability", config.action_evidence_reliability),
        ("rejection_evidence_reliability", config.rejection_evidence_reliability),
    ):
        if not 0.5 <= reliability < 1.0:
            raise ValueError(f"{name} must lie in [0.5, 1)")
    if not 0.0 <= config.opponent_concession_forecast_weight <= 1.0:
        raise ValueError("opponent_concession_forecast_weight must lie in [0, 1]")
    for name, fraction in (
        ("rejection_drift_fraction", config.rejection_drift_fraction),
        ("continuation_survival", config.continuation_survival),
        ("certainty_equivalent_discount", config.certainty_equivalent_discount),
        ("private_evidence_certainty_discount", config.private_evidence_certainty_discount),
    ):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")


def _offers(state: Mapping[str, Any]) -> list[_Offer]:
    """Read primary offers and counteroffers across documented history shapes."""

    extracted: list[_Offer] = []
    history = state.get("history") or []
    if isinstance(history, list):
        for position, raw_record in enumerate(history, start=1):
            if not isinstance(raw_record, Mapping):
                continue
            record_round = _round_or_default(raw_record.get("round"), position)
            primary = raw_record.get("offer")
            if isinstance(primary, Mapping):
                _append_offer(extracted, primary, None, record_round)
            elif "price" in raw_record or "product_price" in raw_record:
                _append_offer(extracted, raw_record, None, record_round)

            counter = raw_record.get("counteroffer")
            counter_player = raw_record.get("decided_by", raw_record.get("current_player"))
            if isinstance(counter, Mapping):
                _append_offer(extracted, counter, counter_player, record_round + 1)
            elif counter is not None:
                synthetic = {"price": counter, "from_player": counter_player}
                _append_offer(extracted, synthetic, counter_player, record_round + 1)

    last_offer = state.get("last_offer")
    if isinstance(last_offer, Mapping):
        _append_offer(
            extracted,
            last_offer,
            None,
            _round_or_default(last_offer.get("round"), _round_or_default(state.get("round"), 1)),
        )

    # ``last_offer`` is commonly also the last history entry.  Exact duplicate
    # observations would create a false zero-time concession signal.
    unique: list[_Offer] = []
    seen: set[tuple[float, str, int]] = set()
    for offer in extracted:
        key = (offer.price, offer.player, offer.round)
        if key not in seen:
            seen.add(key)
            unique.append(offer)
    # Normally chronological already; make posterior censoring deterministic for
    # shuffled fixtures and any future transport ordering changes.
    unique.sort(key=lambda offer: offer.round)
    return unique


def _append_offer(
    destination: list[_Offer],
    raw: Mapping[str, Any],
    fallback_player: Any,
    fallback_round: int,
) -> None:
    player_raw = raw.get("from_player", raw.get("proposer", fallback_player))
    if player_raw not in {"player_1", "player_2"}:
        return
    price_raw = raw.get("price", raw.get("product_price"))
    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        return
    if not math.isfinite(price):
        return
    destination.append(
        _Offer(price, str(player_raw), _round_or_default(raw.get("round"), fallback_round))
    )


def _known_nonpositive_surplus(
    state: Mapping[str, Any], me: str, role: str, own_value: float
) -> bool:
    other_raw = state.get(f"{_other_player(me)}_value")
    if other_raw is None:
        return False
    other_value = _finite_number(other_raw, "opponent value")
    seller_value = own_value if role == "seller" else other_value
    buyer_value = other_value if role == "seller" else own_value
    return buyer_value <= seller_value


def _price_for_utility(own_value: float, role: str, utility: float) -> float:
    return own_value + utility if role == "seller" else own_value - utility


def _utility(price: float, own_value: float, role: str) -> float:
    return price - own_value if role == "seller" else own_value - price


def _offer_action(price: float, role: str, state: Mapping[str, Any]) -> dict[str, Any]:
    action: dict[str, Any] = {"product_price": _clean_number(price)}
    if _messages_allowed(state):
        if role == "seller":
            action["message"] = f"I propose a price of {price:.2f}."
        else:
            action["message"] = f"I offer {price:.2f}."
    return action


def _progress(state: Mapping[str, Any], unknown_scale: float) -> float:
    round_number = max(1, int(state.get("round", 1)))
    if _has_known_horizon(state):
        maximum = int(state["max_rounds"])
        if maximum < 1:
            raise ValueError("max_rounds must be positive")
        if maximum == 1:
            return 1.0
        return _clamp((round_number - 1) / (maximum - 1), 0.0, 1.0)
    if unknown_scale <= 0.0:
        raise ValueError("unknown_horizon_scale must be positive")
    return 1.0 - math.exp(-(round_number - 1) / unknown_scale)


def _progress_at_round(state: Mapping[str, Any], round_number: int, unknown_scale: float) -> float:
    projected = dict(state)
    projected["round"] = round_number
    return _progress(projected, unknown_scale)


def _state_at_round(state: Mapping[str, Any], round_number: int) -> Mapping[str, Any]:
    projected = dict(state)
    projected["round"] = max(1, round_number)
    return projected


def _has_known_horizon(state: Mapping[str, Any]) -> bool:
    known = bool(state.get("horizon_known", state.get("max_rounds") is not None))
    return known and state.get("max_rounds") is not None


def _is_final_round(state: Mapping[str, Any]) -> bool:
    return _has_known_horizon(state) and int(state.get("round", 1)) >= int(state["max_rounds"])


def _messages_allowed(state: Mapping[str, Any]) -> bool:
    return bool(state.get("messages_allowed", False))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _action_type(game: Mapping[str, Any]) -> str:
    actions = _mapping(game.get("valid_actions"), "valid_actions")
    action_type = actions.get("type")
    if not isinstance(action_type, str):
        raise TypeError("valid_actions.type must be present")
    return action_type


def _player(value: Any) -> str:
    if value not in {"player_1", "player_2"}:
        raise ValueError(f"Unknown player identifier: {value!r}")
    return str(value)


def _other_player(player: str) -> str:
    return "player_2" if player == "player_1" else "player_1"


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _round_or_default(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, default)


def _clean_number(value: float) -> Number:
    if not math.isfinite(value):
        raise ValueError("product_price must be finite")
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
