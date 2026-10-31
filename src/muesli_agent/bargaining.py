"""Adaptive policy for GLEE's alternating-offer bargaining game.

The policy combines two deliberately small models rather than assuming that every
opponent is either perfectly rational or a fixed 40-percent threshold bot:

* a complete-information Rubinstein anchor accounts for both players' discount
  factors; and
* a discrete, noisy posterior represents the receiver's minimum acceptable share.

Rejected offers update the posterior.  Opponent counteroffers provide weaker
aspiration evidence and, importantly, can show that a rejection was strategic rather
than a literal threshold observation.  Offer selection compares agreement now with
the discounted value of receiving the opponent's likely counteroffer.  Acceptance
uses the corresponding one-step value of becoming proposer next round.

All evidence comes from ``game_state["history"]``.  There is no process-global state,
so games remain independent under SDK concurrency and actions are reproducible after
a restart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Number = int | float


@dataclass(frozen=True)
class BargainingConfig:
    """Auditable behavioral assumptions used by :func:`bargaining_strategy`."""

    # A threshold grid is more transparent than a fitted black-box accept model.
    threshold_min: float = 0.00
    threshold_max: float = 1.00
    threshold_step: float = 0.01
    hidden_threshold_center: float = 0.42
    agent_threshold_center: float = 0.40
    human_threshold_center: float = 0.45
    threshold_sigma: float = 0.075

    # Identity is disclosed in only half of GLEE games.  When it is visible, use the
    # opponent class to change the mixture between backward induction and inequity
    # aversion.  Hidden games deliberately remain conservative.
    rational_mixture_weight: float = 0.28
    agent_rational_weight: float = 0.58
    human_rational_weight: float = 0.08
    rational_sigma: float = 0.025
    incomplete_rational_sigma: float = 0.14
    rejection_tremble: float = 0.08
    aspiration_evidence_weight: float = 0.22

    min_offer_share: float = 0.00
    minimum_positive_amount: float = 0.01
    disclosed_agent_terminal_floor: float = 0.25
    routine_max_offer_share: float = 0.50
    adaptation_max_offer_share: float = 0.58
    equilibrium_max_offer_share: float = 0.95
    rejection_concession: float = 0.02
    agent_rejection_scale: float = 0.50
    human_rejection_scale: float = 1.50
    contradiction_margin: float = 0.02
    acceptance_slack: float = 0.005
    unknown_opponent_discount: float = 0.90
    counteroffer_reliability: float = 0.88
    finite_lookahead_rounds: int = 6


DEFAULT_BARGAINING_CONFIG = BargainingConfig()


def bargaining_strategy(
    game: Mapping[str, Any],
    config: BargainingConfig = DEFAULT_BARGAINING_CONFIG,
) -> dict[str, Any]:
    """Return a legal, history-adaptive action for a GLEE bargaining payload."""

    state = _state(game)
    action_type = _action_type(game)
    me = _player(game, state)
    money = _finite_float(state.get("money_to_divide"), "money_to_divide")
    if money <= 0.0:
        raise ValueError("money_to_divide must be positive")

    if action_type == "offer":
        receiver_share = _choose_receiver_share(game, state, me, config)
        receiver_amount = _clean_number(money * receiver_share)
        # Derive the second field by subtraction.  This is safer than independently
        # rounding both fields because the API requires an exact sum.
        proposer_amount_raw = money - float(receiver_amount)
        proposer_amount: Number = (
            int(proposer_amount_raw) if proposer_amount_raw.is_integer() else proposer_amount_raw
        )

        if me == "player_1":
            action: dict[str, Any] = {
                "alice_gain": proposer_amount,
                "bob_gain": receiver_amount,
            }
        else:
            action = {
                "alice_gain": receiver_amount,
                "bob_gain": proposer_amount,
            }
        if _messages_allowed(state):
            action["message"] = _offer_message(game, state, me, receiver_share, config)
        return action

    if action_type == "decision":
        offer = state.get("last_offer")
        if not isinstance(offer, Mapping):
            raise TypeError("last_offer must be present for a bargaining decision")
        offered = _gain(offer, me)
        if not math.isfinite(offered):
            raise ValueError("last_offer contains a non-finite gain")

        # With no outside option, every strictly positive terminal agreement beats
        # disagreement.  Rejecting zero is payoff-equivalent and preserves the rule
        # that an agreement must improve our payoff.
        if _is_final_round(state):
            return {"decision": "accept" if offered > 0.0 else "reject"}
        if offered <= 0.0:
            return {"decision": "reject"}

        offered_share = offered / money
        reservation = _continuation_as_next_proposer(game, state, me, config)
        decision = (
            "accept" if offered_share + config.acceptance_slack + 1e-12 >= reservation else "reject"
        )
        return {"decision": decision}

    raise ValueError(f"Unsupported bargaining action type: {action_type!r}")


def _choose_receiver_share(
    game: Mapping[str, Any],
    state: Mapping[str, Any],
    proposer: str,
    config: BargainingConfig,
    *,
    rollout_rounds: int | None = None,
) -> float:
    thresholds, weights = _threshold_posterior(game, state, proposer, config)
    if rollout_rounds is None:
        rollout_rounds = max(0, config.finite_lookahead_rounds)
    continuation = _counteroffer_value(
        game,
        state,
        proposer,
        config,
        rollout_rounds=rollout_rounds,
    )
    final_round = _is_final_round(state)
    if final_round:
        continuation = 0.0

    upper = config.routine_max_offer_share
    credible_rejections = _credible_rejected_shares(state, proposer, config)
    if credible_rejections:
        upper = config.adaptation_max_offer_share

    rational_share = _rational_receiver_share(state, proposer, config)
    if _opponent_type(game) == "agent" and _complete_information(state):
        # Make shares above one half available to the optimizer only where a visible
        # agent's continuation value makes them potentially rational.  They are not
        # a hard concession: the expected-payoff search can still prefer 50%.
        upper = max(
            upper,
            min(
                config.equilibrium_max_offer_share,
                rational_share + 2.0 * config.rational_sigma,
            ),
        )

    money = _finite_float(state.get("money_to_divide"), "money_to_divide")
    minimum_share = max(
        config.min_offer_share,
        min(1.0, config.minimum_positive_amount / money),
    )
    upper = max(upper, minimum_share)
    candidates = [
        share
        for share in _candidate_shares(
            config.min_offer_share,
            upper,
            config.threshold_step,
        )
        if share + 1e-12 >= minimum_share
    ]
    candidates.append(minimum_share)
    candidates = sorted(set(candidates))
    best_share = candidates[0]
    best_value = -math.inf
    for share in candidates:
        acceptance = _cdf(thresholds, weights, share)
        value = acceptance * (1.0 - share) + (1.0 - acceptance) * continuation
        # Stable tie-breaking favors keeping more of the pot.
        if value > best_value + 1e-12:
            best_share = share
            best_value = value

    # When the receiver's visible continuation value is at least one half, an early
    # equal split is a principled insurance policy against costly rejection.  Unlike
    # the old blanket equilibrium floor, this guard does not suppress deadline
    # leverage in other configuration/role cells.
    if (
        _complete_information(state)
        and rational_share >= 0.50
        and not _is_final_round(state)
        and _deadline_pressure(state) < 1.0
    ):
        pressure = _deadline_pressure(state)
        insurance_floor = (1.0 - pressure) * config.routine_max_offer_share + pressure * min(
            best_share, config.routine_max_offer_share
        )
        best_share = max(best_share, insurance_floor)

    if credible_rejections and not _is_final_round(state):
        concession = config.rejection_concession * _rejection_scale(game, config)
        best_share = max(
            best_share,
            min(
                config.adaptation_max_offer_share,
                max(credible_rejections) + concession,
            ),
        )

    if _is_final_round(state) and _opponent_type(game) == "agent":
        # A rational terminal receiver should accept an arbitrarily small positive
        # amount, but the disclosed-agent pool also contains LLM and rule-based
        # inequity-averse policies.  Until live calibration is available, retain
        # substantial proposer leverage without making the brittle 6% model optimum.
        best_share = max(best_share, config.disclosed_agent_terminal_floor)

    return _clamp(best_share, minimum_share, upper)


def _continuation_as_next_proposer(
    game: Mapping[str, Any],
    state: Mapping[str, Any],
    me: str,
    config: BargainingConfig,
) -> float:
    """Current-round value of rejecting and making the next proposal."""

    next_state = dict(state)
    next_state["round"] = max(1, int(state.get("round", 1))) + 1
    next_state["current_player"] = me
    next_state["proposer"] = me
    next_game = dict(game)
    next_game["game_state"] = next_state

    next_round_value = _proposal_value(
        next_game,
        next_state,
        me,
        config,
        rollout_rounds=max(0, config.finite_lookahead_rounds),
    )
    return _own_discount(state, me) * next_round_value


def _proposal_value(
    game: Mapping[str, Any],
    state: Mapping[str, Any],
    proposer: str,
    config: BargainingConfig,
    *,
    rollout_rounds: int,
) -> float:
    """Expected normalized value of a proposal under the bounded rollout."""

    target = _choose_receiver_share(
        game,
        state,
        proposer,
        config,
        rollout_rounds=rollout_rounds,
    )
    thresholds, weights = _threshold_posterior(game, state, proposer, config)
    acceptance = _cdf(thresholds, weights, target)
    after_rejection = 0.0
    if not _is_final_round(state):
        after_rejection = _counteroffer_value(
            game,
            state,
            proposer,
            config,
            rollout_rounds=rollout_rounds,
        )
    return acceptance * (1.0 - target) + (1.0 - acceptance) * after_rejection


def _counteroffer_value(
    game: Mapping[str, Any],
    state: Mapping[str, Any],
    me: str,
    config: BargainingConfig,
    *,
    rollout_rounds: int,
) -> float:
    """Current value after rejection, with a bounded finite-horizon rollout.

    For a known horizon, Granola compares the predicted counteroffer with the value
    of rejecting it and proposing two rounds from now, for at most the configured six
    future rounds.  Unknown horizons and the rollout boundary use the conservative
    one-counteroffer approximation; no claim of full game-tree solution is made.
    """

    if _is_final_round(state):
        return 0.0
    predicted_share = _predicted_counteroffer_share(game, state, me, config)
    effective_offer = config.counteroffer_reliability * predicted_share
    own_discount = _own_discount(state, me)

    next_round = max(1, int(state.get("round", 1))) + 1
    maximum = state.get("max_rounds")
    known = bool(state.get("horizon_known", maximum is not None)) and maximum is not None
    if not known or rollout_rounds < 2 or next_round >= int(maximum):
        return own_discount * effective_offer

    later_state = dict(state)
    later_state["round"] = next_round + 1
    later_state["current_player"] = me
    later_state["proposer"] = me
    later_game = dict(game)
    later_game["game_state"] = later_state
    later_proposal = _proposal_value(
        later_game,
        later_state,
        me,
        config,
        rollout_rounds=rollout_rounds - 2,
    )
    receiver_value_next_round = max(effective_offer, own_discount * later_proposal)
    return own_discount * receiver_value_next_round


def _predicted_counteroffer_share(
    game: Mapping[str, Any],
    state: Mapping[str, Any],
    me: str,
    config: BargainingConfig,
) -> float:
    observed = _offers_to(state, recipient=me, proposer=_other(me))
    if observed:
        prediction = observed[-1]
        if len(observed) >= 2:
            # Only extrapolate concessions in our favor, and damp the last change to
            # avoid one erratic message controlling the policy.
            prediction += 0.5 * max(0.0, observed[-1] - observed[-2])
        return _clamp(prediction, config.min_offer_share, 1.0)

    behavior = _behavioral_center(game, config)
    if not _complete_information(state):
        return behavior
    next_state = dict(state)
    next_state["round"] = max(1, int(state.get("round", 1))) + 1
    rational = _rational_receiver_share(next_state, _other(me), config)
    rational_weight = _rational_weight(game, config)
    prediction = (1.0 - rational_weight) * behavior + rational_weight * rational
    return _clamp(prediction, config.min_offer_share, 1.0)


def _threshold_posterior(
    game: Mapping[str, Any],
    state: Mapping[str, Any],
    proposer: str,
    config: BargainingConfig,
) -> tuple[list[float], list[float]]:
    thresholds = _candidate_shares(
        config.threshold_min,
        config.threshold_max,
        config.threshold_step,
    )
    behavior_center = _behavioral_center(game, config)
    rational_center = _rational_receiver_share(state, proposer, config)
    behavioral_weights: list[float] = []
    for threshold in thresholds:
        weight = _normal_kernel(
            threshold,
            behavior_center,
            config.threshold_sigma,
        )

        for rejected in _rejected_shares(state, proposer):
            if _contradicted_rejection(state, proposer, rejected, config):
                # A counteroffer claiming less for the opponent than we just offered
                # them is evidence of strategic delay, not a hard threshold.
                likelihood = 0.60 if threshold > rejected else 0.40
            else:
                likelihood = (
                    1.0 - config.rejection_tremble
                    if threshold > rejected
                    else config.rejection_tremble
                )
            weight *= likelihood

        for own_claim in _opponent_claims(state, proposer):
            similarity = _normal_kernel(
                threshold,
                own_claim,
                2.0 * config.threshold_sigma,
            )
            # Aspiration and acceptance are correlated but not identical, hence the
            # bounded multiplier rather than treating a claim as direct evidence.
            weight *= (
                1.0
                - config.aspiration_evidence_weight
                + config.aspiration_evidence_weight * similarity
            )
        behavioral_weights.append(weight)

    # Normalize each component before mixing.  This matters at a terminal deadline:
    # a rational threshold is concentrated at zero and must retain its intended
    # mixture mass instead of losing weight at the boundary of the grid.
    behavioral_weights = _normalized(behavioral_weights)
    rational_sigma = (
        config.rational_sigma if _complete_information(state) else config.incomplete_rational_sigma
    )
    rational_weights = _normalized(
        [_normal_kernel(threshold, rational_center, rational_sigma) for threshold in thresholds]
    )
    rational_weight = _rational_weight(game, config)
    weights = [
        (1.0 - rational_weight) * behavioral + rational_weight * rational
        for behavioral, rational in zip(
            behavioral_weights,
            rational_weights,
            strict=True,
        )
    ]

    return thresholds, _normalized(weights)


def _credible_rejected_shares(
    state: Mapping[str, Any],
    proposer: str,
    config: BargainingConfig,
) -> list[float]:
    return [
        share
        for share in _rejected_shares(state, proposer)
        if not _contradicted_rejection(state, proposer, share, config)
    ]


def _contradicted_rejection(
    state: Mapping[str, Any],
    proposer: str,
    rejected_share: float,
    config: BargainingConfig,
) -> bool:
    money = _finite_float(state.get("money_to_divide"), "money_to_divide")
    history = _history(state)
    for index, record in enumerate(history):
        offer = record.get("offer")
        if not isinstance(offer, Mapping):
            continue
        record_proposer = record.get("proposer", offer.get("proposer"))
        if record_proposer != proposer or _decision_name(record.get("decision")) != "reject":
            continue
        try:
            share = _gain(offer, _other(proposer)) / money
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isclose(share, rejected_share, abs_tol=1e-9):
            continue
        for later_record in history[index + 1 :]:
            later_offer = later_record.get("offer")
            if not isinstance(later_offer, Mapping):
                continue
            later_proposer = later_record.get("proposer", later_offer.get("proposer"))
            if later_proposer != _other(proposer):
                continue
            try:
                later_claim = _gain(later_offer, _other(proposer)) / money
            except (KeyError, TypeError, ValueError):
                continue
            if later_claim + config.contradiction_margin < rejected_share:
                return True
    return False


def _rejected_shares(state: Mapping[str, Any], proposer: str) -> list[float]:
    money = _finite_float(state.get("money_to_divide"), "money_to_divide")
    receiver = _other(proposer)
    rejected: list[float] = []
    for record in _history(state):
        offer = record.get("offer")
        if not isinstance(offer, Mapping):
            continue
        record_proposer = record.get("proposer", offer.get("proposer"))
        if record_proposer != proposer or _decision_name(record.get("decision")) != "reject":
            continue
        try:
            share = _gain(offer, receiver) / money
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(share) and 0.0 <= share <= 1.0:
            rejected.append(share)
    return rejected


def _opponent_claims(state: Mapping[str, Any], proposer: str) -> list[float]:
    """Shares the current opponent requested for itself in previous proposals."""

    opponent = _other(proposer)
    return _offers_to(state, recipient=opponent, proposer=opponent)


def _offers_to(
    state: Mapping[str, Any],
    recipient: str,
    proposer: str,
) -> list[float]:
    money = _finite_float(state.get("money_to_divide"), "money_to_divide")
    shares: list[float] = []
    for record_proposer, offer in _observed_offers(state):
        if record_proposer != proposer:
            continue
        try:
            share = _gain(offer, recipient) / money
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(share) and 0.0 <= share <= 1.0:
            shares.append(share)
    return shares


def _observed_offers(
    state: Mapping[str, Any],
) -> list[tuple[str | None, Mapping[str, Any]]]:
    """History offers plus a current ``last_offer``, without double counting it."""

    observed: list[tuple[str | None, Mapping[str, Any]]] = []
    history = _history(state)
    for record in history:
        offer = record.get("offer")
        if not isinstance(offer, Mapping):
            continue
        record_proposer = record.get("proposer", offer.get("proposer"))
        proposer = str(record_proposer) if record_proposer is not None else None
        observed.append((proposer, offer))

    last_offer = state.get("last_offer")
    if not isinstance(last_offer, Mapping):
        return observed
    last_proposer_raw = last_offer.get("proposer")
    if last_proposer_raw is None and state.get("current_player") in {"player_1", "player_2"}:
        # In a decision phase the current player is the receiver.
        last_proposer_raw = _other(str(state["current_player"]))
    last_proposer = str(last_proposer_raw) if last_proposer_raw is not None else None

    # The SDK may put proposer/round on the history record but inline them in
    # ``last_offer``. Compare the exact contextual offer, not only dict shape.
    if not any(_same_offer_observation(record, last_proposer, last_offer) for record in history):
        observed.append((last_proposer, last_offer))
    return observed


def _same_offer_observation(
    record: Mapping[str, Any],
    last_proposer: str | None,
    last_offer: Mapping[str, Any],
) -> bool:
    offer = record.get("offer")
    if not isinstance(offer, Mapping):
        return False
    record_proposer_raw = record.get("proposer", offer.get("proposer"))
    record_proposer = str(record_proposer_raw) if record_proposer_raw is not None else None
    if record_proposer != last_proposer:
        return False

    record_round = record.get("round", offer.get("round"))
    last_round = last_offer.get("round")
    if record_round is not None and last_round is not None and record_round != last_round:
        return False
    if offer.get("message") != last_offer.get("message"):
        return False
    try:
        return _gain(offer, "player_1") == _gain(last_offer, "player_1") and _gain(
            offer, "player_2"
        ) == _gain(last_offer, "player_2")
    except (KeyError, TypeError, ValueError):
        return offer == last_offer


def _stationary_receiver_share(
    state: Mapping[str, Any],
    proposer: str,
    config: BargainingConfig,
) -> float:
    delta_proposer = _discount(state, proposer, fallback=1.0)
    delta_receiver = _discount(
        state,
        _other(proposer),
        fallback=config.unknown_opponent_discount,
    )
    denominator = 1.0 - delta_proposer * delta_receiver
    if denominator <= 1e-12:
        # At delta=(1,1), the stationary formula is indeterminate.  Equal division
        # is the neutral limit used by the behavioral game implementations.
        return 0.50
    return _clamp(
        delta_receiver * (1.0 - delta_proposer) / denominator,
        0.0,
        1.0,
    )


def _rational_receiver_share(
    state: Mapping[str, Any],
    proposer: str,
    config: BargainingConfig,
) -> float:
    """Receiver's subgame-perfect reservation share for the visible horizon.

    A capped final-round receiver has no continuation value.  Working backward,
    rejection makes today's receiver tomorrow's proposer, so their current nominal
    reservation is their discount factor times the share that proposer would retain.
    With no known cap, the recursion's stationary limit is Rubinstein's share.
    """

    if not bool(state.get("horizon_known", state.get("max_rounds") is not None)):
        return _stationary_receiver_share(state, proposer, config)
    maximum_raw = state.get("max_rounds")
    if maximum_raw is None:
        return _stationary_receiver_share(state, proposer, config)
    current = max(1, int(state.get("round", 1)))
    remaining = max(1, int(maximum_raw) - current + 1)

    # Iterate backward rather than recurse so an unusual large cap cannot exhaust
    # Python's call stack.  ``next_receiver_share`` starts at the terminal offer.
    next_proposer = proposer if (remaining - 1) % 2 == 0 else _other(proposer)
    next_receiver_share = 0.0
    for _ in range(remaining - 1):
        current_proposer = _other(next_proposer)
        current_receiver = next_proposer
        receiver_discount = _discount(
            state,
            current_receiver,
            fallback=config.unknown_opponent_discount,
        )
        next_receiver_share = receiver_discount * (1.0 - next_receiver_share)
        next_proposer = current_proposer
    return _clamp(next_receiver_share, 0.0, 1.0)


def _deadline_pressure(state: Mapping[str, Any]) -> float:
    """Fade an infinite-horizon anchor only during the final quarter."""

    if not bool(state.get("horizon_known", state.get("max_rounds") is not None)):
        return 0.0
    maximum_raw = state.get("max_rounds")
    if maximum_raw is None:
        return 0.0
    maximum = max(1, int(maximum_raw))
    current = max(1, int(state.get("round", 1)))
    if maximum <= 1:
        return 1.0
    progress = _clamp((current - 1) / (maximum - 1), 0.0, 1.0)
    return _clamp((progress - 0.75) / 0.25, 0.0, 1.0)


def _is_final_round(state: Mapping[str, Any]) -> bool:
    if not bool(state.get("horizon_known", state.get("max_rounds") is not None)):
        return False
    maximum = state.get("max_rounds")
    if maximum is None:
        return False
    return int(state.get("round", 1)) >= int(maximum)


def _complete_information(state: Mapping[str, Any]) -> bool:
    if not bool(state.get("complete_information", False)):
        return False
    return state.get("delta_1") is not None and state.get("delta_2") is not None


def _behavioral_center(game: Mapping[str, Any], config: BargainingConfig) -> float:
    opponent_type = _opponent_type(game)
    if opponent_type == "human":
        return config.human_threshold_center
    if opponent_type == "agent":
        return config.agent_threshold_center
    return config.hidden_threshold_center


def _opponent_type(game: Mapping[str, Any]) -> str:
    opponent = game.get("opponent")
    opponent_type = opponent.get("type") if isinstance(opponent, Mapping) else "hidden"
    return str(opponent_type).lower()


def _rational_weight(game: Mapping[str, Any], config: BargainingConfig) -> float:
    opponent_type = _opponent_type(game)
    if opponent_type == "agent":
        return _clamp(config.agent_rational_weight, 0.0, 1.0)
    if opponent_type == "human":
        return _clamp(config.human_rational_weight, 0.0, 1.0)
    return _clamp(config.rational_mixture_weight, 0.0, 1.0)


def _rejection_scale(game: Mapping[str, Any], config: BargainingConfig) -> float:
    opponent_type = _opponent_type(game)
    if opponent_type == "agent":
        return max(0.0, config.agent_rejection_scale)
    if opponent_type == "human":
        return max(0.0, config.human_rejection_scale)
    return 1.0


def _offer_message(
    game: Mapping[str, Any],
    state: Mapping[str, Any],
    proposer: str,
    receiver_share: float,
    config: BargainingConfig,
) -> str:
    """Frame the numerical action using only true, payoff-relevant facts."""

    money = _finite_float(state.get("money_to_divide"), "money_to_divide")
    amount = money * receiver_share
    allocation = f"{receiver_share:.1%} of the pot ({amount:.2f})"
    if _is_final_round(state):
        return (
            f"This is the final round. Accepting gives you {allocation} now; "
            "rejection gives both of us zero."
        )

    rejected = _credible_rejected_shares(state, proposer, config)
    if rejected:
        return (
            f"I updated the proposal after your rejection: you receive {allocation}. "
            "Agreement now avoids another round of discounting."
        )

    if _complete_information(state):
        receiver_discount = _discount(state, _other(proposer), fallback=1.0)
        return (
            f"This proposal gives you {allocation} now. Your next-round value is "
            f"multiplied by {receiver_discount:.2f}, so prompt agreement preserves value."
        )
    return (
        f"This proposal gives you {allocation} now; agreement avoids the loss from "
        "another discounted round."
    )


def _discount(state: Mapping[str, Any], player: str, fallback: float) -> float:
    key = "delta_1" if player == "player_1" else "delta_2"
    raw = state.get(key)
    if raw is None:
        return fallback
    value = _finite_float(raw, key)
    return _clamp(value, 0.0, 1.0)


def _own_discount(state: Mapping[str, Any], me: str) -> float:
    return _discount(state, me, fallback=0.95)


def _gain(offer: Mapping[str, Any], player: str) -> float:
    aliases = {
        "player_1": ("player_1_gain", "alice_gain"),
        "player_2": ("player_2_gain", "bob_gain"),
    }
    for key in aliases[player]:
        if key in offer:
            return float(offer[key])
    raise KeyError(f"Offer has no gain field for {player}")


def _decision_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("decision")
    return str(value or "").strip().lower()


def _history(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = state.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return []
    return [record for record in history if isinstance(record, Mapping)]


def _candidate_shares(lower: float, upper: float, step: float) -> list[float]:
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("share bounds must satisfy 0 <= lower <= upper <= 1")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("threshold_step must be positive and finite")
    count = math.floor((upper - lower) / step + 1e-9)
    candidates = [round(lower + index * step, 10) for index in range(count + 1)]
    if candidates[-1] < upper - 1e-10:
        candidates.append(upper)
    return candidates


def _cdf(thresholds: list[float], weights: list[float], share: float) -> float:
    return sum(
        weight
        for threshold, weight in zip(thresholds, weights, strict=True)
        if threshold <= share + 1e-12
    )


def _normal_kernel(value: float, center: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-9)
    z = (value - center) / sigma
    return math.exp(-0.5 * z * z)


def _normalized(weights: list[float]) -> list[float]:
    total = sum(weights)
    if not weights:
        raise ValueError("cannot normalize an empty distribution")
    if not math.isfinite(total) or total <= 0.0:
        uniform = 1.0 / len(weights)
        return [uniform] * len(weights)
    return [weight / total for weight in weights]


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


def _player(game: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    player = game.get("your_player", state.get("current_player"))
    if player not in {"player_1", "player_2"}:
        raise ValueError(f"Unknown player identifier: {player!r}")
    return str(player)


def _other(player: str) -> str:
    return "player_2" if player == "player_1" else "player_1"


def _messages_allowed(state: Mapping[str, Any]) -> bool:
    return bool(state.get("messages_allowed", False))


def _finite_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _clean_number(value: float) -> Number:
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
