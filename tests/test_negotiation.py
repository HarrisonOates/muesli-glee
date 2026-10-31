from __future__ import annotations

import math
import random

import pytest

from muesli_agent.negotiation import (
    NegotiationConfig,
    _offers,
    _opponent_concession_forecast,
    _robust_opponent_concession_forecast,
    _should_accept_stalled_profitable_offer,
    _should_walk_away_after_persistent_losses,
    _should_walk_away_from_stalled_opponent,
    negotiation_strategy,
)


def game(
    action_type: str = "offer",
    player: str = "player_1",
    complete: bool = True,
    **updates,
):
    state = {
        "phase": action_type,
        "current_player": player,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "round": 1,
        "max_rounds": 5,
        "horizon_known": True,
        "last_offer": None,
        "history": [],
        "messages_allowed": True,
        "complete_information": complete,
    }
    if complete or player == "player_1":
        state["player_1_value"] = 40
    if complete or player == "player_2":
        state["player_2_value"] = 100
    state.update(updates)
    return {
        "game_id": "negotiation-focused-test",
        "game_family": "negotiation",
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def own_utility(price: float, player: str, value: float) -> float:
    return price - value if player == "player_1" else value - price


def test_complete_information_search_is_role_symmetric():
    seller = negotiation_strategy(game(player="player_1"))["product_price"]
    buyer = negotiation_strategy(game(player="player_2"))["product_price"]

    assert seller == pytest.approx(74.5)
    assert buyer == pytest.approx(65.5)
    assert seller - 40 == pytest.approx(100 - buyer)
    # The old 72%-surplus Boulware aspiration (83.2) is now a prior parameter,
    # not a price mechanically sent despite low immediate acceptance odds.
    assert seller < 83.2


@pytest.mark.parametrize("player,value", [("player_1", 40), ("player_2", 100)])
def test_rejected_offers_produce_monotone_concessions(player, value):
    prices = []
    history = []
    opponent = "player_2" if player == "player_1" else "player_1"
    for round_number in range(1, 6):
        price = negotiation_strategy(game(player=player, round=round_number, history=history))[
            "product_price"
        ]
        prices.append(price)
        history.append(
            {
                "round": round_number,
                "offer": {
                    "price": price,
                    "from_player": player,
                    "round": round_number,
                },
                "decision": "RejectOffer",
                "decided_by": opponent,
            }
        )
    utilities = [own_utility(price, player, value) for price in prices]

    assert utilities == sorted(utilities, reverse=True)
    assert utilities[-1] >= 0


def test_single_round_complete_information_uses_final_proposer_leverage():
    seller = negotiation_strategy(game(max_rounds=1, player="player_1"))["product_price"]
    buyer = negotiation_strategy(game(max_rounds=1, player="player_2"))["product_price"]
    assert seller == pytest.approx(99.4)
    assert buyer == pytest.approx(40.6)
    assert seller - 40 == pytest.approx(100 - buyer)


def test_terminal_leverage_can_replay_legacy_share_for_comparison():
    legacy = NegotiationConfig(complete_terminal_own_surplus_share=0.475)

    assert negotiation_strategy(game(max_rounds=1), legacy)["product_price"] == 68.5


def test_terminal_leverage_never_reverses_an_earlier_concession():
    history = [
        {
            "round": 1,
            "offer": {"price": 73, "from_player": "player_1", "round": 1},
            "decision": "RejectOffer",
            "decided_by": "player_2",
        }
    ]
    action = negotiation_strategy(game(round=5, max_rounds=5, history=history))

    assert action["product_price"] == 73


def test_terminal_leverage_uses_fair_split_for_observed_punisher():
    payload = game(max_rounds=1, player="player_1")
    payload["opponent"] = {"type": "agent", "name": " Punish "}

    action = negotiation_strategy(payload)

    assert action["product_price"] == 70


def test_complete_information_nontrade_uses_reservation_and_walks_away():
    initial = negotiation_strategy(game(player_1_value=110, player_2_value=100))
    assert initial["product_price"] == 110

    response = negotiation_strategy(
        game(
            action_type="decision",
            player_1_value=110,
            player_2_value=100,
            last_offer={"price": 105, "from_player": "player_2", "round": 1},
        )
    )
    assert response == {"decision": "WalkAway"}


@pytest.mark.parametrize(
    ("player", "value", "price"),
    [("player_1", 40, 40), ("player_1", 40, 80), ("player_2", 100, 100), ("player_2", 100, 60)],
)
def test_final_round_accepts_every_weakly_profitable_offer(player, value, price):
    action = negotiation_strategy(
        game(
            action_type="decision",
            player=player,
            round=5,
            last_offer={
                "price": price,
                "from_player": "player_2" if player == "player_1" else "player_1",
            },
        )
    )
    assert action == {"decision": "AcceptOffer"}
    assert own_utility(price, player, value) >= 0


@pytest.mark.parametrize(("player", "price"), [("player_1", 39.999), ("player_2", 100.001)])
def test_final_round_loss_rejection_never_contains_a_counteroffer(player, price):
    action = negotiation_strategy(
        game(
            action_type="decision",
            player=player,
            round=5,
            last_offer={
                "price": price,
                "from_player": "player_2" if player == "player_1" else "player_1",
            },
        )
    )
    assert action == {"decision": "RejectOffer"}


def test_nonfinal_rejection_has_a_legal_ir_counteroffer_and_optional_message():
    with_message = negotiation_strategy(
        game(
            action_type="decision",
            last_offer={"price": 41, "from_player": "player_2"},
        )
    )
    without_message = negotiation_strategy(
        game(
            action_type="decision",
            messages_allowed=False,
            last_offer={"price": 41, "from_player": "player_2"},
        )
    )

    assert with_message["decision"] == "RejectOffer"
    assert with_message["product_price"] >= 40
    assert len(with_message["message"]) < 2000
    assert set(without_message) == {"decision", "product_price"}


def test_unknown_horizon_prior_adapts_gradually_without_a_history():
    seller_prices = []
    buyer_prices = []
    for round_number in (1, 10, 40):
        seller_game = game(
            player="player_1", complete=False, round=round_number, horizon_known=False
        )
        seller_game["game_state"].pop("max_rounds")
        buyer_game = game(
            player="player_2", complete=False, round=round_number, horizon_known=False
        )
        buyer_game["game_state"].pop("max_rounds")
        seller_prices.append(negotiation_strategy(seller_game)["product_price"])
        buyer_prices.append(negotiation_strategy(buyer_game)["product_price"])

    # With no deadline or discounting, elapsed time increases the prior chance
    # that the opponent will accept a demanding offer.  Crucially this is a
    # gradual, deterministic update rather than a frozen round-one action.
    assert seller_prices == sorted(seller_prices)
    assert buyer_prices == sorted(buyer_prices, reverse=True)
    assert len(set(seller_prices)) == len(seller_prices)
    assert len(set(buyer_prices)) == len(buyer_prices)
    assert seller_prices[1] > 40
    assert buyer_prices[1] < 100


def test_own_counteroffers_never_reverse_even_if_forecast_becomes_optimistic():
    state_history = [
        {
            "round": 1,
            "offer": {"price": 130, "from_player": "player_1"},
            "decision": "RejectOffer",
            "counteroffer": {"price": 70, "from_player": "player_2"},
            "decided_by": "player_2",
        },
        {
            "round": 3,
            "offer": {"price": 105, "from_player": "player_1"},
            "decision": "RejectOffer",
            "counteroffer": 75,
            "decided_by": "player_2",
        },
    ]
    action = negotiation_strategy(
        game(
            action_type="decision",
            player="player_2",
            complete=False,
            round=4,
            history=state_history,
            last_offer={"price": 95, "from_player": "player_1", "round": 4},
        )
    )
    assert action["decision"] == "RejectOffer"
    assert 75 <= action["product_price"] <= 100


def test_duplicate_last_offer_does_not_change_posterior():
    history = [
        {"round": 1, "offer": {"price": 120, "from_player": "player_1", "round": 1}},
        {"round": 3, "offer": {"price": 105, "from_player": "player_1", "round": 3}},
    ]
    common = {
        "player": "player_2",
        "complete": False,
        "round": 3,
        "history": history,
        "last_offer": {"price": 105, "from_player": "player_1", "round": 3},
    }
    duplicated = negotiation_strategy(game(**common))["product_price"]
    history_without_last = history[:-1]
    separate = negotiation_strategy(game(**(common | {"history": history_without_last})))[
        "product_price"
    ]
    assert duplicated == separate


def test_live_regression_accepts_profitable_unknown_horizon_opening():
    # In Granola's smoke run a buyer worth 150 received an opening ask of 86.
    action = negotiation_strategy(
        game(
            action_type="decision",
            player="player_2",
            complete=False,
            player_2_value=150,
            horizon_known=False,
            last_offer={"price": 86, "from_player": "player_1", "round": 1},
        )
    )
    assert action == {"decision": "AcceptOffer"}


def test_uncapped_complete_stall_banks_positive_offer_at_patience_boundary():
    history = [
        {
            "round": 2 * index + 1,
            "offer": {
                "price": 99.99,
                "from_player": "player_1",
                "round": 2 * index + 1,
            },
        }
        for index in range(10)
    ]
    common = {
        "action_type": "decision",
        "player": "player_2",
        "horizon_known": False,
        "max_rounds": None,
    }

    before_boundary = negotiation_strategy(
        game(
            **common,
            round=17,
            history=history[:9],
            last_offer=history[8]["offer"],
        )
    )
    at_boundary = negotiation_strategy(
        game(
            **common,
            round=19,
            history=history,
            last_offer=history[9]["offer"],
        )
    )

    assert before_boundary["decision"] == "RejectOffer"
    assert at_boundary == {"decision": "AcceptOffer"}


def test_live_regression_waits_after_large_early_known_horizon_concession():
    # The observed seller moved from 1.429m to 926,656 by round 3 of 10.  A
    # positive payoff is not by itself enough reason to stop this early.
    history = [
        {
            "round": 1,
            "offer": {"price": 1_429_000, "from_player": "player_1"},
            "decision": "RejectOffer",
            "counteroffer": {"price": 700_000, "from_player": "player_2"},
            "decided_by": "player_2",
        }
    ]
    action = negotiation_strategy(
        game(
            action_type="decision",
            player="player_2",
            complete=False,
            player_2_value=1_000_000,
            round=3,
            max_rounds=10,
            history=history,
            last_offer={"price": 926_656, "from_player": "player_1", "round": 3},
        )
    )
    assert action["decision"] == "RejectOffer"
    assert 700_000 <= action["product_price"] < 926_656


def test_private_evidence_discount_promotes_millet_acceptance_cell():
    history = [
        {
            "round": 1,
            "offer": {"price": 60, "from_player": "player_2"},
            "decision": "RejectOffer",
            "decided_by": "player_1",
        }
    ]
    payload = game(
        action_type="decision",
        player="player_2",
        complete=False,
        round=3,
        max_rounds=10,
        history=history,
        last_offer={"price": 90, "from_player": "player_1", "round": 3},
    )

    legacy = NegotiationConfig(private_evidence_certainty_discount=0.85)
    assert negotiation_strategy(payload, legacy)["decision"] == "RejectOffer"
    assert negotiation_strategy(payload) == {"decision": "AcceptOffer"}


@pytest.mark.parametrize(
    "updates",
    [
        {"history": []},
        {"horizon_known": False, "max_rounds": None},
        {"complete": True},
    ],
    ids=["no-evidence", "uncapped", "public-values"],
)
def test_private_evidence_discount_does_not_leak_outside_validated_cell(updates):
    arguments = {
        "action_type": "decision",
        "player": "player_2",
        "complete": False,
        "round": 3,
        "max_rounds": 10,
        "history": [
            {
                "round": 1,
                "offer": {"price": 60, "from_player": "player_2"},
                "decision": "RejectOffer",
                "decided_by": "player_1",
            }
        ],
        "last_offer": {"price": 90, "from_player": "player_1", "round": 3},
    }
    arguments.update(updates)
    payload = game(**arguments)
    legacy = NegotiationConfig(private_evidence_certainty_discount=0.85)

    assert negotiation_strategy(payload) == negotiation_strategy(payload, legacy)


def test_private_value_initial_offer_is_scale_invariant():
    prices = [
        negotiation_strategy(game(complete=False, player_1_value=value))["product_price"]
        for value in (80, 8_000, 800_000)
    ]
    ratios = [price / value for price, value in zip(prices, (80, 8_000, 800_000))]
    assert ratios[0] == pytest.approx(ratios[1]) == pytest.approx(ratios[2])
    assert prices[1] > 10_400  # the prior 1.30x ask accepted in the smoke run


def _enumerated_grid_optimum(own_value: float, role: str, scale: float) -> float:
    """Independent final-round reference over the documented four factors."""

    surpluses = []
    for factor in (0.8, 1.0, 1.2, 1.5):
        opponent_value = factor * scale
        surplus = opponent_value - own_value if role == "seller" else own_value - opponent_value
        surpluses.append(surplus if surplus > 1e-12 else None)
    candidates = {0.0, *(surplus for surplus in surpluses if surplus is not None)}

    def score(utility):
        accepted = sum(surplus is not None and surplus + 1e-12 >= utility for surplus in surpluses)
        probability = accepted / len(surpluses)
        return utility * probability, probability, utility

    utility = max(candidates, key=score)
    return own_value + utility if role == "seller" else own_value - utility


@pytest.mark.parametrize(
    ("player", "own_value", "expected_price"),
    [
        ("player_2", 100, 80.2),
        ("player_2", 120, 100.2),
        ("player_1", 120, 149.7),
        ("player_2", 150, 100.5),
    ],
)
def test_private_final_offer_buffers_the_enumerated_factor_grid(player, own_value, expected_price):
    role = "seller" if player == "player_1" else "buyer"
    value_field = f"{player}_value"
    action = negotiation_strategy(
        game(complete=False, player=player, max_rounds=1, **{value_field: own_value})
    )
    reference = _enumerated_grid_optimum(own_value, role, 100)
    legacy = negotiation_strategy(
        game(complete=False, player=player, max_rounds=1, **{value_field: own_value}),
        NegotiationConfig(private_terminal_opponent_buffer_fraction=0.0),
    )

    assert legacy["product_price"] == pytest.approx(reference)
    assert action["product_price"] == pytest.approx(expected_price)
    if player == "player_2" and own_value in {100, 120}:
        assert action["product_price"] >= 80
    if player == "player_1":
        assert action["product_price"] <= 150


def test_single_round_candidate_search_improves_modeled_private_value_payoff():
    # The private-value prior is deliberately uniform over a scale-free surplus
    # span.  On that prior, explicit candidate search should outperform sending
    # the old final Boulware margin mechanically.
    price = negotiation_strategy(game(complete=False, player_1_value=40, max_rounds=1))[
        "product_price"
    ]
    searched_utility = price - 40
    old_boulware_utility = 0.12 * 40
    thresholds = [32 * index / 10_000 for index in range(10_001)]

    def modeled_payoff(utility):
        return sum(utility if threshold >= utility else 0.0 for threshold in thresholds) / len(
            thresholds
        )

    assert searched_utility == pytest.approx(15.84)
    assert modeled_payoff(searched_utility) > 1.8 * modeled_payoff(old_boulware_utility)


def test_rejection_censors_belief_and_improves_offline_counter_value():
    no_evidence_price = negotiation_strategy(
        game(
            player="player_2",
            complete=False,
            player_2_value=40,
            round=3,
            max_rounds=10,
        )
    )["product_price"]
    rejected_history = [
        {
            "round": 1,
            "offer": {"price": no_evidence_price, "from_player": "player_2"},
            "decision": "RejectOffer",
            "decided_by": "player_1",
        }
    ]
    updated_price = negotiation_strategy(
        game(
            player="player_2",
            complete=False,
            player_2_value=40,
            round=3,
            max_rounds=10,
            history=rejected_history,
        )
    )["product_price"]
    old_utility = 40 - no_evidence_price
    updated_utility = 40 - updated_price
    progress = (3 - 1) / (10 - 1)
    threshold_fraction = 0.70 + 0.30 * progress**3
    thresholds = [32 * threshold_fraction * index / 10_000 for index in range(10_001)]

    def modeled_payoff(utility):
        return sum(utility if threshold >= utility else 0.0 for threshold in thresholds) / len(
            thresholds
        )

    assert updated_price > no_evidence_price
    assert modeled_payoff(updated_utility) > modeled_payoff(old_utility)


def test_opponent_concession_forecast_uses_recent_improvement_only():
    offers = _offers(
        {
            "history": [
                {"round": 1, "offer": {"price": 80, "from_player": "player_1"}},
                {"round": 2, "offer": {"price": 70, "from_player": "player_1"}},
            ]
        }
    )
    # Buyer utility rises from 20 to 30; a half-step forecast is 35.
    assert _opponent_concession_forecast(offers, "player_1", 100, "buyer", 0.5) == pytest.approx(35)
    assert _opponent_concession_forecast(offers[:1], "player_1", 100, "buyer", 0.5) is None


def test_robust_opponent_forecast_uses_median_recent_concession_step():
    offers = _offers(
        {
            "history": [
                {"round": 1, "offer": {"price": 90, "from_player": "player_1"}},
                {"round": 2, "offer": {"price": 80, "from_player": "player_1"}},
                {"round": 3, "offer": {"price": 75, "from_player": "player_1"}},
                {"round": 4, "offer": {"price": 74, "from_player": "player_1"}},
            ]
        }
    )
    # Buyer utilities are 10, 20, 25, 26; the median positive step is 5.
    assert _robust_opponent_concession_forecast(
        offers, "player_1", 100, "buyer", 0.5, 4
    ) == pytest.approx(28.5)


def test_robust_opponent_forecast_ignores_one_step_without_a_recent_window():
    offers = _offers(
        {"history": [{"round": 1, "offer": {"price": 80, "from_player": "player_1"}}]}
    )
    assert _robust_opponent_concession_forecast(offers, "player_1", 100, "buyer", 0.5, 4) is None


def test_repeated_reservation_offer_triggers_walkaway_instead_of_infinite_countering():
    offers = _offers(
        {
            "history": [
                {"round": 1, "offer": {"price": 150, "from_player": "player_1"}},
                {"round": 2, "offer": {"price": 150, "from_player": "player_1"}},
                {"round": 3, "offer": {"price": 150, "from_player": "player_1"}},
            ],
            "last_offer": {"price": 150, "from_player": "player_1", "round": 3},
        }
    )
    assert _should_walk_away_from_stalled_opponent(offers, "player_1", 150, "buyer", 0, 3)


def test_uncapped_private_loss_patience_bounds_asymptotic_concessions():
    history = [
        {
            "round": 2 * index + 1,
            "offer": {
                "price": 220 - 10 * index,
                "from_player": "player_1",
                "round": 2 * index + 1,
            },
        }
        for index in range(10)
    ]
    common = {
        "action_type": "decision",
        "player": "player_2",
        "complete": False,
        "horizon_known": False,
        "max_rounds": None,
    }

    before_boundary = negotiation_strategy(
        game(
            **common,
            round=17,
            history=history[:9],
            last_offer=history[8]["offer"],
        )
    )
    at_boundary = negotiation_strategy(
        game(
            **common,
            round=19,
            history=history,
            last_offer=history[9]["offer"],
        )
    )

    assert before_boundary["decision"] == "RejectOffer"
    assert at_boundary == {"decision": "WalkAway"}


def test_uncapped_private_loss_patience_does_not_trigger_on_a_profitable_offer():
    history = [
        {
            "round": 2 * index + 1,
            "offer": {
                "price": 220 - 10 * index,
                "from_player": "player_1",
                "round": 2 * index + 1,
            },
        }
        for index in range(9)
    ]
    profitable = {"price": 110, "from_player": "player_1", "round": 19}
    history.append({"round": 19, "offer": profitable})

    offers = _offers({"history": history, "last_offer": profitable})

    assert not _should_walk_away_after_persistent_losses(
        offers,
        "player_1",
        own_value=120,
        role="buyer",
        patience=10,
    )


def test_repeated_profitable_offer_is_accepted_when_continuation_is_only_marginally_better():
    offers = _offers(
        {
            "history": [
                {"round": 1, "offer": {"price": 80, "from_player": "player_1"}},
                {"round": 2, "offer": {"price": 80, "from_player": "player_1"}},
                {"round": 3, "offer": {"price": 80, "from_player": "player_1"}},
            ]
        }
    )
    assert _should_accept_stalled_profitable_offer(
        offers,
        "player_1",
        100,
        "buyer",
        incoming_utility=20,
        continuation_value=22,
        patience=3,
        acceptance_fraction=0.85,
    )


def test_stalled_positive_acceptance_does_not_accept_a_loss_or_zero_deal():
    offers = _offers(
        {
            "history": [
                {"round": 1, "offer": {"price": 100, "from_player": "player_1"}},
                {"round": 2, "offer": {"price": 100, "from_player": "player_1"}},
                {"round": 3, "offer": {"price": 100, "from_player": "player_1"}},
            ]
        }
    )
    assert not _should_accept_stalled_profitable_offer(
        offers, "player_1", 100, "buyer", 0, 1, 3, 0.85
    )


def test_opponent_concession_model_can_target_a_conceding_seller():
    history = [
        {"round": 1, "offer": {"price": 107.47328517690156, "from_player": "player_1"}},
        {"round": 2, "offer": {"price": 91.18451012920849, "from_player": "player_1"}},
    ]
    common = {
        "action_type": "offer",
        "player": "player_2",
        "complete": False,
        "player_2_value": 120,
        "round": 3,
        "max_rounds": 10,
        "history": history,
    }
    modeled = negotiation_strategy(game(**common))
    unmodeled = negotiation_strategy(
        game(**common), NegotiationConfig(opponent_concession_forecast_weight=0.0)
    )
    assert modeled["product_price"] < unmodeled["product_price"]
    assert modeled["product_price"] <= 120


def test_opponent_offer_updates_revealed_concession_floor():
    baseline = negotiation_strategy(
        game(player="player_2", complete=False, round=3, max_rounds=10)
    )["product_price"]
    history = [{"round": 1, "offer": {"price": 80, "from_player": "player_1"}}]
    informed = negotiation_strategy(
        game(
            player="player_2",
            complete=False,
            round=3,
            max_rounds=10,
            history=history,
        )
    )["product_price"]
    assert informed < baseline
    assert 100 - informed >= 20  # never discard the already revealed utility


def test_impossible_offer_does_not_create_out_of_grid_acceptance_mass():
    baseline = negotiation_strategy(
        game(player="player_2", complete=False, round=3, max_rounds=10)
    )["product_price"]
    # A seller on the M=100 grid cannot rationally offer 60, below every
    # possible cost. Treat it as noisy action evidence, not a new valuation.
    history = [{"round": 1, "offer": {"price": 60, "from_player": "player_1"}}]
    informed = negotiation_strategy(
        game(
            player="player_2",
            complete=False,
            round=3,
            max_rounds=10,
            history=history,
        )
    )["product_price"]
    assert informed == baseline


def test_penultimate_counter_has_no_fictitious_post_deadline_continuation():
    def counter(round_number, survival):
        return negotiation_strategy(
            game(
                action_type="decision",
                player="player_2",
                complete=False,
                player_2_value=150,
                round=round_number,
                max_rounds=10,
                messages_allowed=False,
                last_offer={
                    "price": 151,
                    "from_player": "player_1",
                    "round": round_number,
                },
            ),
            NegotiationConfig(continuation_survival=survival),
        )

    # At round eight, the counter can be rejected and followed by another
    # action, so its Bellman value legitimately depends on survival.
    assert counter(8, 0.0) != counter(8, 1.0)
    # A round-nine decision produces the round-ten counter. There is no action
    # after a rejection, hence no continuation term to tune.
    assert (
        counter(9, 0.0)
        == counter(9, 1.0)
        == {
            "decision": "RejectOffer",
            "product_price": 100,
        }
    )


def test_history_is_sorted_and_implicit_counter_advances_one_round():
    chronological = [
        {"round": 2, "offer": {"price": 90, "from_player": "player_1"}},
        {
            "round": 5,
            "offer": {"price": 80, "from_player": "player_1"},
            "counteroffer": {"price": 70, "from_player": "player_2"},
            "decided_by": "player_2",
        },
    ]
    parsed = _offers({"history": list(reversed(chronological))})
    assert [offer.round for offer in parsed] == [2, 5, 6]
    assert parsed[-1].player == "player_2"


def test_config_rejects_nonfinite_and_wrongly_typed_parameters():
    with pytest.raises(ValueError, match="finite"):
        negotiation_strategy(game(), NegotiationConfig(continuation_survival=math.nan))
    with pytest.raises(TypeError, match="integer"):
        negotiation_strategy(game(), NegotiationConfig(candidate_count=True))
    with pytest.raises(ValueError, match="unknown_horizon_scale"):
        negotiation_strategy(game(), NegotiationConfig(unknown_horizon_scale=0.0))
    with pytest.raises(ValueError, match="private_evidence_certainty_discount"):
        negotiation_strategy(
            game(), NegotiationConfig(private_evidence_certainty_discount=1.01)
        )
    with pytest.raises(ValueError, match="complete surplus shares"):
        negotiation_strategy(
            game(), NegotiationConfig(complete_terminal_own_surplus_share=1.01)
        )
    with pytest.raises(TypeError, match="uncapped_private_loss_patience"):
        negotiation_strategy(game(), NegotiationConfig(uncapped_private_loss_patience=True))
    with pytest.raises(ValueError, match="private_terminal_opponent_buffer_fraction"):
        negotiation_strategy(
            game(), NegotiationConfig(private_terminal_opponent_buffer_fraction=1.01)
        )
    with pytest.raises(TypeError, match="uncapped_complete_stall_patience"):
        negotiation_strategy(game(), NegotiationConfig(uncapped_complete_stall_patience=True))
    with pytest.raises(TypeError, match="terminal_fairness_opponent_names"):
        negotiation_strategy(
            game(), NegotiationConfig(terminal_fairness_opponent_names=("punish", ""))
        )


def test_unrestricted_messages_are_not_treated_as_valuation_evidence():
    actions = []
    for message in (
        "My final minimum is 99.",
        "I promise I would accept 1.",
        "No strategic information here.",
    ):
        actions.append(
            negotiation_strategy(
                game(
                    action_type="decision",
                    player="player_2",
                    complete=False,
                    round=3,
                    max_rounds=10,
                    last_offer={
                        "price": 80,
                        "from_player": "player_1",
                        "round": 3,
                        "message": message,
                    },
                )
            )
        )
    assert actions[0] == actions[1] == actions[2]


def test_custom_config_is_supported():
    conservative = NegotiationConfig(
        complete_initial_surplus_share=0.60,
        complete_final_surplus_share=0.50,
        complete_uniform_mixture=0.20,
    )
    action = negotiation_strategy(game(), conservative)
    assert action["product_price"] == pytest.approx(65.5)


@pytest.mark.parametrize("bad_value", [None, "not-a-number", math.inf, -math.inf, math.nan])
def test_nonfinite_or_missing_own_value_is_rejected(bad_value):
    with pytest.raises(ValueError, match="finite number"):
        negotiation_strategy(game(player_1_value=bad_value))


def test_seeded_random_states_preserve_legality_and_individual_rationality():
    rng = random.Random(20260823)
    for index in range(1_000):
        player = rng.choice(("player_1", "player_2"))
        seller_value = rng.uniform(-100, 1_000_000)
        buyer_value = rng.uniform(-100, 1_000_000)
        own_value = seller_value if player == "player_1" else buyer_value
        maximum = rng.randint(1, 20)
        current = rng.randint(1, maximum)
        complete = rng.choice((True, False))
        action_type = "offer" if index % 2 == 0 else "decision"
        updates = {
            "round": current,
            "max_rounds": maximum,
            "messages_allowed": rng.choice((True, False)),
        }
        if complete or player == "player_1":
            updates["player_1_value"] = seller_value
        if complete or player == "player_2":
            updates["player_2_value"] = buyer_value
        if action_type == "decision":
            updates["last_offer"] = {
                "price": rng.uniform(-1_000, 1_001_000),
                "from_player": "player_2" if player == "player_1" else "player_1",
                "round": current,
            }
        candidate = negotiation_strategy(
            game(action_type=action_type, player=player, complete=complete, **updates)
        )

        if action_type == "offer":
            assert math.isfinite(float(candidate["product_price"]))
            assert own_utility(candidate["product_price"], player, own_value) >= -1e-6
            if not updates["messages_allowed"]:
                assert "message" not in candidate
        elif candidate["decision"] == "AcceptOffer":
            assert own_utility(updates["last_offer"]["price"], player, own_value) >= -1e-6
        elif current == maximum:
            assert candidate == {"decision": "RejectOffer"}
        elif candidate["decision"] == "RejectOffer":
            assert "product_price" in candidate
            assert own_utility(candidate["product_price"], player, own_value) >= -1e-6
