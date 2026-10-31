from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

import muesli_agent.challenger as challenger_module
from muesli_agent.bargaining import bargaining_strategy
from muesli_agent.challenger import (
    DEFAULT_CHALLENGER_CONFIG,
    ChallengerConfig,
    challenger_strategy,
    millet_strategy,
)
from muesli_agent.negotiation import negotiation_strategy
from muesli_agent.persuasion import persuasion_strategy


def bargaining_game(action_type: str = "decision", player: str = "player_1", **updates) -> dict:
    state = {
        "phase": action_type,
        "current_player": player,
        "money_to_divide": 10_000,
        "round": 3,
        "max_rounds": None,
        "horizon_known": False,
        "complete_information": False,
        "delta_1": 0.9,
        "delta_2": 0.9,
        "last_offer": {
            "alice_gain": 4_000,
            "bob_gain": 6_000,
            "proposer": "player_2" if player == "player_1" else "player_1",
            "round": 3,
        },
        "history": [],
        "messages_allowed": False,
    }
    state.update(updates)
    return {
        "game_id": "challenger-bargaining",
        "game_family": "bargaining",
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def negotiation_game(action_type: str = "offer", player: str = "player_1", **updates) -> dict:
    state = {
        "phase": action_type,
        "current_player": player,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_1_value": 40,
        "player_2_value": 100,
        "round": 1,
        "horizon_known": False,
        "complete_information": True,
        "last_offer": None,
        "history": [],
        "messages_allowed": False,
    }
    state.update(updates)
    return {
        "game_id": "challenger-negotiation",
        "game_family": "negotiation",
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def persuasion_game(action_type: str, **updates) -> dict:
    player = "player_1" if action_type.startswith("seller") else "player_2"
    state = {
        "phase": action_type,
        "current_player": player,
        "round": 1,
        "total_rounds": 20,
        "p": 0.5,
        "v": 200,
        "u": 0,
        "product_price": 100,
        "current_quality": "high",
        "seller_message": "yes",
        "seller_message_type": "text",
        "history": [],
    }
    state.update(updates)
    return {
        "game_id": "challenger-persuasion",
        "game_family": "persuasion",
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def negative_exposures(count: int) -> list[dict]:
    return [
        {
            "round": round_number,
            "seller_message": "It is serviceable — nothing fancy. Worth it.",
            "buyer_decision": "no",
            "bought": False,
        }
        for round_number in range(1, count + 1)
    ]


def test_dispatcher_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown game family"):
        challenger_strategy({"game_family": "auction"})


def test_discount_latch_accepts_repeated_positive_offer_base_rejects():
    history = [
        {
            "round": 1,
            "proposer": "player_2",
            "offer": {"player_1_gain": 4_000, "player_2_gain": 6_000},
            "decision": "reject",
        }
    ]
    game = bargaining_game(history=history)

    assert bargaining_strategy(game) == {"decision": "reject"}
    assert challenger_strategy(game) == {"decision": "accept"}


def test_discount_latch_is_role_symmetric_and_reads_aliases_and_decider():
    history = [
        {
            "round": 1,
            "offer": {"alice_gain": 6_000, "bob_gain": 4_000},
            "decided_by": "player_2",
            "decision": {"decision": "reject"},
        }
    ]
    game = bargaining_game(
        player="player_2",
        history=history,
        last_offer={
            "player_1_gain": 6_000,
            "player_2_gain": 4_000,
            "proposer": "player_1",
            "round": 3,
        },
    )
    assert challenger_strategy(game) == {"decision": "accept"}


@pytest.mark.parametrize(
    "updates",
    [
        {"history": []},
        {
            "delta_1": 1.0,
            "history": [
                {
                    "round": 1,
                    "proposer": "player_2",
                    "offer": {"alice_gain": 4_000, "bob_gain": 6_000},
                    "decision": "reject",
                }
            ],
        },
        {
            "round": 5,
            "max_rounds": 5,
            "horizon_known": True,
            "history": [
                {
                    "round": 1,
                    "proposer": "player_2",
                    "offer": {"alice_gain": 4_000, "bob_gain": 6_000},
                    "decision": "reject",
                }
            ],
        },
        {
            "history": [
                {
                    "round": 1,
                    "proposer": "player_2",
                    "offer": {"alice_gain": 4_001, "bob_gain": 5_999},
                    "decision": "reject",
                }
            ],
        },
    ],
)
def test_discount_latch_delegates_outside_exact_nonfinal_repeat(updates):
    game = bargaining_game(**updates)
    assert challenger_strategy(game) == bargaining_strategy(game)


def test_discount_latch_does_not_treat_current_history_duplicate_as_repeat():
    current = {
        "alice_gain": 4_000,
        "bob_gain": 6_000,
        "proposer": "player_2",
        "round": 3,
    }
    game = bargaining_game(
        last_offer=current,
        history=[{"round": 3, "proposer": "player_2", "offer": dict(current)}],
    )
    assert challenger_strategy(game) == bargaining_strategy(game)


def test_discount_latch_requires_a_permanent_exact_opponent_offer():
    offer = {"alice_gain": 4_000, "bob_gain": 6_000, "proposer": "player_2"}
    game = bargaining_game(
        last_offer={**offer, "round": 3},
        history=[
            {"round": 1, "offer": dict(offer), "decision": "reject"},
            {"round": 2, "offer": dict(offer), "decision": "reject"},
        ],
    )
    assert challenger_strategy(game) == {"decision": "accept"}


def test_discount_latch_ignores_repeat_then_large_concession():
    old_offer = {"alice_gain": 1_000, "bob_gain": 9_000, "proposer": "player_2"}
    game = bargaining_game(
        last_offer={"alice_gain": 7_000, "bob_gain": 3_000, "proposer": "player_2", "round": 3},
        history=[
            {"round": 1, "offer": dict(old_offer), "decision": "reject"},
            {"round": 2, "offer": dict(old_offer), "decision": "reject"},
        ],
    )
    assert challenger_strategy(game) == bargaining_strategy(game)


def test_discount_latch_ignores_strictly_worsening_opponent_offers():
    game = bargaining_game(
        history=[
            {
                "round": 1,
                "offer": {"alice_gain": 6_000, "bob_gain": 4_000, "proposer": "player_2"},
                "decision": "reject",
            },
            {
                "round": 2,
                "offer": {"alice_gain": 5_000, "bob_gain": 5_000, "proposer": "player_2"},
                "decision": "reject",
            },
        ]
    )
    assert challenger_strategy(game) == bargaining_strategy(game)


def test_discount_latch_preserves_both_roles_for_complete_exact_allocations():
    offer = {"player_1_gain": 6_000, "player_2_gain": 4_000, "proposer": "player_1"}
    game = bargaining_game(
        player="player_2",
        last_offer={**offer, "round": 3},
        history=[{"round": 1, "offer": offer, "decision": "reject"}],
    )
    assert challenger_strategy(game) == {"decision": "accept"}


@pytest.mark.parametrize(
    "updates",
    [
        {
            "last_offer": {
                "alice_gain": 4_000,
                "bob_gain": 6_000,
                "proposer": "player_2",
                "round": 3,
            },
            "history": [
                {
                    "round": 1,
                    "proposer": "player_1",
                    "offer": {"alice_gain": 4_000, "bob_gain": 6_000},
                },
                {
                    "round": 2,
                    "proposer": "player_2",
                    "offer": {"alice_gain": 4_000, "bob_gain": 6_001},
                },
            ],
        },
        {
            "last_offer": {
                "alice_gain": 4_000,
                "bob_gain": 6_000,
                "proposer": "player_1",
                "round": 3,
            },
            "history": [
                {
                    "round": 1,
                    "proposer": "player_2",
                    "offer": {"alice_gain": 4_000, "bob_gain": 6_000},
                }
            ],
        },
    ],
    ids=["conflicting-history-allocation", "current-not-opponent-offer"],
)
def test_discount_latch_ignores_malformed_or_ambiguous_offer_metadata(updates):
    game = bargaining_game(**updates)
    assert challenger_strategy(game) == bargaining_strategy(game)


def test_bargaining_offer_phase_is_byte_for_byte_base_policy():
    game = bargaining_game(
        action_type="offer",
        last_offer=None,
        history=[
            {
                "round": 1,
                "proposer": "player_2",
                "offer": {"alice_gain": 4_000, "bob_gain": 6_000},
                "decision": "reject",
            }
        ],
    )
    assert challenger_strategy(game) == bargaining_strategy(game)


@pytest.mark.parametrize(("player", "expected_price"), [("player_1", 73), ("player_2", 67)])
def test_nash_floor_initial_offer_claims_55_percent_symmetrically(player, expected_price):
    game = negotiation_game(player=player)
    assert challenger_strategy(game)["product_price"] == expected_price
    # Granola now owns this validated no-deadline public-information module.
    assert negotiation_strategy(game)["product_price"] == expected_price


@pytest.mark.parametrize(("player", "incoming_price"), [("player_1", 67), ("player_2", 73)])
def test_nash_floor_accepts_exactly_45_percent_for_both_roles(player, incoming_price):
    other = "player_2" if player == "player_1" else "player_1"
    game = negotiation_game(
        action_type="decision",
        player=player,
        last_offer={"price": incoming_price, "from_player": other},
    )
    assert challenger_strategy(game) == {"decision": "AcceptOffer"}


@pytest.mark.parametrize(
    ("player", "own_price", "incoming_price", "expected_counter"),
    [
        ("player_1", 73, 64, 68.5),
        ("player_2", 67, 76, 71.5),
    ],
)
def test_nash_bracket_midpoint_is_symmetric_monotone_and_ir(
    player, own_price, incoming_price, expected_counter
):
    other = "player_2" if player == "player_1" else "player_1"
    history = [
        {
            "round": 1,
            "offer": {"price": own_price, "from_player": player},
            "decision": "RejectOffer",
            "decided_by": other,
        }
    ]
    game = negotiation_game(
        action_type="decision",
        player=player,
        round=2,
        history=history,
        last_offer={"product_price": incoming_price, "proposer": other, "round": 2},
    )
    action = challenger_strategy(game)

    assert action == {"decision": "RejectOffer", "product_price": expected_counter}
    assert 40 <= action["product_price"] <= 100
    if player == "player_1":
        assert action["product_price"] <= own_price
    else:
        assert action["product_price"] >= own_price


def test_nash_bracket_without_prior_demand_counters_at_55_percent():
    game = negotiation_game(
        action_type="decision",
        last_offer={"price": 61, "from_player": "player_2"},
    )
    assert challenger_strategy(game) == {"decision": "RejectOffer", "product_price": 73}


def test_nash_bracket_reads_own_demand_from_nested_counteroffer_history():
    history = [
        {
            "round": 1,
            "offer": {"price": 58, "from_player": "player_2"},
            "decision": "RejectOffer",
            "counteroffer": 73,
            "decided_by": "player_1",
        }
    ]
    game = negotiation_game(
        action_type="decision",
        round=3,
        history=history,
        last_offer={"price": 64, "from_player": "player_2", "round": 3},
    )
    assert challenger_strategy(game) == {"decision": "RejectOffer", "product_price": 68.5}


@pytest.mark.parametrize(
    "updates",
    [
        {"horizon_known": True, "max_rounds": 5},
        {"horizon_known": False, "max_rounds": 5},
        {"complete_information": False, "player_2_value": None},
        {"player_1_value": 110, "player_2_value": 100},
    ],
)
def test_nash_floor_delegates_private_known_cap_and_nonpositive_cells(updates):
    game = negotiation_game(**updates)
    assert challenger_strategy(game) == negotiation_strategy(game)


def test_millet_private_acceptance_experiment_is_now_owned_by_granola():
    game = negotiation_game(
        action_type="decision",
        player="player_2",
        complete_information=False,
        player_1_value=None,
        round=3,
        horizon_known=True,
        max_rounds=10,
        history=[
            {
                "round": 1,
                "offer": {"price": 60, "from_player": "player_2"},
                "decision": "RejectOffer",
                "decided_by": "player_1",
            }
        ],
        last_offer={"price": 90, "from_player": "player_1", "round": 3},
    )

    assert negotiation_strategy(game) == {"decision": "AcceptOffer"}
    assert millet_strategy(game) == negotiation_strategy(game)


def test_known_final_negotiation_preserves_base_legality_without_counteroffer():
    game = negotiation_game(
        action_type="decision",
        round=5,
        horizon_known=True,
        max_rounds=5,
        last_offer={"price": 39, "from_player": "player_2"},
    )
    assert challenger_strategy(game) == {"decision": "RejectOffer"}


def test_negotiation_messages_have_documented_shapes():
    offer = challenger_strategy(negotiation_game(messages_allowed=True))
    counter = challenger_strategy(
        negotiation_game(
            action_type="decision",
            messages_allowed=True,
            last_offer={"price": 61, "from_player": "player_2"},
        )
    )
    assert set(offer) == {"product_price", "message"}
    assert set(counter) == {"decision", "product_price", "message"}
    assert len(offer["message"]) < 2_000
    assert len(counter["message"]) < 2_000


def test_text_seller_uses_explicit_high_grade_template():
    game = persuasion_game("seller_message", current_quality="high")
    assert challenger_strategy(game) == {
        "message": DEFAULT_CHALLENGER_CONFIG.persuasion_high_template
    }
    assert "high-grade" in challenger_strategy(game)["message"].lower()


def test_text_seller_pooled_and_unpooled_low_templates_follow_base_choice():
    pooled = persuasion_game("seller_message", current_quality="low", round=20)
    unpooled = persuasion_game(
        "seller_message",
        current_quality="low",
        round=1,
        product_price=200,
    )
    assert challenger_strategy(pooled) == {
        "message": DEFAULT_CHALLENGER_CONFIG.persuasion_pooled_low_template
    }
    assert challenger_strategy(unpooled) == {
        "message": DEFAULT_CHALLENGER_CONFIG.persuasion_pass_template
    }


def test_text_seller_consults_base_pooling_on_exact_original_payload(monkeypatch):
    game = persuasion_game(
        "seller_message",
        current_quality="low",
        history=[{"seller_message": "This is high-grade. Worth it."}],
    )
    seen = []

    def fake_base(payload):
        seen.append(payload)
        return {"message": "I highly recommend buying this product."}

    monkeypatch.setattr(challenger_module, "persuasion_strategy", fake_base)
    action = challenger_strategy(game)

    assert seen == [game]
    assert seen[0] is game
    assert action == {"message": DEFAULT_CHALLENGER_CONFIG.persuasion_pooled_low_template}


def test_binary_seller_delegates_base_policy():
    game = persuasion_game(
        "seller_recommendation", seller_message_type="binary", current_quality="low", round=20
    )
    assert challenger_strategy(game) == persuasion_strategy(game)


@pytest.mark.parametrize("variant", ["high-grade", "HIGH GRADE", "high‑grade", "high—grade"])
def test_text_buyer_canonicalizes_high_grade_variants_after_two_negative_exposures(variant):
    history = negative_exposures(2)
    game = persuasion_game(
        "buyer_decision",
        round=3,
        seller_message=f"A {variant} product. Worth it.",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "no"}
    assert challenger_strategy(game) == {"decision": "yes"}


@pytest.mark.parametrize(
    "message",
    [
        "This is serviceable. Worth it.",
        "Nothing fancy, but it does the job. Worth it.",
        {"message": "SERVICEABLE — NOTHING FANCY. Worth it."},
    ],
)
def test_text_buyer_canonicalizes_low_templates_as_negative(message):
    game = persuasion_game("buyer_decision", seller_message=message)
    assert challenger_strategy(game) == {"decision": "no"}


def test_exact_guard_requires_two_negative_exposures_before_positive_purchase():
    one_negative = persuasion_game(
        "buyer_decision",
        round=2,
        seller_message="This is high-grade. Worth it.",
        history=negative_exposures(1),
    )
    two_negative = persuasion_game(
        "buyer_decision",
        round=3,
        seller_message="This is high-grade. Worth it.",
        history=negative_exposures(2),
    )
    assert challenger_strategy(one_negative) == {"decision": "no"}
    assert challenger_strategy(two_negative) == {"decision": "yes"}


def test_exact_guard_stops_after_first_purchased_positive_reveals_low():
    history = [
        *negative_exposures(2),
        {
            "round": 3,
            "seller_message": "This is high-grade. Worth it.",
            "buyer_decision": "yes",
            "bought": True,
            "quality": "low",
        },
    ]
    game = persuasion_game(
        "buyer_decision",
        round=4,
        seller_message="Another HIGH GRADE unit. Worth it.",
        history=history,
    )
    assert challenger_strategy(game) == {"decision": "no"}


def test_binary_exact_break_even_guard_uses_same_exposure_rule():
    one_negative = persuasion_game(
        "buyer_decision",
        seller_message_type="binary",
        seller_message="yes",
        history=[{"seller_message": "no", "buyer_decision": "no"}],
    )
    two_negative = persuasion_game(
        "buyer_decision",
        seller_message_type="binary",
        seller_message="yes",
        history=[
            {"seller_message": "no", "buyer_decision": "no"},
            {"seller_message": "no", "buyer_decision": "no"},
        ],
    )
    assert challenger_strategy(one_negative) == {"decision": "no"}
    assert challenger_strategy(two_negative) == {"decision": "yes"}


def test_persuasion_outside_exact_break_even_delegates_after_text_normalization():
    game = persuasion_game(
        "buyer_decision",
        product_price=90,
        seller_message="ordinary unfamiliar prose",
    )
    assert challenger_strategy(game) == persuasion_strategy(game)


def test_live_style_template_replay_buys_only_high_after_learning_separation():
    messages = [
        "This is serviceable — nothing fancy, but useful. Worth it.",
        "Nothing fancy today. Worth it.",
        "This is a high-grade product. Worth it.",
        "A HIGH GRADE product. Worth it.",
    ]
    history: list[dict] = []
    decisions = []
    for round_number, message in enumerate(messages, start=1):
        game = persuasion_game(
            "buyer_decision",
            round=round_number,
            seller_message=message,
            history=copy.deepcopy(history),
        )
        decision = challenger_strategy(game)["decision"]
        decisions.append(decision)
        history.append(
            {
                "round": round_number,
                "seller_message": message,
                "buyer_decision": decision,
                "bought": decision == "yes",
                "quality": "low" if round_number <= 2 else "high",
            }
        )
    assert decisions == ["no", "no", "yes", "yes"]


def test_challenger_never_mutates_payload_and_is_retry_stable_under_concurrency():
    game = persuasion_game(
        "buyer_decision",
        round=3,
        seller_message={"message": "A high‑grade unit. Worth it."},
        history=negative_exposures(2),
    )
    before = copy.deepcopy(game)
    expected = challenger_strategy(game)
    with ThreadPoolExecutor(max_workers=8) as executor:
        actions = list(executor.map(lambda _: challenger_strategy(game), range(100)))

    assert game == before
    assert actions == [expected] * 100


def test_challenger_config_is_frozen_for_reproducible_ablations():
    with pytest.raises(AttributeError):
        DEFAULT_CHALLENGER_CONFIG.negotiation_initial_share = 0.6
    custom = ChallengerConfig(negotiation_initial_share=0.6)
    assert custom.negotiation_initial_share == 0.6
