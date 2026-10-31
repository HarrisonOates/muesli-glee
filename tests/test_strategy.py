from __future__ import annotations

import pytest

from muesli_agent.strategy import (
    bargaining_strategy,
    negotiation_strategy,
    persuasion_strategy,
    strategy,
)


def bargaining_game(action_type: str = "offer", player: str = "player_1", **updates):
    state = {
        "phase": action_type,
        "current_player": player,
        "round": 1,
        "max_rounds": 5,
        "horizon_known": True,
        "money_to_divide": 1000,
        "delta_1": 0.9,
        "delta_2": 0.9,
        "last_offer": None,
        "history": [],
        "messages_allowed": True,
        "complete_information": True,
    }
    state.update(updates)
    return {
        "game_id": "bargaining-test",
        "game_family": "bargaining",
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def negotiation_game(
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
        "player_1_value": 40,
        "round": 1,
        "max_rounds": 5,
        "horizon_known": True,
        "last_offer": None,
        "history": [],
        "messages_allowed": True,
        "complete_information": complete,
    }
    if complete or player == "player_2":
        state["player_2_value"] = 100
    state.update(updates)
    return {
        "game_id": "negotiation-test",
        "game_family": "negotiation",
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def persuasion_game(action_type: str, **updates):
    player = "player_1" if action_type.startswith("seller") else "player_2"
    state = {
        "phase": action_type,
        "current_player": player,
        "round": 1,
        "total_rounds": 10,
        "horizon_known": True,
        "p": 0.4,
        "v": 2.0,
        "u": 0.0,
        "product_price": 1.0,
        "current_quality": "high",
        "seller_message": "yes",
        "history": [],
    }
    state.update(updates)
    return {
        "game_id": "persuasion-test",
        "game_family": "persuasion",
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def test_dispatches_all_families():
    assert set(strategy(bargaining_game())) >= {"alice_gain", "bob_gain"}
    assert "product_price" in strategy(negotiation_game())
    assert strategy(persuasion_game("seller_recommendation")) == {"decision": "yes"}


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="Unknown game family"):
        strategy({"game_family": "chess"})


@pytest.mark.parametrize("player", ["player_1", "player_2"])
def test_bargaining_offer_is_feasible_for_both_roles(player):
    action = bargaining_strategy(bargaining_game(player=player))
    assert action["alice_gain"] + action["bob_gain"] == 1000
    assert 0 <= action["alice_gain"] <= 1000
    assert 0 <= action["bob_gain"] <= 1000
    assert len(action["message"]) < 2000


def test_bargaining_omits_message_when_language_disabled():
    action = bargaining_strategy(bargaining_game(messages_allowed=False))
    assert "message" not in action


def test_bargaining_acceptance_uses_player_gain_aliases():
    game = bargaining_game(
        action_type="decision",
        player="player_2",
        last_offer={"player_1_gain": 400, "player_2_gain": 600},
    )
    assert bargaining_strategy(game) == {"decision": "accept"}


@pytest.mark.parametrize("amount,expected", [(1, "accept"), (0, "reject")])
def test_bargaining_final_round_prefers_positive_agreement(amount, expected):
    game = bargaining_game(
        action_type="decision",
        round=5,
        last_offer={"player_1_gain": amount, "player_2_gain": 1000 - amount},
    )
    assert bargaining_strategy(game)["decision"] == expected


def test_bargaining_known_horizon_actions_are_deterministic_and_feasible():
    offered_to_bob = []
    for round_number in range(1, 6):
        game = bargaining_game(round=round_number)
        action = bargaining_strategy(game)
        assert action == bargaining_strategy(game)
        assert action["alice_gain"] + action["bob_gain"] == 1000
        offered_to_bob.append(action["bob_gain"])
    # Alternating proposer leverage can make the finite-horizon sequence nonmonotone.
    assert len(set(offered_to_bob)) > 1


def test_bargaining_unknown_horizon_needs_no_max_rounds():
    game = bargaining_game(round=12, horizon_known=False)
    game["game_state"].pop("max_rounds")
    action = bargaining_strategy(game)
    assert action["alice_gain"] + action["bob_gain"] == 1000


def test_negotiation_complete_information_uses_surplus():
    seller_action = negotiation_strategy(negotiation_game(player="player_1"))
    buyer_action = negotiation_strategy(negotiation_game(player="player_2"))
    assert 70 < seller_action["product_price"] < 100
    assert 40 < buyer_action["product_price"] < 70


def test_negotiation_accepts_offer_better_than_target():
    game = negotiation_game(
        action_type="decision",
        last_offer={"price": 90, "from_player": "player_2"},
    )
    assert negotiation_strategy(game) == {"decision": "AcceptOffer"}


def test_negotiation_never_accepts_a_loss():
    game = negotiation_game(
        action_type="decision",
        round=5,
        last_offer={"price": 39, "from_player": "player_2"},
    )
    assert negotiation_strategy(game) == {"decision": "RejectOffer"}


def test_negotiation_final_rejection_has_no_illegal_counteroffer():
    game = negotiation_game(
        action_type="decision",
        player="player_2",
        round=5,
        last_offer={"price": 101, "from_player": "player_1"},
    )
    assert negotiation_strategy(game) == {"decision": "RejectOffer"}


def test_negotiation_incomplete_information_uses_opponent_history():
    game = negotiation_game(
        player="player_1",
        complete=False,
        round=3,
        history=[
            {"offer": {"price": 50, "from_player": "player_2"}},
            {"offer": {"price": 56, "from_player": "player_2"}},
        ],
    )
    action = negotiation_strategy(game)
    assert 52 < action["product_price"] <= 62


def test_persuasion_seller_always_recommends_high_quality():
    assert persuasion_strategy(persuasion_game("seller_recommendation")) == {"decision": "yes"}
    text = persuasion_strategy(persuasion_game("seller_message"))["message"]
    assert "high" in text.lower()


def test_persuasion_low_quality_is_truthful_without_buyer_values():
    game = persuasion_game("seller_recommendation", current_quality="low", v=None, u=None)
    assert persuasion_strategy(game) == {"decision": "no"}


@pytest.mark.parametrize(
    "price,expected",
    [(-0.1, "yes"), (2.0, "no"), (3.0, "no")],
)
def test_persuasion_buyer_respects_value_bounds(price, expected):
    game = persuasion_game("buyer_decision", product_price=price)
    assert persuasion_strategy(game) == {"decision": expected}


def test_persuasion_buyer_understands_explicit_negative_text():
    game = persuasion_game(
        "buyer_decision",
        seller_message="I do not recommend buying this unit.",
        product_price=0.5,
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_persuasion_buyer_updates_after_bad_recommendations():
    positive = persuasion_game("buyer_decision", seller_message="yes", p=0.8)
    assert persuasion_strategy(positive) == {"decision": "yes"}

    bad_history = [{"seller_message": "yes", "bought": True, "quality": "low"} for _ in range(4)]
    updated = persuasion_game("buyer_decision", seller_message="yes", history=bad_history, p=0.8)
    assert persuasion_strategy(updated) == {"decision": "no"}


def test_low_quality_pooling_is_reproducible():
    game = persuasion_game("seller_recommendation", current_quality="low", round=10)
    assert persuasion_strategy(game) == persuasion_strategy(game)
