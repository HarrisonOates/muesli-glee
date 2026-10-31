from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from muesli_agent.bargaining import (
    BargainingConfig,
    _observed_offers,
    bargaining_strategy,
)


def game(
    action_type: str = "offer",
    player: str = "player_1",
    opponent_type: str = "hidden",
    **updates,
):
    state = {
        "phase": action_type,
        "current_player": player,
        "proposer": player,
        "round": 1,
        "max_rounds": 10,
        "horizon_known": True,
        "money_to_divide": 10_000,
        "delta_1": 0.90,
        "delta_2": 0.90,
        "last_offer": None,
        "history": [],
        "messages_allowed": True,
        "complete_information": True,
    }
    state.update(updates)
    return {
        "game_id": "bargaining-focused-test",
        "game_family": "bargaining",
        "your_player": player,
        "opponent": {"type": opponent_type, "name": None},
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


@pytest.mark.parametrize("player", ["player_1", "player_2"])
def test_offer_schema_and_exact_sum_for_both_roles(player):
    action = bargaining_strategy(game(player=player, money_to_divide=12_345.67))

    assert set(action) == {"alice_gain", "bob_gain", "message"}
    assert action["alice_gain"] + action["bob_gain"] == 12_345.67
    assert 0 <= action["alice_gain"] <= 12_345.67
    assert 0 <= action["bob_gain"] <= 12_345.67
    assert len(action["message"]) < 2_000


def test_exact_sum_is_preserved_for_nonterminating_decimal_pot():
    money = 1 / 3
    action = bargaining_strategy(game(money_to_divide=money))
    assert action["alice_gain"] + action["bob_gain"] == money


def test_offer_omits_language_when_messages_are_disabled():
    action = bargaining_strategy(game(messages_allowed=False))
    assert "message" not in action


def test_complete_information_protects_against_patient_receiver():
    # This is the parameterization of Granola's first live game.  The Rubinstein
    # anchor keeps the safe equal split that was accepted immediately.
    action = bargaining_strategy(game(delta_1=0.80, delta_2=0.90, max_rounds=12))
    assert action["bob_gain"] == pytest.approx(5_000)


def test_finite_horizon_does_not_force_a_monotone_fifty_fifty_concession():
    opening = bargaining_strategy(game(round=1, max_rounds=10))["bob_gain"]
    three_moves_left = bargaining_strategy(game(round=8, max_rounds=10))["bob_gain"]
    two_moves_left = bargaining_strategy(game(round=9, max_rounds=10))["bob_gain"]

    # Backward induction has alternating proposer leverage.  The behavioral layer
    # smooths the theoretical extremes but preserves that useful horizon signal.
    assert three_moves_left < opening
    assert three_moves_left < two_moves_left
    assert three_moves_left > 0


def test_disclosed_agent_terminal_offer_uses_take_it_or_leave_it_leverage():
    action = bargaining_strategy(game(round=1, max_rounds=1, opponent_type="agent"))

    assert action["bob_gain"] == 2_500
    assert "final round" in action["message"].lower()
    assert "zero" in action["message"].lower()


def test_terminal_offer_conditions_on_disclosed_opponent_class():
    agent_offer = bargaining_strategy(game(round=1, max_rounds=1, opponent_type="agent"))[
        "bob_gain"
    ]
    human_offer = bargaining_strategy(game(round=1, max_rounds=1, opponent_type="human"))[
        "bob_gain"
    ]
    hidden_offer = bargaining_strategy(game(round=1, max_rounds=1, opponent_type="hidden"))[
        "bob_gain"
    ]

    assert agent_offer < hidden_offer <= human_offer
    assert hidden_offer >= 4_000


def test_terminal_agent_floor_is_an_explicit_risk_control():
    cautious = BargainingConfig(disclosed_agent_terminal_floor=0.25)
    aggressive = replace(cautious, disclosed_agent_terminal_floor=0.15)

    cautious_offer = bargaining_strategy(
        game(round=1, max_rounds=1, opponent_type="agent"), cautious
    )["bob_gain"]
    aggressive_offer = bargaining_strategy(
        game(round=1, max_rounds=1, opponent_type="agent"), aggressive
    )["bob_gain"]

    assert cautious_offer == 2_500
    assert aggressive_offer == 1_500
    assert 10_000 - cautious_offer == 7_500


def test_earlier_rejection_updates_behavior_but_is_not_a_terminal_hard_floor():
    history = [
        {
            "round": 1,
            "proposer": "player_1",
            "offer": {"player_1_gain": 6_000, "player_2_gain": 4_000},
            "decision": "reject",
        }
    ]
    action = bargaining_strategy(
        game(
            round=10,
            max_rounds=10,
            opponent_type="agent",
            history=history,
        )
    )

    assert 0 < action["bob_gain"] < 4_000


def test_credible_rejection_causes_a_bounded_concession():
    history = [
        {
            "round": 1,
            "proposer": "player_1",
            "offer": {"player_1_gain": 5_200, "player_2_gain": 4_800},
            "decision": "reject",
        }
    ]
    without_evidence = bargaining_strategy(game(round=3, horizon_known=False, max_rounds=None))[
        "bob_gain"
    ]
    adapted = bargaining_strategy(
        game(round=3, horizon_known=False, max_rounds=None, history=history)
    )["bob_gain"]

    assert adapted > without_evidence
    assert 5_000 <= adapted <= 5_800


def test_contradictory_counteroffer_is_not_treated_as_a_hard_threshold():
    # The pattern occurred in the live smoke run: the opponent rejected receiving
    # 50%, then proposed keeping only 38.85%.  Escalating above 50% would mistake a
    # strategic rejection for evidence of inequity aversion.
    history = [
        {
            "round": 1,
            "proposer": "player_1",
            "offer": {"player_1_gain": 5_000, "player_2_gain": 5_000},
            "decision": "reject",
        },
        {
            "round": 2,
            "proposer": "player_2",
            "offer": {"player_1_gain": 6_115, "player_2_gain": 3_885},
            "decision": "reject",
        },
    ]
    action = bargaining_strategy(
        game(
            round=3,
            horizon_known=False,
            max_rounds=None,
            delta_1=0.80,
            delta_2=0.95,
            history=history,
        )
    )

    assert action["bob_gain"] <= 5_000


def test_only_a_later_counteroffer_can_contradict_a_rejection():
    history = [
        {
            "round": 1,
            "proposer": "player_2",
            "offer": {"player_1_gain": 7_000, "player_2_gain": 3_000},
            "decision": "reject",
        },
        {
            "round": 2,
            "proposer": "player_1",
            "offer": {"player_1_gain": 6_000, "player_2_gain": 4_000},
            "decision": "reject",
        },
    ]
    config = BargainingConfig(
        routine_max_offer_share=0.40,
        rational_mixture_weight=0.0,
        agent_rational_weight=0.0,
        human_rational_weight=0.0,
        threshold_sigma=0.03,
    )
    action = bargaining_strategy(
        game(
            round=4,
            horizon_known=False,
            complete_information=False,
            delta_1=0.99,
            delta_2=0.50,
            history=history,
        ),
        config,
    )

    assert action["bob_gain"] >= 4_200


def test_current_last_offer_supplies_weak_aspiration_evidence():
    without_evidence = bargaining_strategy(
        game(round=3, horizon_known=False, max_rounds=None, messages_allowed=False)
    )["bob_gain"]
    last_offer = {
        "player_1_gain": 7_000,
        "player_2_gain": 3_000,
        "proposer": "player_2",
        "round": 2,
    }
    with_evidence = bargaining_strategy(
        game(
            round=3,
            horizon_known=False,
            max_rounds=None,
            last_offer=last_offer,
            messages_allowed=False,
        )
    )["bob_gain"]

    assert with_evidence < without_evidence


def test_last_offer_is_exactly_deduped_when_already_in_history():
    last_offer = {
        "player_1_gain": 7_000,
        "player_2_gain": 3_000,
        "proposer": "player_2",
        "round": 2,
    }
    current_only = game(
        round=3,
        horizon_known=False,
        max_rounds=None,
        last_offer=last_offer,
        messages_allowed=False,
    )
    mirrored = game(
        round=3,
        horizon_known=False,
        max_rounds=None,
        last_offer=last_offer,
        history=[
            {
                "round": 2,
                "proposer": "player_2",
                "offer": {"player_1_gain": 7_000, "player_2_gain": 3_000},
            }
        ],
        messages_allowed=False,
    )

    assert len(_observed_offers(mirrored["game_state"])) == 1
    assert bargaining_strategy(current_only) == bargaining_strategy(mirrored)


def test_equal_split_insurance_guard_fades_near_deadline():
    config = BargainingConfig(
        hidden_threshold_center=0.20,
        agent_threshold_center=0.20,
        human_threshold_center=0.20,
        threshold_sigma=0.02,
        rational_mixture_weight=0.0,
        agent_rational_weight=0.0,
        human_rational_weight=0.0,
        routine_max_offer_share=0.40,
        equilibrium_max_offer_share=0.40,
    )
    early = bargaining_strategy(game(round=1, max_rounds=12, delta_1=0.80, delta_2=0.90), config)[
        "bob_gain"
    ]
    late = bargaining_strategy(game(round=11, max_rounds=12, delta_1=0.80, delta_2=0.90), config)[
        "bob_gain"
    ]

    assert early == 4_000
    assert 0 < late < early


def test_bounded_rollout_can_reject_counteroffer_for_later_proposer_value():
    last_offer = {
        "player_1_gain": 3_000,
        "player_2_gain": 7_000,
        "proposer": "player_2",
        "round": 6,
    }
    payload = game(
        action_type="decision",
        player="player_1",
        opponent_type="agent",
        round=6,
        max_rounds=10,
        last_offer=last_offer,
    )
    one_step = replace(BargainingConfig(), finite_lookahead_rounds=0)

    assert bargaining_strategy(payload, one_step) == {"decision": "accept"}
    assert bargaining_strategy(payload) == {"decision": "reject"}


def test_unknown_horizon_explicitly_uses_one_step_boundary():
    payload = game(
        round=6,
        horizon_known=False,
        max_rounds=None,
        opponent_type="agent",
        messages_allowed=False,
    )
    no_rollout = replace(BargainingConfig(), finite_lookahead_rounds=0)
    deep_rollout = replace(BargainingConfig(), finite_lookahead_rounds=20)

    assert bargaining_strategy(payload, no_rollout) == bargaining_strategy(payload, deep_rollout)


def test_accepts_live_smoke_offer_when_future_is_discounted():
    action = bargaining_strategy(
        game(
            action_type="decision",
            player="player_2",
            horizon_known=False,
            max_rounds=None,
            delta_1=1.0,
            delta_2=0.90,
            last_offer={"player_1_gain": 5_200, "player_2_gain": 4_800},
        )
    )
    assert action == {"decision": "accept"}


def test_rejects_offer_below_discounted_continuation_value():
    action = bargaining_strategy(
        game(
            action_type="decision",
            player="player_1",
            horizon_known=False,
            max_rounds=None,
            last_offer={"alice_gain": 2_000, "bob_gain": 8_000},
        )
    )
    assert action == {"decision": "reject"}


@pytest.mark.parametrize("amount, expected", [(1, "accept"), (0, "reject")])
def test_final_round_accepts_every_strictly_positive_gain(amount, expected):
    action = bargaining_strategy(
        game(
            action_type="decision",
            round=10,
            last_offer={
                "player_1_gain": amount,
                "player_2_gain": 10_000 - amount,
            },
        )
    )
    assert action == {"decision": expected}


def test_incomplete_information_and_unknown_horizon_need_no_hidden_fields():
    payload = game(
        player="player_2",
        complete_information=False,
        horizon_known=False,
        round=37,
    )
    payload["game_state"].pop("delta_1")
    payload["game_state"].pop("max_rounds")

    action = bargaining_strategy(payload)
    assert action["alice_gain"] + action["bob_gain"] == 10_000


def test_large_known_horizon_does_not_depend_on_recursion_depth():
    action = bargaining_strategy(game(max_rounds=2_000))
    assert action["alice_gain"] + action["bob_gain"] == 10_000


@pytest.mark.parametrize("player", ["player_1", "player_2"])
def test_representative_configuration_grid_preserves_offer_legality(player):
    horizons = ((True, 1, 1), (True, 3, 8), (False, 17, None))
    for opponent_type, complete, horizon, deltas in product(
        ("agent", "human", "hidden"),
        (True, False),
        horizons,
        product((0.0, 0.50, 0.90, 1.0), repeat=2),
    ):
        known, round_number, maximum = horizon
        delta_1, delta_2 = deltas
        payload = game(
            player=player,
            opponent_type=opponent_type,
            complete_information=complete,
            horizon_known=known,
            round=round_number,
            max_rounds=maximum,
            delta_1=delta_1,
            delta_2=delta_2,
            messages_allowed=False,
        )
        if maximum is None:
            payload["game_state"].pop("max_rounds")
        if not complete:
            hidden_delta = "delta_2" if player == "player_1" else "delta_1"
            payload["game_state"].pop(hidden_delta)

        action = bargaining_strategy(payload)
        assert set(action) == {"alice_gain", "bob_gain"}
        assert action["alice_gain"] + action["bob_gain"] == 10_000
        assert 0 <= action["alice_gain"] <= 10_000
        assert 0 <= action["bob_gain"] <= 10_000


def test_policy_is_deterministic_and_does_not_mutate_payload():
    payload = game(opponent_type="human")
    original_state = dict(payload["game_state"])

    assert bargaining_strategy(payload) == bargaining_strategy(payload)
    assert payload["game_state"] == original_state


def test_custom_config_controls_rejection_concession():
    history = [
        {
            "proposer": "player_1",
            "offer": {"alice_gain": 6_000, "bob_gain": 4_000},
            "decision": {"decision": "reject"},
        }
    ]
    config = BargainingConfig(rejection_concession=0.05)
    action = bargaining_strategy(game(round=3, history=history), config)

    assert action["bob_gain"] >= 4_500
