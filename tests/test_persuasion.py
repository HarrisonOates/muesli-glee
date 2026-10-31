from __future__ import annotations

import pytest

from muesli_agent.persuasion import (
    DEFAULT_PERSUASION_CONFIG,
    PersuasionConfig,
    _buyer_signal_responsiveness,
    _classify_signal,
    _estimate_seller_low_history,
    _low_pooling_probability,
    _positive_quality_posterior,
    _private_draw,
    persuasion_strategy,
)


def persuasion_game(action_type: str, **updates):
    player = "player_1" if action_type.startswith("seller") else "player_2"
    state = {
        "phase": action_type,
        "current_player": player,
        "round": 1,
        "total_rounds": 20,
        "p": 0.4,
        "v": 200,
        "u": 0,
        "product_price": 100,
        "current_quality": "high",
        "seller_message": "yes",
        "seller_message_type": "binary",
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


@pytest.mark.parametrize("action_type", ["seller_recommendation", "seller_message"])
def test_seller_always_recommends_high_quality(action_type):
    action = persuasion_strategy(persuasion_game(action_type))
    if action_type == "seller_recommendation":
        assert action == {"decision": "yes"}
    else:
        assert "recommend buying" in action["message"].lower()
        assert "high" in action["message"].lower()


def test_seller_limits_early_pooling_even_when_prior_purchase_is_profitable():
    # Live seller configuration A has E[value] = (1/3) * 400 > 100, but the
    # observed buyer still conditioned purchases on recommendations.  Do not
    # destroy that useful separation on the first low product.
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        p=1 / 3,
        v=400,
        u=0,
        product_price=100,
    )
    probability = _low_pooling_probability(game["game_state"], DEFAULT_PERSUASION_CONFIG)
    assert 0 < probability <= 0.25


def test_seller_increases_pooling_after_building_a_profitable_reputation():
    history = [
        *[
            {
                "round": round_number,
                "seller_message": "yes",
                "buyer_decision": "yes",
                "bought": True,
                "quality": "high",
            }
            for round_number in range(1, 8)
        ],
        *[
            {
                "round": round_number,
                "seller_message": "no",
                "buyer_decision": "no",
                "bought": False,
                "quality": "low",
            }
            for round_number in range(8, 14)
        ],
    ]
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=14,
        p=1 / 3,
        v=400,
        u=0,
        product_price=100,
        history=history,
    )
    later_probability = _low_pooling_probability(game["game_state"], DEFAULT_PERSUASION_CONFIG)
    early_game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        p=1 / 3,
        v=400,
        u=0,
        product_price=100,
    )
    early_probability = _low_pooling_probability(
        early_game["game_state"], DEFAULT_PERSUASION_CONFIG
    )
    assert later_probability > early_probability


def test_buyer_signal_responsiveness_distinguishes_adaptive_and_ignoring_buyers():
    responsive = [
        {"seller_message": "yes", "bought": True},
        {"seller_message": "yes", "bought": True},
        {"seller_message": "no", "bought": False},
        {"seller_message": "no", "bought": False},
    ]
    ignoring = [
        {"seller_message": "yes", "bought": True},
        {"seller_message": "yes", "bought": False},
        {"seller_message": "no", "bought": True},
        {"seller_message": "no", "bought": False},
    ]
    responsive_score, responsive_count = _buyer_signal_responsiveness(responsive)
    ignoring_score, ignoring_count = _buyer_signal_responsiveness(ignoring)
    assert responsive_count == ignoring_count == 4
    assert responsive_score > ignoring_score


def test_seller_spends_pooling_budget_faster_against_signal_ignoring_buyer():
    common_history = [
        {"round": 1, "seller_message": "yes", "bought": True},
        {"round": 2, "seller_message": "yes", "bought": False},
        {"round": 3, "seller_message": "no", "bought": True},
        {"round": 4, "seller_message": "no", "bought": False},
    ]
    responsive_history = [
        {"round": 1, "seller_message": "yes", "bought": True},
        {"round": 2, "seller_message": "yes", "bought": True},
        {"round": 3, "seller_message": "no", "bought": False},
        {"round": 4, "seller_message": "no", "bought": False},
    ]
    ignoring = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=6,
        p=0.5,
        v=400,
        u=0,
        product_price=100,
        history=common_history,
    )
    responsive = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=6,
        p=0.5,
        v=400,
        u=0,
        product_price=100,
        history=responsive_history,
    )
    assert _low_pooling_probability(ignoring["game_state"], DEFAULT_PERSUASION_CONFIG) > _low_pooling_probability(
        responsive["game_state"], DEFAULT_PERSUASION_CONFIG
    )


def test_seller_handles_hidden_buyer_values_and_preserves_early_reputation():
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        p=0.5,
        v=None,
        u=None,
        product_price=10_000,
    )
    probability = _low_pooling_probability(game["game_state"], DEFAULT_PERSUASION_CONFIG)
    assert 0 < probability < 0.1


@pytest.mark.parametrize("action_type", ["seller_recommendation", "seller_message"])
def test_seller_spends_reputation_on_final_low_quality_round(action_type):
    game = persuasion_game(
        action_type,
        current_quality="low",
        round=20,
        v=None,
        u=None,
    )
    action = persuasion_strategy(game)
    if action_type == "seller_recommendation":
        assert action == {"decision": "yes"}
    else:
        assert "recommend buying" in action["message"].lower()
        assert "quality is high" not in action["message"].lower()


def test_text_pooling_does_not_leak_quality_through_different_positive_phrasing():
    high = persuasion_game("seller_message", current_quality="high", round=20)
    low = persuasion_game("seller_message", current_quality="low", round=20)
    assert persuasion_strategy(high) == persuasion_strategy(low)


def test_seller_policy_is_deterministic_for_identical_full_history():
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=17,
        p=0.5,
        v=None,
        u=None,
        history=[
            {
                "round": round_number,
                "seller_message": "yes" if round_number % 2 else "no",
                "buyer_decision": "yes" if round_number % 2 else "no",
                "bought": bool(round_number % 2),
                "quality": "high" if round_number % 2 else "low",
            }
            for round_number in range(1, 17)
        ],
    )
    assert persuasion_strategy(game) == persuasion_strategy(game)


def test_seller_learns_from_a_buyer_who_purchases_a_negative_signal():
    history = [
        {
            "round": 1,
            "seller_message": "no",
            "buyer_decision": "yes",
            "bought": True,
            "quality": "low",
        }
    ]
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=10,
        p=0.5,
        v=None,
        u=None,
        product_price=10_000,
        history=history,
    )
    learned_probability = _low_pooling_probability(game["game_state"], DEFAULT_PERSUASION_CONFIG)
    baseline = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=10,
        p=0.5,
        v=None,
        u=None,
        product_price=10_000,
        history=[],
    )
    baseline_probability = _low_pooling_probability(
        baseline["game_state"], DEFAULT_PERSUASION_CONFIG
    )
    assert learned_probability > baseline_probability


def test_representative_quality_mix_exercises_both_pooling_actions(monkeypatch):
    # This synthetic ordering is a branch-coverage probe, not a reconstruction or
    # performance estimate for the smoke game.
    monkeypatch.delenv("GLEE_API_KEY", raising=False)
    qualities = [
        "low",
        "high",
        "low",
        "low",
        "high",
        "low",
        "low",
        "high",
        "low",
        "low",
        "high",
        "low",
        "low",
        "high",
        "low",
        "low",
        "high",
        "low",
        "low",
        "high",
    ]
    history = []
    for round_number, quality in enumerate(qualities, start=1):
        game = persuasion_game(
            "seller_recommendation",
            current_quality=quality,
            round=round_number,
            p=1 / 3,
            v=400,
            u=0,
            product_price=100,
            history=list(history),
        )
        signal = persuasion_strategy(game)["decision"]
        bought = signal == "yes"
        history.append(
            {
                "round": round_number,
                "seller_message": signal,
                "buyer_decision": "yes" if bought else "no",
                "bought": bought,
                "quality": quality,
            }
        )

    low_signals = [record["seller_message"] for record in history if record["quality"] == "low"]
    assert "yes" in low_signals
    assert "no" in low_signals
    assert all(record["bought"] for record in history if record["quality"] == "high")


@pytest.mark.parametrize(
    ("price", "expected"),
    [(-1, "yes"), (200, "no"), (250, "no")],
)
def test_buyer_obeys_value_dominance_guards(price, expected):
    game = persuasion_game("buyer_decision", product_price=price)
    assert persuasion_strategy(game) == {"decision": expected}


def test_buyer_does_not_invent_information_from_an_initial_positive_signal():
    # Live buyer configuration: p*v == price.  Cheap talk alone must not turn the
    # known break-even prior of .8 into the former hard-coded posterior of .9.
    game = persuasion_game(
        "buyer_decision",
        p=0.8,
        v=12_500,
        product_price=10_000,
        seller_message="yes",
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_buyer_uses_all_signal_exposures_not_only_purchased_products():
    # Six passed positive signals contain no quality labels.  Treating missing
    # labels as good outcomes would manufacture trust; complete pooling stays at p.
    history = [
        {
            "round": round_number,
            "seller_message": "yes",
            "buyer_decision": "no",
            "bought": False,
        }
        for round_number in range(1, 7)
    ]
    game = persuasion_game(
        "buyer_decision",
        round=7,
        p=0.8,
        v=12_500,
        product_price=10_000,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_buyer_can_learn_separation_from_observed_signal_frequency():
    # A preceding negative recommendation is observable even though its hidden
    # quality was never purchased.  It is evidence against complete pooling.
    history = [
        {
            "round": 1,
            "seller_message": "no",
            "buyer_decision": "no",
            "bought": False,
        }
    ]
    game = persuasion_game(
        "buyer_decision",
        round=2,
        p=0.8,
        v=12_500,
        product_price=10_000,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "yes"}


def test_buyer_uses_bounded_early_information_value_near_indifference():
    game = persuasion_game(
        "buyer_decision",
        round=1,
        p=0.5,
        v=100,
        u=0,
        product_price=50.3,
        seller_message="yes",
    )
    assert persuasion_strategy(game) == {"decision": "yes"}
    assert persuasion_strategy(
        game,
        PersuasionConfig(buyer_information_value_weight=0.0),
    ) == {"decision": "no"}


def test_buyer_accepts_structured_signal_alias_when_message_is_missing():
    game = persuasion_game(
        "buyer_decision",
        p=0.8,
        v=400,
        product_price=100,
        seller_message=None,
        seller_recommendation="no",
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_one_revealed_low_stops_live_trace_before_the_second_large_loss():
    # Granola observed one negative/pass, then six purchased high products under
    # positive signals.  The next purchased positive was low.  A recency-weighted,
    # asymmetric calibration must stop here rather than buying the second low too.
    history = [
        {"round": 1, "seller_message": "no", "bought": False},
        *[
            {
                "round": round_number,
                "seller_message": "yes",
                "buyer_decision": "yes",
                "bought": True,
                "quality": "high",
            }
            for round_number in range(2, 8)
        ],
        {
            "round": 8,
            "seller_message": "yes",
            "buyer_decision": "yes",
            "bought": True,
            "quality": "low",
        },
    ]
    game = persuasion_game(
        "buyer_decision",
        round=9,
        p=0.8,
        v=12_500,
        product_price=10_000,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_buyer_still_buys_before_the_first_failure_in_live_trace():
    history = [
        {"round": 1, "seller_message": "no", "bought": False},
        *[
            {
                "round": round_number,
                "seller_message": "yes",
                "bought": True,
                "quality": "high",
            }
            for round_number in range(2, 8)
        ],
    ]
    game = persuasion_game(
        "buyer_decision",
        round=8,
        p=0.8,
        v=12_500,
        product_price=10_000,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "yes"}


def test_one_ordinary_low_does_not_destroy_a_strictly_profitable_pooled_prior():
    history = [
        {
            "round": 1,
            "seller_message": "yes",
            "buyer_decision": "yes",
            "bought": True,
            "quality": "low",
        }
    ]
    game = persuasion_game(
        "buyer_decision",
        round=2,
        p=1 / 3,
        v=400,
        product_price=100,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "yes"}


@pytest.mark.parametrize(
    ("p", "v", "positive_count", "negative_count"),
    [
        (1 / 3, 1.2, 4, 9),
        (0.5, 1.2, 6, 6),
        (0.8, 1.2, 10, 2),
    ],
)
def test_buyer_can_find_truthful_sellers_across_the_official_prior_grid(
    p, v, positive_count, negative_count
):
    # All labels are deliberately missing: pass-inclusive signal frequency alone
    # should eventually identify a representative separating seller.
    signals = ["yes"] * positive_count + ["no"] * negative_count
    history = [
        {
            "round": round_number,
            "seller_message": signal,
            "buyer_decision": "no",
            "bought": False,
        }
        for round_number, signal in enumerate(signals, start=1)
    ]
    game = persuasion_game(
        "buyer_decision",
        round=len(history) + 1,
        p=p,
        v=v,
        u=0,
        product_price=1,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "yes"}


def test_buyer_does_not_accept_a_merely_indifferent_cap_pooling_frequency():
    # For p=1/3 and v=1.2, q*=5/6 and an incentive-compatible seller can send
    # positives at rate p/q*=0.4.  Conservative estimation preserves strictness.
    signals = ["yes"] * 5 + ["no"] * 7
    history = [
        {"round": index, "seller_message": signal, "bought": False}
        for index, signal in enumerate(signals, start=1)
    ]
    game = persuasion_game(
        "buyer_decision",
        round=13,
        p=1 / 3,
        v=1.2,
        u=0,
        product_price=1,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_buyer_discounts_reputation_on_the_final_round():
    history = [
        {"round": 1, "seller_message": "no", "bought": False},
        *[
            {
                "round": round_number,
                "seller_message": "yes",
                "bought": True,
                "quality": "high",
            }
            for round_number in range(2, 20)
        ],
    ]
    game = persuasion_game(
        "buyer_decision",
        round=20,
        p=0.5,
        v=120,
        product_price=100,
        seller_message="yes",
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_negative_text_takes_precedence_over_the_word_recommend():
    game = persuasion_game(
        "buyer_decision",
        p=0.8,
        v=400,
        seller_message="I do not recommend buying this product.",
    )
    assert persuasion_strategy(game) == {"decision": "no"}


@pytest.mark.parametrize(
    "seller_message",
    [
        "I strongly recommend purchasing this excellent product.",
        "This looks worthwhile and valuable.",
        "You should buy this premium item.",
    ],
)
def test_buyer_recognizes_varied_positive_text_semantics(seller_message):
    game = persuasion_game(
        "buyer_decision",
        p=0.8,
        v=400,
        seller_message=seller_message,
    )
    assert persuasion_strategy(game) == {"decision": "yes"}


@pytest.mark.parametrize(
    "seller_message",
    [
        "This is poor quality; skip it.",
        "I cannot recommend this purchase.",
        "This is not worth the price.",
    ],
)
def test_buyer_recognizes_varied_negative_text_semantics(seller_message):
    game = persuasion_game(
        "buyer_decision",
        p=0.8,
        v=400,
        seller_message=seller_message,
    )
    assert persuasion_strategy(game) == {"decision": "no"}


def test_random_identity_disclosure_does_not_change_policy_without_evidence():
    hidden = persuasion_game("buyer_decision", p=0.8, v=400)
    disclosed = persuasion_game("buyer_decision", p=0.8, v=400)
    hidden["opponent"] = {"type": "hidden", "name": None}
    disclosed["opponent"] = {"type": "human", "name": "Disclosed opponent"}
    assert persuasion_strategy(hidden) == persuasion_strategy(disclosed)


def test_unknown_horizon_is_not_mistaken_for_a_final_seller_round():
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=200,
        horizon_known=False,
        v=None,
        u=None,
    )
    game["game_state"].pop("total_rounds")
    probability = _low_pooling_probability(game["game_state"], DEFAULT_PERSUASION_CONFIG)
    assert 0 < probability < 1

    known_final = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=200,
        total_rounds=200,
        horizon_known=True,
        v=None,
        u=None,
    )
    assert _low_pooling_probability(known_final["game_state"], DEFAULT_PERSUASION_CONFIG) == 1


def test_missing_total_without_flag_also_uses_continuation_policy():
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=50,
        v=None,
        u=None,
    )
    game["game_state"].pop("total_rounds")
    probability = _low_pooling_probability(game["game_state"], DEFAULT_PERSUASION_CONFIG)
    assert 0 < probability < 1


def test_unknown_horizon_preserves_buyer_reputation_inference():
    history = [
        *[
            {
                "round": round_number,
                "seller_message": "yes",
                "buyer_decision": "yes",
                "bought": True,
                "quality": "high",
            }
            for round_number in range(88, 94)
        ],
        *[
            {
                "round": round_number,
                "seller_message": "no",
                "buyer_decision": "no",
                "bought": False,
            }
            for round_number in range(94, 100)
        ],
    ]
    game = persuasion_game(
        "buyer_decision",
        round=100,
        horizon_known=False,
        p=0.5,
        v=1.2,
        u=0,
        product_price=1,
        seller_message="yes",
        history=history,
    )
    game["game_state"].pop("total_rounds")
    assert persuasion_strategy(game) == {"decision": "yes"}


def test_one_negative_signal_improves_but_does_not_perfect_the_posterior():
    state = persuasion_game(
        "buyer_decision",
        round=2,
        p=0.8,
        seller_message="yes",
        history=[{"round": 1, "seller_message": "no", "bought": False}],
    )["game_state"]
    posterior = _positive_quality_posterior(state, 0.8, DEFAULT_PERSUASION_CONFIG)
    assert 0.8 < posterior < 0.9


def test_private_draw_is_retry_stable_and_keyed(monkeypatch):
    game = persuasion_game("seller_recommendation", current_quality="low", round=10)
    monkeypatch.delenv("GLEE_API_KEY", raising=False)
    fallback_first = _private_draw(game, game["game_state"])
    assert fallback_first == _private_draw(game, game["game_state"])

    monkeypatch.setenv("GLEE_API_KEY", "test-only-private-key-a")
    keyed_first = _private_draw(game, game["game_state"])
    assert keyed_first == _private_draw(game, game["game_state"])
    monkeypatch.setenv("GLEE_API_KEY", "test-only-private-key-b")
    keyed_second = _private_draw(game, game["game_state"])
    assert keyed_first != keyed_second
    assert fallback_first != keyed_first


def test_fractional_low_signal_action_is_stable_for_move_retries(monkeypatch):
    monkeypatch.setenv("GLEE_API_KEY", "test-only-retry-key")
    game = persuasion_game(
        "seller_recommendation",
        current_quality="low",
        round=10,
        v=None,
        u=None,
    )
    assert persuasion_strategy(game) == persuasion_strategy(game)


def test_partial_quality_accounting_estimates_only_unresolved_records():
    history = [
        {"seller_message": "yes", "quality": "low", "bought": True},
        {"seller_message": "yes", "quality": "high", "bought": True},
        {"seller_message": "no", "bought": False},
        {"seller_message": "yes", "bought": False},
        {"seller_message": "unclear", "bought": False},
    ]
    lows, pooled_lows = _estimate_seller_low_history(history, p=0.5)
    assert lows == pytest.approx(3.0)
    assert pooled_lows == pytest.approx(1.5)


@pytest.mark.parametrize(
    "message",
    [
        "I recommend buying this, but it is low quality.",
        "I cannot recommend it, although it is high quality.",
        "This is low quality but may be worth it.",
        "This is not low quality.",
    ],
)
def test_conflicting_or_double_negated_text_is_unknown(message):
    assert _classify_signal(message) == "unknown"


@pytest.mark.parametrize(
    "message",
    [
        "Being straight with you: this is not the premium one. Your call.",
        (
            "Honestly, today's product is not up to my usual standard. "
            "I'd understand if you sit this round out."
        ),
        "I'll be straight with you — this one isn't my best unit. You may want to pass today.",
    ],
)
def test_exact_live_negative_templates_are_negative(message):
    assert _classify_signal(message) == "negative"


@pytest.mark.parametrize(
    "message",
    [
        "This is not the premium one, but it is actually high quality.",
        "This isn't my best unit, although it is excellent and worthwhile.",
    ],
)
def test_live_negative_phrases_with_positive_conflicts_remain_unknown(message):
    assert _classify_signal(message) == "unknown"


def test_replay_learns_separation_in_negative_margin_two_template_game():
    negative = "Being straight with you: this is not the premium one. Your call."
    positive = "Honestly: this one is the high-quality type. Worth it."
    history = [
        {"round": 1, "seller_message": negative, "buyer_decision": "no", "bought": False},
        {"round": 2, "seller_message": negative, "buyer_decision": "no", "bought": False},
    ]
    high_game = persuasion_game(
        "buyer_decision",
        round=3,
        p=1 / 3,
        v=2_000_000,
        u=0,
        product_price=1_000_000,
        seller_message=positive,
        history=history,
    )
    assert persuasion_strategy(high_game) == {"decision": "yes"}

    negative_game = persuasion_game(
        "buyer_decision",
        round=3,
        p=1 / 3,
        v=2_000_000,
        u=0,
        product_price=1_000_000,
        seller_message=negative,
        history=history,
    )
    assert persuasion_strategy(negative_game) == {"decision": "no"}


def test_replay_learns_high_threshold_separation_after_repeated_negative_templates():
    negative_messages = [
        (
            "Honestly, today's product is not up to my usual standard. "
            "I'd understand if you sit this round out."
        ),
        "I'll be straight with you — this one isn't my best unit. You may want to pass today.",
    ]
    history = [
        {
            "round": round_number,
            "seller_message": negative_messages[round_number % 2],
            "buyer_decision": "no",
            "bought": False,
        }
        for round_number in range(1, 7)
    ]
    game = persuasion_game(
        "buyer_decision",
        round=7,
        p=1 / 3,
        v=125,
        u=0,
        product_price=100,
        seller_message=(
            "I can recommend this one without reservation: it's a high-quality unit "
            "and worth the price. I'd buy it."
        ),
        history=history,
    )
    assert persuasion_strategy(game) == {"decision": "yes"}


def test_structured_binary_label_precedes_conflicting_free_text():
    assert _classify_signal({"decision": "NO.", "message": "Please buy this"}) == "negative"
    assert _classify_signal({"label": "PASS!", "message": "Recommended"}) == "negative"


def test_conflicting_structured_labels_are_unknown():
    assert _classify_signal({"decision": "yes", "recommendation": "no"}) == "unknown"


def test_missing_buyer_value_is_rejected_only_on_buyer_turn():
    game = persuasion_game("buyer_decision", v=None)
    with pytest.raises(ValueError, match="v must be a finite number"):
        persuasion_strategy(game)
