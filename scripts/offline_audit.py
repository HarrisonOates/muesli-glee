"""Generate a reproducible policy-safety and behavior audit.

This is not a claim about leaderboard performance.  It exercises randomized legal
states and checks invariants that should hold before the agent is put in a live queue.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from muesli_agent.strategy import strategy as granola_strategy

SEED = 20260823
PER_FAMILY = 4_000
StrategyCallback = Callable[[dict[str, Any]], dict[str, Any]]


def _profile_strategy(profile: str) -> StrategyCallback:
    if profile == "granola":
        return granola_strategy
    if profile == "challenger":
        from muesli_agent.challenger import challenger_strategy

        return challenger_strategy
    raise ValueError(f"Unknown profile: {profile!r}")


def game(family: str, player: str, action_type: str, state: dict[str, Any], index: int):
    return {
        "game_id": f"audit-{family}-{index}",
        "game_family": family,
        "your_player": player,
        "game_state": state,
        "valid_actions": {"type": action_type, "fields": {}},
    }


def audit_bargaining(rng: random.Random, strategy: StrategyCallback) -> dict[str, int]:
    violations = 0
    offers = decisions = 0
    for index in range(PER_FAMILY):
        money = rng.randint(10, 10_000)
        player = rng.choice(("player_1", "player_2"))
        rounds = rng.randint(1, 20)
        current = rng.randint(1, rounds)
        common = {
            "current_player": player,
            "round": current,
            "max_rounds": rounds,
            "horizon_known": True,
            "money_to_divide": money,
            "delta_1": rng.uniform(0.5, 0.99),
            "delta_2": rng.uniform(0.5, 0.99),
            "history": [],
            "messages_allowed": rng.choice((True, False)),
        }
        if index % 2 == 0:
            action = strategy(game("bargaining", player, "offer", common, index))
            offers += 1
            total = float(action.get("alice_gain", -money)) + float(action.get("bob_gain", -money))
            violations += int(abs(total - money) > 1e-6)
            violations += int(not (0 <= float(action["alice_gain"]) <= money))
            violations += int(not (0 <= float(action["bob_gain"]) <= money))
            violations += int((not common["messages_allowed"]) and "message" in action)
        else:
            my_share = rng.random()
            if player == "player_1":
                last_offer = {
                    "player_1_gain": money * my_share,
                    "player_2_gain": money * (1 - my_share),
                }
            else:
                last_offer = {
                    "player_1_gain": money * (1 - my_share),
                    "player_2_gain": money * my_share,
                }
            common["last_offer"] = last_offer
            action = strategy(game("bargaining", player, "decision", common, index))
            decisions += 1
            violations += int(action.get("decision") not in {"accept", "reject", "walkaway"})
            if current == rounds and my_share > 0:
                violations += int(action["decision"] != "accept")
    return {
        "states": PER_FAMILY,
        "offers": offers,
        "decisions": decisions,
        "violations": violations,
    }


def audit_negotiation(rng: random.Random, strategy: StrategyCallback) -> dict[str, int]:
    violations = 0
    offers = decisions = private_evidence_decisions = 0
    for index in range(PER_FAMILY):
        seller_value = rng.uniform(5, 100)
        buyer_value = rng.uniform(5, 100)
        player = rng.choice(("player_1", "player_2"))
        role = "seller" if player == "player_1" else "buyer"
        own_value = seller_value if role == "seller" else buyer_value
        rounds = rng.randint(1, 20)
        current = rng.randint(1, rounds)
        complete = rng.choice((True, False))
        common: dict[str, Any] = {
            "current_player": player,
            "player_1_role": "seller",
            "player_2_role": "buyer",
            f"{player}_value": own_value,
            "round": current,
            "max_rounds": rounds,
            "horizon_known": True,
            "history": [],
            "messages_allowed": rng.choice((True, False)),
            "complete_information": complete,
        }
        if complete:
            common["player_1_value"] = seller_value
            common["player_2_value"] = buyer_value

        if index % 2 == 0:
            action = strategy(game("negotiation", player, "offer", common, index))
            offers += 1
            price = action.get("product_price")
            violations += int(not isinstance(price, (int, float)))
            violations += int((not common["messages_allowed"]) and "message" in action)
        else:
            price = rng.uniform(0, 120)
            if not complete and current > 1 and index % 4 == 3:
                common["history"] = [
                    {
                        "round": current - 1,
                        "offer": {
                            "price": own_value,
                            "from_player": player,
                            "round": current - 1,
                        },
                        "decision": "RejectOffer",
                        "decided_by": (
                            "player_2" if player == "player_1" else "player_1"
                        ),
                    }
                ]
                private_evidence_decisions += 1
            common["last_offer"] = {
                "price": price,
                "from_player": "player_2" if player == "player_1" else "player_1",
            }
            action = strategy(game("negotiation", player, "decision", common, index))
            decisions += 1
            profitable = price >= own_value if role == "seller" else price <= own_value
            accepted = action.get("decision") == "AcceptOffer"
            walked_away = action.get("decision") == "WalkAway"
            violations += int(
                action.get("decision") not in {"AcceptOffer", "RejectOffer", "WalkAway"}
            )
            violations += int(accepted and not profitable)
            violations += int(walked_away and not (complete and buyer_value <= seller_value))
            if current == rounds:
                violations += int(profitable and not accepted)
                violations += int("product_price" in action)
            elif not accepted and not walked_away:
                violations += int("product_price" not in action)
    return {
        "states": PER_FAMILY,
        "offers": offers,
        "decisions": decisions,
        "private_evidence_decisions": private_evidence_decisions,
        "violations": violations,
    }


def audit_persuasion(rng: random.Random, strategy: StrategyCallback) -> dict[str, int]:
    violations = 0
    seller_states = buyer_states = 0
    for index in range(PER_FAMILY):
        p = rng.uniform(0.05, 0.95)
        low_value = rng.uniform(0, 0.5)
        high_value = rng.uniform(1.5, 4.0)
        price = rng.uniform(-0.2, 4.2)
        common: dict[str, Any] = {
            "round": rng.randint(1, 20),
            "total_rounds": 20,
            "horizon_known": True,
            "p": p,
            "v": high_value,
            "u": low_value,
            "product_price": price,
            "history": [],
        }
        if index % 2 == 0:
            quality = rng.choice(("high", "low"))
            action_type = rng.choice(("seller_message", "seller_recommendation"))
            common.update({"current_player": "player_1", "current_quality": quality})
            action = strategy(game("persuasion", "player_1", action_type, common, index))
            seller_states += 1
            if action_type == "seller_recommendation":
                violations += int(action.get("decision") not in {"yes", "no"})
                if quality == "high":
                    violations += int(action["decision"] != "yes")
            else:
                violations += int(not isinstance(action.get("message"), str))
                if quality == "high":
                    violations += int("recommend" not in action["message"].lower())
        else:
            common.update(
                {
                    "current_player": "player_2",
                    "seller_message": rng.choice(
                        ("yes", "no", "I recommend buying.", "No view on this unit.")
                    ),
                }
            )
            action = strategy(game("persuasion", "player_2", "buyer_decision", common, index))
            buyer_states += 1
            violations += int(action.get("decision") not in {"yes", "no"})
            if price <= low_value:
                violations += int(action["decision"] != "yes")
            if price >= high_value:
                violations += int(action["decision"] != "no")
    return {
        "states": PER_FAMILY,
        "seller_states": seller_states,
        "buyer_states": buyer_states,
        "violations": violations,
    }


def behavioral_probes(strategy: StrategyCallback) -> dict[str, bool]:
    bargaining_actions = []
    negotiation_asks = []
    negotiation_history: list[dict[str, Any]] = []
    for round_number in range(1, 11):
        bargaining_state = {
            "current_player": "player_1",
            "round": round_number,
            "max_rounds": 10,
            "horizon_known": True,
            "money_to_divide": 1000,
            "delta_1": 0.9,
            "delta_2": 0.9,
            "history": [],
            "messages_allowed": False,
        }
        bargaining_actions.append(
            strategy(game("bargaining", "player_1", "offer", bargaining_state, round_number))
        )
        negotiation_state = {
            "current_player": "player_1",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 40,
            "player_2_value": 100,
            "round": round_number,
            "max_rounds": 10,
            "horizon_known": True,
            "history": list(negotiation_history),
            "messages_allowed": False,
        }
        price = strategy(
            game("negotiation", "player_1", "offer", negotiation_state, round_number)
        )["product_price"]
        negotiation_asks.append(price)
        negotiation_history.append(
            {
                "round": round_number,
                "offer": {
                    "price": price,
                    "from_player": "player_1",
                    "round": round_number,
                },
                "decision": "RejectOffer",
                "decided_by": "player_2",
            }
        )

    buyer_state = {
        "current_player": "player_2",
        "round": 5,
        "total_rounds": 10,
        "horizon_known": True,
        "p": 0.8,
        "v": 2.0,
        "u": 0.0,
        "product_price": 1.0,
        "seller_message": "yes",
        "history": [],
    }
    before = strategy(game("persuasion", "player_2", "buyer_decision", buyer_state, 1))
    buyer_state["history"] = [
        {"seller_message": "yes", "bought": True, "quality": "low"} for _ in range(4)
    ]
    after = strategy(game("persuasion", "player_2", "buyer_decision", buyer_state, 2))

    return {
        "bargaining_offers_feasible_and_state_sensitive": all(
            action["alice_gain"] + action["bob_gain"] == 1000 for action in bargaining_actions
        )
        and len({action["bob_gain"] for action in bargaining_actions}) > 1,
        "negotiation_asks_concede_monotonically": negotiation_asks
        == sorted(negotiation_asks, reverse=True),
        "buyer_reacts_to_bad_recommendations": before["decision"] == "yes"
        and after["decision"] == "no",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("granola", "challenger"),
        default="granola",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()
    output_path = args.output or Path(
        "results/offline_audit.json"
        if args.profile == "granola"
        else "results/offline_audit_challenger.json"
    )
    strategy = _profile_strategy(args.profile)
    rng = random.Random(SEED)
    families = {
        "bargaining": audit_bargaining(rng, strategy),
        "negotiation": audit_negotiation(rng, strategy),
        "persuasion": audit_persuasion(rng, strategy),
    }
    total_violations = sum(result["violations"] for result in families.values())
    output = {
        "description": "Randomized legal-state audit; not a leaderboard-performance estimate.",
        "profile": args.profile,
        "seed": SEED,
        "generated_states": sum(result["states"] for result in families.values()),
        "violations": total_violations,
        "families": families,
        "behavioral_probes": behavioral_probes(strategy),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return int(total_violations != 0 or not all(output["behavioral_probes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
