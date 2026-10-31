"""Run a deterministic, algorithmic negotiation parameter sweep.

This is a development simulator, not a leaderboard estimator.  It uses the
same structured payload shape as GLEE and pairs every policy variant with the
same values, roles, and opponent seeds.  The opponent models are deliberately
simple: they provide a useful stress-test distribution without pretending to
reconstruct hidden counterfactual responses from the live archive.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from muesli_agent.negotiation import DEFAULT_CONFIG, NegotiationConfig, negotiation_strategy


@dataclass(frozen=True)
class Scenario:
    seller_value: float
    buyer_value: float
    our_player: str
    max_rounds: int
    horizon_known: bool = True
    complete_information: bool = True


@dataclass(frozen=True)
class Opponent:
    name: str
    initial_claim: float
    final_claim: float
    concession_power: float
    walkaway_after: int | None = None


OPPONENTS = (
    Opponent("boulware", 0.86, 0.56, 3.5),
    Opponent("conceder", 0.66, 0.50, 1.0),
    Opponent("tit_for_tat", 0.70, 0.50, 1.8),
    Opponent("fixed_threshold", 0.62, 0.62, 1.0),
    Opponent("walkaway_sensitive", 0.78, 0.52, 2.2, walkaway_after=3),
)


@dataclass(frozen=True)
class Outcome:
    agreement: bool
    own_payoff: float
    pot: float
    moves: int
    walked_away: bool = False

    @property
    def normalized_payoff(self) -> float:
        return self.own_payoff / self.pot if self.pot > 0 else 0.0


Strategy = Callable[[dict[str, Any], NegotiationConfig], dict[str, Any]]


def _utility(price: float, value: float, role: str) -> float:
    return price - value if role == "seller" else value - price


def _role(player: str) -> str:
    return "seller" if player == "player_1" else "buyer"


def _other(player: str) -> str:
    return "player_2" if player == "player_1" else "player_1"


def _progress(round_number: int, max_rounds: int, known: bool) -> float:
    if known:
        return min(1.0, max(0.0, (round_number - 1) / max(1, max_rounds - 1)))
    return 1.0 - 2.718281828459045 ** (-(round_number - 1) / 18.0)


def _opponent_claim(opponent: Opponent, round_number: int, max_rounds: int, known: bool) -> float:
    progress = _progress(round_number, max_rounds, known)
    eased = progress**opponent.concession_power
    return opponent.initial_claim + (opponent.final_claim - opponent.initial_claim) * eased


def _opponent_offer(scenario: Scenario, opponent: Opponent, round_number: int) -> float:
    seller = scenario.seller_value
    buyer = scenario.buyer_value
    surplus = max(0.0, buyer - seller)
    actor = _other(scenario.our_player)
    claim = _opponent_claim(opponent, round_number, scenario.max_rounds, scenario.horizon_known)
    if actor == "player_1":
        return seller + claim * surplus
    return buyer - claim * surplus


def _opponent_accepts(
    scenario: Scenario,
    opponent: Opponent,
    price: float,
    round_number: int,
    rejected_count: int,
) -> bool:
    if opponent.walkaway_after is not None and rejected_count >= opponent.walkaway_after:
        return False
    opponent_player = _other(scenario.our_player)
    opponent_role = _role(opponent_player)
    own_value = scenario.seller_value if opponent_role == "seller" else scenario.buyer_value
    surplus = max(0.0, scenario.buyer_value - scenario.seller_value)
    claim = _opponent_claim(opponent, round_number, scenario.max_rounds, scenario.horizon_known)
    required = claim * surplus
    return _utility(price, own_value, opponent_role) + 1e-9 >= required


def _payload(
    scenario: Scenario,
    current_player: str,
    phase: str,
    round_number: int,
    history: list[dict[str, Any]],
    last_offer: dict[str, Any] | None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "phase": phase,
        "current_player": current_player,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "round": round_number,
        "max_rounds": scenario.max_rounds,
        "horizon_known": scenario.horizon_known,
        "last_offer": last_offer,
        "history": history,
        "messages_allowed": False,
        "complete_information": scenario.complete_information,
    }
    if scenario.complete_information or current_player == "player_1":
        state["player_1_value"] = scenario.seller_value
    if scenario.complete_information or current_player == "player_2":
        state["player_2_value"] = scenario.buyer_value
    return {
        "game_id": "simulated-negotiation",
        "game_family": "negotiation",
        "your_player": current_player,
        "game_state": state,
        "valid_actions": {"type": phase, "fields": {}},
    }


def simulate(
    scenario: Scenario,
    opponent: Opponent,
    config: NegotiationConfig,
) -> Outcome:
    """Simulate one alternating-offer game from a structured GLEE state."""

    history: list[dict[str, Any]] = []
    current_player = "player_1"
    phase = "offer"
    last_offer: dict[str, Any] | None = None
    rejected_count = 0
    moves = 0

    while moves < scenario.max_rounds:
        round_number = moves + 1
        if current_player == scenario.our_player:
            action = negotiation_strategy(
                _payload(scenario, current_player, phase, round_number, history, last_offer),
                config,
            )
        elif phase == "offer":
            price = _opponent_offer(scenario, opponent, round_number)
            action = {"product_price": price}
        else:
            assert last_offer is not None
            price = float(last_offer["price"])
            if _opponent_accepts(scenario, opponent, price, round_number, rejected_count):
                action = {"decision": "AcceptOffer"}
            else:
                action = {
                    "decision": "RejectOffer",
                    "product_price": _opponent_offer(scenario, opponent, round_number),
                }

        moves += 1
        if phase == "offer":
            price = float(action["product_price"])
            last_offer = {"price": price, "from_player": current_player, "round": round_number}
            current_player = _other(current_player)
            phase = "decision"
            continue

        decision = action.get("decision")
        if decision == "AcceptOffer":
            assert last_offer is not None
            price = float(last_offer["price"])
            own_role = _role(scenario.our_player)
            own_value = scenario.seller_value if own_role == "seller" else scenario.buyer_value
            own_payoff = _utility(price, own_value, own_role)
            return Outcome(True, max(0.0, own_payoff), scenario.buyer_value - scenario.seller_value, moves)
        if decision == "WalkAway":
            return Outcome(False, 0.0, scenario.buyer_value - scenario.seller_value, moves, True)

        counter_price = float(action["product_price"])
        history.append(
            {
                "round": round_number,
                "offer": last_offer,
                "decision": "RejectOffer",
                "counteroffer": {"price": counter_price, "from_player": current_player},
                "decided_by": current_player,
            }
        )
        last_offer = {
            "price": counter_price,
            "from_player": current_player,
            "round": round_number + 1,
        }
        rejected_count += 1
        current_player = _other(current_player)

    return Outcome(False, 0.0, scenario.buyer_value - scenario.seller_value, moves)


def scenarios(seed: int, count: int) -> list[Scenario]:
    rng = random.Random(seed)
    values = (40.0, 80.0, 100.0, 120.0, 150.0, 1_000.0, 1_200.0, 1_500.0)
    result = []
    for _ in range(count):
        seller = rng.choice(values[:-1])
        larger_values = tuple(value for value in values if value > seller)
        buyer = rng.choice(larger_values) if larger_values else values[-1]
        result.append(
            Scenario(
                seller,
                buyer,
                rng.choice(("player_1", "player_2")),
                rng.choice((5, 8, 10, 15)),
                rng.random() < 0.75,
                rng.random() < 0.60,
            )
        )
    return result


def variants() -> dict[str, NegotiationConfig]:
    return {
        "baseline": DEFAULT_CONFIG,
        "legacy_complete_terminal_0.475": replace(
            DEFAULT_CONFIG,
            complete_terminal_own_surplus_share=0.475,
        ),
        "legacy_private_accept_0.85": replace(
            DEFAULT_CONFIG,
            private_evidence_certainty_discount=0.85,
        ),
        "accept_0.70": replace(DEFAULT_CONFIG, certainty_equivalent_discount=0.70),
        "accept_0.60": replace(DEFAULT_CONFIG, certainty_equivalent_discount=0.60),
        "faster_unknown": replace(DEFAULT_CONFIG, unknown_horizon_scale=10.0),
        "private_threshold_0.60": replace(
            DEFAULT_CONFIG, private_initial_threshold_fraction=0.60
        ),
        "combined_0.70_fast": replace(
            DEFAULT_CONFIG,
            certainty_equivalent_discount=0.70,
            unknown_horizon_scale=10.0,
        ),
    }


def run_sweep(seed: int, count: int) -> dict[str, Any]:
    sample = scenarios(seed, count)
    aggregate: dict[str, dict[str, float]] = defaultdict(
        lambda: {"games": 0.0, "agreements": 0.0, "walkaways": 0.0, "payoff": 0.0, "share": 0.0}
    )
    for name, config in variants().items():
        for scenario in sample:
            for opponent in OPPONENTS:
                outcome = simulate(scenario, opponent, config)
                row = aggregate[name]
                row["games"] += 1
                row["agreements"] += int(outcome.agreement)
                row["walkaways"] += int(outcome.walked_away)
                row["payoff"] += outcome.own_payoff
                row["share"] += outcome.normalized_payoff
    rendered: dict[str, Any] = {}
    for name, row in aggregate.items():
        games = row["games"]
        rendered[name] = {
            "games": int(games),
            "agreement_rate": row["agreements"] / games,
            "walkaway_rate": row["walkaways"] / games,
            "mean_payoff": row["payoff"] / games,
            "mean_normalized_payoff": row["share"] / games,
        }
    ranked = sorted(
        rendered,
        key=lambda name: rendered[name]["mean_normalized_payoff"],
        reverse=True,
    )
    return {"seed": seed, "scenarios": count, "opponents": [o.name for o in OPPONENTS], "results": rendered, "ranking": ranked}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_sweep(args.seed, args.scenarios)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
