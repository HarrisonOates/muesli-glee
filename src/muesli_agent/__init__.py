"""MUESLI GLEE competition agent."""

from .bargaining import BargainingConfig
from .negotiation import NegotiationConfig
from .persuasion import PersuasionConfig
from .strategy import (
    StrategyConfig,
    bargaining_strategy,
    negotiation_strategy,
    persuasion_strategy,
    strategy,
)

__all__ = [
    "BargainingConfig",
    "NegotiationConfig",
    "PersuasionConfig",
    "StrategyConfig",
    "bargaining_strategy",
    "negotiation_strategy",
    "persuasion_strategy",
    "strategy",
]

__version__ = "0.1.0"
