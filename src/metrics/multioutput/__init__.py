"""Metrics for multi-output learning."""

from .hamming_loss import HammingLoss
from .macro import MacroAverage
from .micro import MicroAverage

__all__ = [
    "HammingLoss",
    "MacroAverage",
    "MicroAverage",
]