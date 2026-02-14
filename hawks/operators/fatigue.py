"""Fatigue model strategy — exponential fatigue growth within shift."""

from __future__ import annotations

import numpy as np

from hawks.types import TimeStep


class FatigueModel:
    """Exponential fatigue growth within shift, reset on break.

    Fatigue grows exponentially during the shift and is reset when the
    operator takes a break (shift change).
    """

    def __init__(self, fatigue_rate: float, vigilance_decrement_rate: float) -> None:
        self._fatigue_rate = fatigue_rate
        self._vigilance_decrement_rate = vigilance_decrement_rate
        self._fatigue = 0.0

    def accumulate(self, time_step: TimeStep) -> None:
        """Accumulate fatigue based on time into shift."""
        # Exponential fatigue growth
        growth = self._fatigue_rate * (1.0 + self._vigilance_decrement_rate * time_step.shift_hour)
        self._fatigue = min(1.0, self._fatigue + growth)

    def get_fatigue(self) -> float:
        """Return current fatigue level (0.0 = fresh, 1.0 = exhausted)."""
        return self._fatigue

    def rest(self) -> None:
        """Reset fatigue on shift change / break."""
        self._fatigue = 0.0
