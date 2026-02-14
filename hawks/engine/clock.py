"""Simulation clock — manages discrete time progression."""

from __future__ import annotations

from hawks.config import ClockConfig, PopulationConfig
from hawks.types import TimeStep


class SimulationClock:
    """Manages discrete time steps and shift tracking."""

    def __init__(self, clock_config: ClockConfig, shift_duration_hours: float) -> None:
        self._num_steps = clock_config.num_steps
        self._parts_per_step = clock_config.parts_per_step
        self._shift_duration = shift_duration_hours
        self._current_step = 0
        self._shift_hour = 0.0
        # Assume each step represents some fraction of an hour
        self._hours_per_step = shift_duration_hours / max(clock_config.num_steps, 1)

    def tick(self) -> TimeStep:
        """Advance one step and return the current TimeStep."""
        ts = TimeStep(step=self._current_step, shift_hour=self._shift_hour)
        self._current_step += 1
        self._shift_hour += self._hours_per_step
        # Wrap shift hours
        if self._shift_hour >= self._shift_duration:
            self._shift_hour = 0.0
        return ts

    def is_shift_change(self) -> bool:
        """Check if a shift change just occurred (shift_hour wrapped to 0)."""
        return self._shift_hour == 0.0 and self._current_step > 0

    @property
    def parts_per_step(self) -> int:
        return self._parts_per_step

    @property
    def total_steps(self) -> int:
        return self._num_steps

    @property
    def current_step(self) -> int:
        return self._current_step
