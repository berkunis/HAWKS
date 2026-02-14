"""Manufacturing digital twin — produces parts with ground-truth defects."""

from __future__ import annotations

import numpy as np

from hawks.config import ManufacturingConfig
from hawks.manufacturing.part import build_part
from hawks.manufacturing.physics_engine import PhysicsEngine
from hawks.manufacturing.physics_state import PrintTrace
from hawks.types import PartResult


class ManufacturingTwin:
    """Digital twin that produces parts with ground-truth defects each time step."""

    def __init__(self, config: ManufacturingConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._physics = PhysicsEngine(config, rng)
        self._part_counter = 0
        self._last_traces: list[PrintTrace] = []

    def produce_parts(self, n: int) -> list[PartResult]:
        """Generate n parts with physics-based defect simulation."""
        parts = []
        traces = []
        for _ in range(n):
            self._part_counter += 1
            part_id = f"PART-{self._part_counter:06d}"
            layer_defects, trace = self._physics.simulate_print(part_id)
            parts.append(build_part(part_id, layer_defects))
            traces.append(trace)
        self._last_traces = traces
        return parts

    def get_last_traces(self) -> list[PrintTrace]:
        """Return physics traces from the most recent produce_parts() call."""
        return self._last_traces
