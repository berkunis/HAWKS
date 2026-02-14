"""Manufacturing digital twin — produces parts with ground-truth defects."""

from __future__ import annotations

import numpy as np

from hawks.config import ManufacturingConfig
from hawks.manufacturing.defect_gen import DefectGenerator
from hawks.manufacturing.part import build_part
from hawks.types import PartResult


class ManufacturingTwin:
    """Digital twin that produces parts with ground-truth defects each time step."""

    def __init__(self, config: ManufacturingConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._defect_gen = DefectGenerator(config, rng)
        self._part_counter = 0

    def produce_parts(self, n: int) -> list[PartResult]:
        """Generate n parts with layer-by-layer defect sampling."""
        parts = []
        for _ in range(n):
            self._part_counter += 1
            part_id = f"PART-{self._part_counter:06d}"
            layer_defects = self._defect_gen.sample_defects(
                self._config.num_layers_per_part
            )
            parts.append(build_part(part_id, layer_defects))
        return parts
