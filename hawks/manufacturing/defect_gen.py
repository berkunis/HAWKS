"""Spatial defect generation with Markov chain correlation across layers."""

from __future__ import annotations

import numpy as np

from hawks.config import ManufacturingConfig
from hawks.types import DefectInstance


class DefectGenerator:
    """Sample spatially-correlated defects across layers.

    Uses a Markov chain: if layer i has a defect, layer i+1 has elevated
    probability (controlled by spatial_correlation). Severity is drawn from
    a Beta distribution.
    """

    def __init__(self, config: ManufacturingConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

        # Normalize defect type weights into probabilities
        weights = config.defect_type_weights
        total = sum(weights.values())
        self._defect_types = list(weights.keys())
        self._defect_probs = np.array([weights[t] / total for t in self._defect_types])

    def sample_defects(self, num_layers: int) -> list[list[DefectInstance]]:
        """Sample spatially-correlated defects across layers."""
        cfg = self._config
        result: list[list[DefectInstance]] = []
        prev_had_defect = False

        for layer_idx in range(num_layers):
            # Markov chain: elevate probability if previous layer had a defect
            if prev_had_defect:
                p_defect = min(
                    cfg.defect_base_rate + cfg.spatial_correlation * cfg.defect_base_rate,
                    1.0,
                )
            else:
                p_defect = cfg.defect_base_rate

            layer_defects: list[DefectInstance] = []

            if self._rng.random() < p_defect:
                # This layer has at least one defect
                defect_type = self._rng.choice(self._defect_types, p=self._defect_probs)
                severity = self._sample_severity()
                spatial_pos = int(self._rng.integers(0, 1000))

                layer_defects.append(
                    DefectInstance(
                        defect_type=defect_type,
                        severity=severity,
                        location=(layer_idx, spatial_pos),
                    )
                )
                prev_had_defect = True
            else:
                prev_had_defect = False

            result.append(layer_defects)

        return result

    def _sample_severity(self) -> float:
        """Draw severity from configured distribution."""
        if self._config.severity_distribution == "beta":
            # Beta(2, 5) skews toward lower severity — most defects are mild
            return float(self._rng.beta(2.0, 5.0))
        else:
            return float(self._rng.uniform(0.0, 1.0))
