"""AI defect detector with configurable ROC characteristics."""

from __future__ import annotations

import numpy as np

from hawks.config import DetectionConfig
from hawks.types import DefectInstance, Detection, DetectionResult, PartResult


class AIDefectDetector:
    """Simulated AI sensor with configurable sensitivity/specificity.

    Detection logic per defect:
    1. Look up sensitivity for this defect_type and severity (sigmoid curve)
    2. Draw Bernoulli(sensitivity) — detected or missed
    3. If detected, assign confidence = clip(true_conf + noise, 0, 1)
    4. Generate false positives via Bernoulli(1 - specificity) per clean layer
    """

    def __init__(self, config: DetectionConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    def inspect_part(self, part: PartResult) -> DetectionResult:
        """Produce detections with confidence scores for a single part."""
        detections: list[Detection] = []
        missed: list[DefectInstance] = []
        false_alarms = 0

        for layer in part.layers:
            if layer.defects:
                for defect in layer.defects:
                    sensitivity = self._effective_sensitivity(defect)
                    if self._rng.random() < sensitivity:
                        # True positive
                        true_conf = sensitivity
                        noise = self._rng.normal(0, self._config.confidence_noise_std)
                        confidence = float(np.clip(true_conf + noise, 0.0, 1.0))
                        detections.append(
                            Detection(
                                defect_ref=defect,
                                confidence=confidence,
                                flagged=confidence >= 0.5,
                            )
                        )
                    else:
                        missed.append(defect)
            else:
                # Clean layer — potential false positive
                if self._rng.random() > self._config.base_specificity:
                    false_alarms += 1
                    noise = self._rng.normal(0, self._config.confidence_noise_std)
                    confidence = float(np.clip(0.3 + noise, 0.0, 1.0))
                    detections.append(
                        Detection(
                            defect_ref=None,
                            confidence=confidence,
                            flagged=confidence >= 0.5,
                        )
                    )

        return DetectionResult(
            part_id=part.part_id,
            detections=detections,
            missed=missed,
            false_alarms=false_alarms,
        )

    def _effective_sensitivity(self, defect: DefectInstance) -> float:
        """Compute detection probability based on defect type and severity.

        Uses a sigmoid curve: low severity = lower detection probability.
        """
        type_sensitivity = self._config.sensitivity_by_type.get(
            defect.defect_type, self._config.base_sensitivity
        )

        if self._config.severity_sensitivity_curve == "sigmoid":
            # Sigmoid mapping: severity -> multiplier in [0.3, 1.0]
            # At severity=0.5, multiplier ≈ 0.65
            k = 8.0  # steepness
            midpoint = 0.4
            sigmoid = 1.0 / (1.0 + np.exp(-k * (defect.severity - midpoint)))
            multiplier = 0.3 + 0.7 * sigmoid
        else:
            multiplier = defect.severity

        return float(np.clip(type_sensitivity * multiplier, 0.0, 1.0))
