"""Physics-aware AI model simulator with calibrated probabilistic predictions."""

from __future__ import annotations

import numpy as np

from hawks.config import AIModelConfig
from hawks.manufacturing.physics_state import PrintTrace
from hawks.types import (
    DefectInstance,
    Detection,
    DetectionResult,
    ModelPrediction,
    PartResult,
)


def _collect_defects(part: PartResult) -> list[DefectInstance]:
    """Flatten all defects from a part's layers into a single list."""
    defects: list[DefectInstance] = []
    for layer in part.layers:
        defects.extend(layer.defects)
    return defects


class AIModelSimulator:
    """Simulated AI model that produces probabilistic predictions from physics traces.

    Takes per-part structural risk (PrintTrace.max_structural_risk) as input
    and produces calibrated probabilistic predictions. Supports preset
    calibration modes: well_calibrated, overconfident, underconfident.

    Mathematical model (exactly 2 RNG draws per part):
    1. Classification: Bernoulli draw using TPR (if defects) or FPR (if clean)
    2. Calibrated probability: sigmoid(logit(clamped_risk) + calibration_bias)
    3. Confidence: |predicted_prob - 0.5| * 2 + N(0, noise^2)
    4. Correctness: predicted_positive == has_defects
    """

    def __init__(self, config: AIModelConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    def predict(self, part: PartResult, trace: PrintTrace) -> ModelPrediction:
        """Produce a probabilistic prediction for a single part.

        Draws exactly 2 random numbers: 1 uniform (Bernoulli) + 1 normal (noise).
        """
        has_defects = part.total_defect_count > 0

        # 1. Classification (Bernoulli)
        rate = self._config.true_positive_rate if has_defects else self._config.false_positive_rate
        predicted_positive = bool(self._rng.random() < rate)

        # 2. Calibrated probability
        eps = 1e-6
        true_risk = trace.max_structural_risk
        clamped = np.clip(true_risk, eps, 1.0 - eps)
        logit_risk = float(np.log(clamped / (1.0 - clamped)))
        predicted_prob = float(1.0 / (1.0 + np.exp(-(logit_risk + self._config.calibration_bias))))

        # 3. Confidence score
        base_confidence = abs(predicted_prob - 0.5) * 2.0
        noise = self._rng.normal(0.0, self._config.confidence_noise)
        confidence_score = float(np.clip(base_confidence + noise, 0.0, 1.0))

        # 4. Correctness
        correct = predicted_positive == has_defects

        return ModelPrediction(
            part_id=part.part_id,
            predicted_defect_probability=predicted_prob,
            confidence_score=confidence_score,
            predicted_positive=predicted_positive,
            correct=correct,
        )

    def to_detection_result(
        self, prediction: ModelPrediction, part: PartResult
    ) -> DetectionResult:
        """Bridge a ModelPrediction to a DetectionResult for downstream consumers."""
        all_defects = _collect_defects(part)
        has_defects = len(all_defects) > 0

        if prediction.predicted_positive and has_defects:
            # True Positive: one Detection per actual defect
            detections = [
                Detection(
                    defect_ref=defect,
                    confidence=prediction.confidence_score,
                    flagged=True,
                )
                for defect in all_defects
            ]
            return DetectionResult(
                part_id=prediction.part_id,
                detections=detections,
                missed=[],
                false_alarms=0,
            )

        if prediction.predicted_positive and not has_defects:
            # False Positive: one Detection with no defect ref
            detections = [
                Detection(
                    defect_ref=None,
                    confidence=prediction.confidence_score,
                    flagged=True,
                )
            ]
            return DetectionResult(
                part_id=prediction.part_id,
                detections=detections,
                missed=[],
                false_alarms=1,
            )

        if not prediction.predicted_positive and has_defects:
            # False Negative: all defects missed
            return DetectionResult(
                part_id=prediction.part_id,
                detections=[],
                missed=list(all_defects),
                false_alarms=0,
            )

        # True Negative: nothing
        return DetectionResult(
            part_id=prediction.part_id,
            detections=[],
            missed=[],
            false_alarms=0,
        )

    def inspect_part(
        self, part: PartResult, trace: PrintTrace
    ) -> tuple[ModelPrediction, DetectionResult]:
        """Convenience: predict + bridge in one call."""
        prediction = self.predict(part, trace)
        det_result = self.to_detection_result(prediction, part)
        return prediction, det_result
