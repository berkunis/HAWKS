"""Metrics collector — append-only recorder decoupled from simulation logic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hawks.types import DetectionResult, OperatorDecision, PartResult, TimeStep


class MetricsCollector:
    """Append-only recorder for simulation data. Converts to DataFrame for analysis."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record_step(
        self,
        time: TimeStep,
        parts: list[PartResult],
        detections: list[DetectionResult],
        decisions: list[OperatorDecision],
    ) -> None:
        """Store all data for one time step."""
        for part, det_result in zip(parts, detections):
            for decision in decisions:
                if decision.part_id != part.part_id:
                    continue
                self._records.append(
                    {
                        "step": time.step,
                        "shift_hour": time.shift_hour,
                        "part_id": part.part_id,
                        "total_defects": part.total_defect_count,
                        "detections_count": len(det_result.detections),
                        "missed_count": len(det_result.missed),
                        "false_alarms": det_result.false_alarms,
                        "operator_id": decision.operator_id,
                        "action": decision.action,
                        "correct": decision.correct,
                        "response_time": decision.response_time,
                        "trust_level": decision.trust_level,
                        "fatigue_level": decision.fatigue_level,
                        "confidence": decision.detection.confidence,
                        "is_true_positive": decision.detection.defect_ref is not None,
                    }
                )

        # Also record parts with no operator decisions (e.g., no flagged detections)
        decided_parts = {d.part_id for d in decisions}
        for part, det_result in zip(parts, detections):
            if part.part_id not in decided_parts:
                self._records.append(
                    {
                        "step": time.step,
                        "shift_hour": time.shift_hour,
                        "part_id": part.part_id,
                        "total_defects": part.total_defect_count,
                        "detections_count": len(det_result.detections),
                        "missed_count": len(det_result.missed),
                        "false_alarms": det_result.false_alarms,
                        "operator_id": None,
                        "action": None,
                        "correct": None,
                        "response_time": None,
                        "trust_level": None,
                        "fatigue_level": None,
                        "confidence": None,
                        "is_true_positive": None,
                    }
                )

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten all recorded data to a pandas DataFrame."""
        return pd.DataFrame(self._records)

    def summary(self) -> dict[str, Any]:
        """Compute aggregate metrics."""
        df = self.to_dataframe()
        if df.empty:
            return {}

        decided = df.dropna(subset=["action"])

        result: dict[str, Any] = {
            "total_parts": df["part_id"].nunique(),
            "total_steps": int(df["step"].max()) + 1 if len(df) > 0 else 0,
            "total_defects": int(df.groupby("part_id")["total_defects"].first().sum()),
        }

        if not decided.empty:
            result["operator_accuracy"] = float(decided["correct"].mean())
            result["mean_response_time"] = float(decided["response_time"].mean())
            result["mean_trust"] = float(decided["trust_level"].mean())
            result["mean_fatigue"] = float(decided["fatigue_level"].mean())

            # Action distribution
            action_counts = decided["action"].value_counts().to_dict()
            result["action_distribution"] = action_counts

            # Detection performance
            total_detected = int(df.groupby("part_id")["detections_count"].first().sum())
            total_missed = int(df.groupby("part_id")["missed_count"].first().sum())
            total_false = int(df.groupby("part_id")["false_alarms"].first().sum())
            result["total_detections"] = total_detected
            result["total_missed"] = total_missed
            result["total_false_alarms"] = total_false

            if total_detected + total_missed > 0:
                result["detection_recall"] = total_detected / (total_detected + total_missed)

        return result
