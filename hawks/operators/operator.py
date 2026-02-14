"""Human operator — a single operator with evolving internal state."""

from __future__ import annotations

import numpy as np

from hawks.operators.decision import DecisionModel
from hawks.operators.trust import TrustModel
from hawks.types import Detection, OperatorDecision, TimeStep


class HumanOperator:
    """A single human operator composed from trust and decision models.

    Each operator has independent trust and decision models
    that evolve over the course of a simulation.
    """

    def __init__(
        self,
        operator_id: str,
        trust_model: TrustModel,
        decision_model: DecisionModel,
        rng: np.random.Generator,
    ) -> None:
        self.operator_id = operator_id
        self._trust_model = trust_model
        self._decision_model = decision_model
        self._rng = rng

    def review_detection(
        self,
        detection: Detection,
        part_id: str,
        time: TimeStep,
        structural_risk: float = 0.0,
    ) -> OperatorDecision:
        """Produce an accept/reject decision for a flagged detection."""
        trust = self._trust_model.get_trust()

        action = self._decision_model.decide(detection, trust, structural_risk)

        # Determine correctness vs ground truth
        has_real_defect = detection.defect_ref is not None
        if action == "accept":
            correct = has_real_defect
        else:  # reject
            correct = not has_real_defect

        response_time = 2.0 + self._rng.exponential(1.0)

        return OperatorDecision(
            operator_id=self.operator_id,
            part_id=part_id,
            detection=detection,
            action=action,
            correct=correct,
            response_time=response_time,
            trust_level=trust,
            fatigue_level=0.0,
        )

    def receive_feedback(self, decision: OperatorDecision, ground_truth: bool) -> None:
        """Update trust based on whether the AI was correct."""
        ai_flagged = decision.detection.flagged
        ai_was_correct = ai_flagged == ground_truth
        self._trust_model.update(ai_was_correct)

    def advance_time(self, time: TimeStep) -> None:
        """No-op, kept for SimulationEngine compatibility."""

    def rest(self) -> None:
        """No-op, kept for SimulationEngine compatibility."""

    @property
    def trust(self) -> float:
        return self._trust_model.get_trust()

    @property
    def trust_history(self) -> list[float]:
        return self._trust_model.get_history()

    @property
    def fatigue(self) -> float:
        return 0.0
