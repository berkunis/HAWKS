"""Human operator — a single operator with evolving internal state."""

from __future__ import annotations

import numpy as np

from hawks.operators.decision import DecisionModel
from hawks.operators.fatigue import FatigueModel
from hawks.operators.trust import TrustModel
from hawks.types import Detection, OperatorDecision, TimeStep


class HumanOperator:
    """A single human operator composed from strategy models.

    Each operator has independent trust, fatigue, and decision models
    that evolve over the course of a simulation.
    """

    def __init__(
        self,
        operator_id: str,
        trust_model: TrustModel,
        fatigue_model: FatigueModel,
        decision_model: DecisionModel,
        skill_level: float,
        rng: np.random.Generator,
    ) -> None:
        self.operator_id = operator_id
        self._trust_model = trust_model
        self._fatigue_model = fatigue_model
        self._decision_model = decision_model
        self._skill_level = skill_level
        self._rng = rng

    def review_detection(
        self, detection: Detection, part_id: str, time: TimeStep
    ) -> OperatorDecision:
        """Produce an accept/reject/escalate decision for a flagged detection."""
        trust = self._trust_model.get_trust()
        fatigue = self._fatigue_model.get_fatigue()

        action = self._decision_model.decide(
            detection, trust, fatigue, self._skill_level
        )

        # Determine correctness vs ground truth
        has_real_defect = detection.defect_ref is not None
        if action == "accept":
            correct = has_real_defect  # correct to accept if there IS a defect
        elif action == "reject":
            correct = not has_real_defect  # correct to reject if NO real defect
        else:  # escalate
            correct = True  # escalation is always "safe"

        # Simulate response time (fatigue increases response time)
        base_time = 2.0 + self._rng.exponential(1.0)
        response_time = base_time * (1.0 + 0.5 * fatigue)

        return OperatorDecision(
            operator_id=self.operator_id,
            part_id=part_id,
            detection=detection,
            action=action,
            correct=correct,
            response_time=response_time,
            trust_level=trust,
            fatigue_level=fatigue,
        )

    def receive_feedback(self, decision: OperatorDecision, ground_truth: bool) -> None:
        """Update trust based on whether the AI was correct."""
        # AI was "correct" if it flagged a real defect or didn't flag a non-defect
        ai_flagged = decision.detection.flagged
        ai_was_correct = ai_flagged == ground_truth
        self._trust_model.update(ai_was_correct)

    def advance_time(self, time: TimeStep) -> None:
        """Accumulate fatigue and vigilance decay."""
        self._fatigue_model.accumulate(time)

    def rest(self) -> None:
        """Reset fatigue on shift change."""
        self._fatigue_model.rest()

    @property
    def trust(self) -> float:
        return self._trust_model.get_trust()

    @property
    def fatigue(self) -> float:
        return self._fatigue_model.get_fatigue()
