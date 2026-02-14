"""Decision model strategy — probabilistic accept/reject based on operator state."""

from __future__ import annotations

import numpy as np

from hawks.types import Detection


class DecisionModel:
    """Combines trust, fatigue, confidence, and automation bias into a decision.

    Higher trust and confidence → more likely to accept AI's assessment.
    Higher fatigue → more likely to default to AI (automation bias).
    """

    def __init__(
        self,
        skill_level: float,
        automation_bias_strength: float,
        rng: np.random.Generator,
    ) -> None:
        self._skill = max(0.0, min(1.0, skill_level))
        self._automation_bias = automation_bias_strength
        self._rng = rng

    def decide(
        self, detection: Detection, trust: float, fatigue: float, skill: float
    ) -> str:
        """Produce accept/reject/escalate decision.

        Returns:
            "accept" — agree with AI's flagging
            "reject" — override AI, declare no defect
            "escalate" — uncertain, send to supervisor
        """
        confidence = detection.confidence

        # Probability of accepting AI's assessment
        # High trust, high confidence, high fatigue → accept
        # Automation bias increases with fatigue
        effective_bias = self._automation_bias * (1.0 + fatigue)
        p_accept = (
            0.3 * trust
            + 0.3 * confidence
            + 0.2 * effective_bias
            + 0.2 * (1.0 - skill)  # lower skill → more reliance on AI
        )
        p_accept = max(0.0, min(1.0, p_accept))

        # Probability of escalating (uncertainty)
        uncertainty = abs(confidence - 0.5) < 0.15
        p_escalate = 0.15 * skill if uncertainty else 0.03

        roll = self._rng.random()
        if roll < p_escalate:
            return "escalate"
        elif roll < p_escalate + p_accept:
            return "accept"
        else:
            return "reject"
