"""Decision model — sigmoid-based binary accept/reject."""

from __future__ import annotations

import numpy as np

from hawks.types import Detection


class DecisionModel:
    """Sigmoid-based binary decision model.

    Computes a logit from trust, AI confidence, structural risk, and noise,
    then draws a Bernoulli sample to produce accept or reject.
    """

    def __init__(
        self,
        trust_weight: float,
        confidence_weight: float,
        risk_tolerance: float,
        decision_noise: float,
        rng: np.random.Generator,
    ) -> None:
        self._trust_weight = trust_weight
        self._confidence_weight = confidence_weight
        self._risk_tolerance = risk_tolerance
        self._decision_noise = decision_noise
        self._rng = rng

    def decide(
        self, detection: Detection, trust: float, structural_risk: float
    ) -> str:
        """Produce accept/reject decision.

        Returns:
            "accept" — agree with AI's flagging
            "reject" — override AI, declare no defect
        """
        noise = self._rng.normal(0.0, self._decision_noise)
        logit = (
            self._trust_weight * trust
            + self._confidence_weight * detection.confidence
            + self._risk_tolerance * structural_risk
            + noise
        )
        p_accept = 1.0 / (1.0 + np.exp(-logit))

        if self._rng.random() < p_accept:
            return "accept"
        return "reject"
