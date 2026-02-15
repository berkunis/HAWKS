"""Adaptive AI detector with trust-dependent accuracy."""

from __future__ import annotations

import numpy as np


class AdaptiveAIDetector:
    """AI detector with trust-dependent accuracy.

    p_t = clamp(p_base + gamma * (T_t - 0.5), min_p, max_p)
    """

    def __init__(
        self,
        p_base: float,
        gamma: float,
        rng: np.random.Generator,
        min_p: float = 0.01,
        max_p: float = 0.99,
    ) -> None:
        self._p_base = p_base
        self._gamma = gamma
        self._rng = rng
        self._min_p = min_p
        self._max_p = max_p

    def effective_accuracy(self, trust: float) -> float:
        """Compute p_t given current trust. Pure, no side effects."""
        p_t = self._p_base + self._gamma * (trust - 0.5)
        return float(np.clip(p_t, self._min_p, self._max_p))

    def sample_correctness(self, trust: float) -> bool:
        """Draw Bernoulli(p_t). Single RNG call per invocation."""
        p_t = self.effective_accuracy(trust)
        return bool(self._rng.random() < p_t)
