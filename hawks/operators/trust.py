"""Trust model strategy — Bayesian trust update."""

from __future__ import annotations


class TrustModel:
    """Bayesian trust update model.

    Trust increases after correct AI alerts and decreases after false alarms.
    Uses a simple exponential moving average approach.
    """

    def __init__(self, initial_trust: float, learning_rate: float = 0.1) -> None:
        self._trust = max(0.0, min(1.0, initial_trust))
        self._learning_rate = learning_rate

    def update(self, ai_was_correct: bool) -> None:
        """Update trust based on whether the AI's assessment was correct."""
        if ai_was_correct:
            # Trust increases toward 1.0
            self._trust += self._learning_rate * (1.0 - self._trust)
        else:
            # Trust decreases toward 0.0
            self._trust -= self._learning_rate * self._trust

        self._trust = max(0.0, min(1.0, self._trust))

    def get_trust(self) -> float:
        """Return current trust level."""
        return self._trust
