"""Trust model — asymmetric alpha/beta trust update with history."""

from __future__ import annotations


class TrustModel:
    """Asymmetric trust update model.

    Trust increases after correct AI assessments (rate alpha) and
    decreases after incorrect assessments (rate beta).
    """

    def __init__(self, initial_trust: float, alpha: float, beta: float) -> None:
        self._trust = max(0.0, min(1.0, initial_trust))
        self._alpha = alpha
        self._beta = beta
        self._history: list[float] = [self._trust]

    def update(self, ai_was_correct: bool) -> None:
        """Update trust based on whether the AI's assessment was correct."""
        if ai_was_correct:
            self._trust += self._alpha * (1.0 - self._trust)
        else:
            self._trust -= self._beta * self._trust

        self._trust = max(0.0, min(1.0, self._trust))
        self._history.append(self._trust)

    def get_trust(self) -> float:
        """Return current trust level."""
        return self._trust

    def get_history(self) -> list[float]:
        """Return a copy of the trust history."""
        return list(self._history)
