"""Operator population — archetype-based factory + collection manager."""

from __future__ import annotations

import numpy as np

from hawks.config import PopulationConfig
from hawks.operators.decision import DecisionModel
from hawks.operators.operator import HumanOperator
from hawks.operators.trust import TrustModel
from hawks.types import TimeStep

ARCHETYPES: dict[str, dict[str, float]] = {
    "conservative_skeptic": {
        "initial_trust": 0.30,
        "alpha": 0.05,
        "beta": 0.20,
        "risk_tolerance": 2.0,
        "trust_weight": 1.5,
        "confidence_weight": 1.0,
        "decision_noise": 0.3,
    },
    "calibrated_professional": {
        "initial_trust": 0.50,
        "alpha": 0.10,
        "beta": 0.10,
        "risk_tolerance": 1.0,
        "trust_weight": 2.0,
        "confidence_weight": 1.5,
        "decision_noise": 0.2,
    },
    "automation_biased": {
        "initial_trust": 0.80,
        "alpha": 0.15,
        "beta": 0.03,
        "risk_tolerance": 0.3,
        "trust_weight": 3.0,
        "confidence_weight": 2.5,
        "decision_noise": 0.15,
    },
    "algorithm_averse": {
        "initial_trust": 0.45,
        "alpha": 0.03,
        "beta": 0.25,
        "risk_tolerance": 0.5,
        "trust_weight": 1.5,
        "confidence_weight": 0.8,
        "decision_noise": 0.35,
    },
}


class OperatorPopulation:
    """Factory and collection manager for archetype-based human operators."""

    def __init__(self, config: PopulationConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._operators: list[HumanOperator] = []
        self._assignment_index = 0

    def create_population(self) -> None:
        """Instantiate operators from archetype definitions."""
        cfg = self._config
        mix = cfg.archetype_mix

        if mix is None:
            # Default: all calibrated_professional
            assignments = ["calibrated_professional"] * cfg.num_operators
        else:
            assignments = []
            for archetype_name, count in mix.items():
                if archetype_name not in ARCHETYPES:
                    raise ValueError(
                        f"Unknown archetype '{archetype_name}'. "
                        f"Valid archetypes: {list(ARCHETYPES.keys())}"
                    )
                assignments.extend([archetype_name] * count)

        for i, archetype_name in enumerate(assignments):
            params = dict(ARCHETYPES[archetype_name])

            # Apply parameter noise if configured
            if cfg.parameter_noise:
                for param, noise_std in cfg.parameter_noise.items():
                    if param in params:
                        noisy = self._rng.normal(params[param], noise_std)
                        if param in ("initial_trust", "alpha", "beta"):
                            noisy = float(np.clip(noisy, 0.0, 1.0))
                        elif param == "decision_noise":
                            noisy = max(0.01, noisy)
                        params[param] = float(noisy)

            trust_model = TrustModel(
                initial_trust=params["initial_trust"],
                alpha=params["alpha"],
                beta=params["beta"],
            )

            op_rng = np.random.default_rng(self._rng.integers(0, 2**32))

            decision_model = DecisionModel(
                trust_weight=params["trust_weight"],
                confidence_weight=params["confidence_weight"],
                risk_tolerance=params["risk_tolerance"],
                decision_noise=params["decision_noise"],
                rng=op_rng,
            )

            operator = HumanOperator(
                operator_id=f"OP-{i + 1:03d}",
                trust_model=trust_model,
                decision_model=decision_model,
                rng=op_rng,
            )
            self._operators.append(operator)

    def assign_operator(self, time: TimeStep) -> HumanOperator:
        """Select next operator per assignment policy."""
        if not self._operators:
            raise RuntimeError("Population not created. Call create_population() first.")

        policy = self._config.assignment_policy
        if policy == "round_robin":
            op = self._operators[self._assignment_index % len(self._operators)]
            self._assignment_index += 1
            return op
        elif policy == "random":
            return self._operators[int(self._rng.integers(0, len(self._operators)))]
        else:
            raise ValueError(f"Unknown assignment policy: {policy}")

    def get_all_operators(self) -> list[HumanOperator]:
        """Return all operators for population-level analysis."""
        return list(self._operators)

    def get_trust_histories(self) -> dict[str, list[float]]:
        """Return trust histories for all operators."""
        return {op.operator_id: op.trust_history for op in self._operators}

    def handle_shift_change(self) -> None:
        """Rest all operators on shift change."""
        for op in self._operators:
            op.rest()
