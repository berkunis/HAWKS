"""Operator population — factory + collection manager."""

from __future__ import annotations

import numpy as np

from hawks.config import PopulationConfig
from hawks.operators.decision import DecisionModel
from hawks.operators.fatigue import FatigueModel
from hawks.operators.operator import HumanOperator
from hawks.operators.trust import TrustModel
from hawks.types import TimeStep


class OperatorPopulation:
    """Factory and collection manager for heterogeneous human operators.

    Generates operators with individual parameters sampled from configured
    distributions, and handles operator assignment each time step.
    """

    def __init__(self, config: PopulationConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._operators: list[HumanOperator] = []
        self._assignment_index = 0

    def create_population(self) -> None:
        """Instantiate heterogeneous operators by sampling from distributions."""
        cfg = self._config
        op_cfg = cfg.operator_defaults

        for i in range(cfg.num_operators):
            # Sample individual parameters from distributions
            trust_mean, trust_std = op_cfg.trust_initial
            initial_trust = float(np.clip(self._rng.normal(trust_mean, trust_std), 0.0, 1.0))

            skill_mean, skill_std = op_cfg.skill_level
            skill = float(np.clip(self._rng.normal(skill_mean, skill_std), 0.0, 1.0))

            # Small variation in fatigue rate
            fatigue_rate = max(0.001, self._rng.normal(op_cfg.fatigue_rate, op_cfg.fatigue_rate * 0.2))

            # Compose strategy models
            trust_model = TrustModel(initial_trust)
            fatigue_model = FatigueModel(fatigue_rate, op_cfg.vigilance_decrement_rate)

            # Each operator gets its own derived RNG
            op_rng = np.random.default_rng(self._rng.integers(0, 2**32))
            decision_model = DecisionModel(skill, op_cfg.automation_bias_strength, op_rng)

            operator = HumanOperator(
                operator_id=f"OP-{i + 1:03d}",
                trust_model=trust_model,
                fatigue_model=fatigue_model,
                decision_model=decision_model,
                skill_level=skill,
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
        elif policy == "skill_based":
            # Pick the operator with highest skill who is least fatigued
            return min(self._operators, key=lambda o: o.fatigue)
        else:
            raise ValueError(f"Unknown assignment policy: {policy}")

    def get_all_operators(self) -> list[HumanOperator]:
        """Return all operators for population-level analysis."""
        return list(self._operators)

    def handle_shift_change(self) -> None:
        """Rest all operators on shift change."""
        for op in self._operators:
            op.rest()
