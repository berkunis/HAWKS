"""Tests for the operators module."""

import numpy as np
import pytest

from hawks.config import OperatorConfig, PopulationConfig
from hawks.operators.decision import DecisionModel
from hawks.operators.fatigue import FatigueModel
from hawks.operators.operator import HumanOperator
from hawks.operators.population import OperatorPopulation
from hawks.operators.trust import TrustModel
from hawks.types import DefectInstance, Detection, TimeStep


class TestTrustModel:
    def test_trust_increases_on_correct(self):
        model = TrustModel(initial_trust=0.5)
        model.update(ai_was_correct=True)
        assert model.get_trust() > 0.5

    def test_trust_decreases_on_incorrect(self):
        model = TrustModel(initial_trust=0.5)
        model.update(ai_was_correct=False)
        assert model.get_trust() < 0.5

    def test_trust_bounded(self):
        model = TrustModel(initial_trust=0.99)
        for _ in range(100):
            model.update(ai_was_correct=True)
        assert model.get_trust() <= 1.0

        model = TrustModel(initial_trust=0.01)
        for _ in range(100):
            model.update(ai_was_correct=False)
        assert model.get_trust() >= 0.0


class TestFatigueModel:
    def test_fatigue_increases_over_time(self):
        model = FatigueModel(fatigue_rate=0.05, vigilance_decrement_rate=0.01)
        ts = TimeStep(step=0, shift_hour=1.0)
        model.accumulate(ts)
        assert model.get_fatigue() > 0.0

    def test_rest_resets_fatigue(self):
        model = FatigueModel(fatigue_rate=0.05, vigilance_decrement_rate=0.01)
        for i in range(20):
            model.accumulate(TimeStep(step=i, shift_hour=float(i)))
        assert model.get_fatigue() > 0.0
        model.rest()
        assert model.get_fatigue() == 0.0

    def test_fatigue_bounded(self):
        model = FatigueModel(fatigue_rate=0.1, vigilance_decrement_rate=0.1)
        for i in range(1000):
            model.accumulate(TimeStep(step=i, shift_hour=float(i)))
        assert model.get_fatigue() <= 1.0


class TestDecisionModel:
    def test_returns_valid_action(self):
        rng = np.random.default_rng(42)
        model = DecisionModel(skill_level=0.7, automation_bias_strength=0.3, rng=rng)

        defect = DefectInstance("porosity", 0.5, (0, 100))
        detection = Detection(defect_ref=defect, confidence=0.8, flagged=True)

        for _ in range(50):
            action = model.decide(detection, trust=0.5, fatigue=0.2, skill=0.7)
            assert action in ("accept", "reject", "escalate")


class TestHumanOperator:
    def _make_operator(self, seed=42):
        rng = np.random.default_rng(seed)
        return HumanOperator(
            operator_id="OP-001",
            trust_model=TrustModel(initial_trust=0.5),
            fatigue_model=FatigueModel(fatigue_rate=0.01, vigilance_decrement_rate=0.005),
            decision_model=DecisionModel(skill_level=0.7, automation_bias_strength=0.3, rng=rng),
            skill_level=0.7,
            rng=rng,
        )

    def test_review_detection_returns_decision(self):
        op = self._make_operator()
        defect = DefectInstance("porosity", 0.5, (0, 100))
        detection = Detection(defect_ref=defect, confidence=0.8, flagged=True)
        time = TimeStep(step=0, shift_hour=1.0)

        decision = op.review_detection(detection, "PART-000001", time)
        assert decision.operator_id == "OP-001"
        assert decision.part_id == "PART-000001"
        assert decision.action in ("accept", "reject", "escalate")
        assert decision.response_time > 0

    def test_fatigue_increases_with_advance_time(self):
        op = self._make_operator()
        assert op.fatigue == 0.0
        for i in range(20):
            op.advance_time(TimeStep(step=i, shift_hour=float(i % 8)))
        assert op.fatigue > 0.0


class TestOperatorPopulation:
    def test_create_population(self):
        config = PopulationConfig(num_operators=5)
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)
        pop.create_population()

        assert len(pop.get_all_operators()) == 5

    def test_round_robin_assignment(self):
        config = PopulationConfig(num_operators=3, assignment_policy="round_robin")
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)
        pop.create_population()

        ts = TimeStep(step=0, shift_hour=0.0)
        ids = [pop.assign_operator(ts).operator_id for _ in range(6)]
        assert ids[0] == ids[3]
        assert ids[1] == ids[4]
        assert ids[2] == ids[5]

    def test_assign_without_creation_raises(self):
        config = PopulationConfig(num_operators=3)
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)

        with pytest.raises(RuntimeError):
            pop.assign_operator(TimeStep(step=0, shift_hour=0.0))

    def test_operators_have_different_params(self):
        config = PopulationConfig(num_operators=10)
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)
        pop.create_population()

        trusts = [op.trust for op in pop.get_all_operators()]
        # Not all the same (heterogeneous)
        assert len(set(trusts)) > 1
