"""Tests for the operators module."""

import numpy as np
import pytest

from hawks.config import PopulationConfig
from hawks.operators.decision import DecisionModel
from hawks.operators.operator import HumanOperator
from hawks.operators.population import ARCHETYPES, OperatorPopulation
from hawks.operators.trust import TrustModel
from hawks.types import DefectInstance, Detection, TimeStep


class TestTrustModel:
    def test_trust_increases_on_correct(self):
        model = TrustModel(initial_trust=0.5, alpha=0.10, beta=0.10)
        model.update(ai_was_correct=True)
        assert model.get_trust() > 0.5

    def test_trust_decreases_on_incorrect(self):
        model = TrustModel(initial_trust=0.5, alpha=0.10, beta=0.10)
        model.update(ai_was_correct=False)
        assert model.get_trust() < 0.5

    def test_asymmetric_rates(self):
        model_fast_up = TrustModel(initial_trust=0.5, alpha=0.20, beta=0.05)
        model_fast_down = TrustModel(initial_trust=0.5, alpha=0.05, beta=0.20)

        model_fast_up.update(ai_was_correct=True)
        model_fast_down.update(ai_was_correct=True)
        assert model_fast_up.get_trust() > model_fast_down.get_trust()

        model_fast_up2 = TrustModel(initial_trust=0.5, alpha=0.20, beta=0.05)
        model_fast_down2 = TrustModel(initial_trust=0.5, alpha=0.05, beta=0.20)
        model_fast_up2.update(ai_was_correct=False)
        model_fast_down2.update(ai_was_correct=False)
        assert model_fast_down2.get_trust() < model_fast_up2.get_trust()

    def test_trust_bounded(self):
        model = TrustModel(initial_trust=0.99, alpha=0.10, beta=0.10)
        for _ in range(100):
            model.update(ai_was_correct=True)
        assert model.get_trust() <= 1.0

        model = TrustModel(initial_trust=0.01, alpha=0.10, beta=0.10)
        for _ in range(100):
            model.update(ai_was_correct=False)
        assert model.get_trust() >= 0.0

    def test_history_tracking(self):
        model = TrustModel(initial_trust=0.5, alpha=0.10, beta=0.10)
        model.update(ai_was_correct=True)
        model.update(ai_was_correct=False)
        history = model.get_history()
        assert len(history) == 3  # initial + 2 updates
        assert history[0] == 0.5

    def test_history_is_a_copy(self):
        model = TrustModel(initial_trust=0.5, alpha=0.10, beta=0.10)
        history = model.get_history()
        history.append(999.0)
        assert len(model.get_history()) == 1  # original unaffected


class TestDecisionModel:
    def _make_detection(self, confidence=0.8, has_defect=True):
        defect = DefectInstance("porosity", 0.5, (0, 100)) if has_defect else None
        return Detection(defect_ref=defect, confidence=confidence, flagged=True)

    def test_returns_accept_or_reject_only(self):
        rng = np.random.default_rng(42)
        model = DecisionModel(
            trust_weight=2.0, confidence_weight=1.5,
            risk_tolerance=1.0, decision_noise=0.2, rng=rng,
        )
        detection = self._make_detection()
        actions = set()
        for _ in range(100):
            action = model.decide(detection, trust=0.5, structural_risk=0.3)
            actions.add(action)
        assert actions <= {"accept", "reject"}
        assert "escalate" not in actions

    def test_no_escalate(self):
        rng = np.random.default_rng(0)
        model = DecisionModel(
            trust_weight=2.0, confidence_weight=1.5,
            risk_tolerance=1.0, decision_noise=0.2, rng=rng,
        )
        detection = self._make_detection(confidence=0.5)
        for _ in range(200):
            assert model.decide(detection, trust=0.5, structural_risk=0.0) != "escalate"

    def test_high_trust_and_confidence_favors_accept(self):
        rng = np.random.default_rng(42)
        model = DecisionModel(
            trust_weight=2.0, confidence_weight=1.5,
            risk_tolerance=1.0, decision_noise=0.1, rng=rng,
        )
        detection = self._make_detection(confidence=0.95)
        accepts = sum(
            1 for _ in range(500)
            if model.decide(detection, trust=0.95, structural_risk=0.8) == "accept"
        )
        assert accepts > 400  # should strongly favor accept

    def test_deterministic_with_seed(self):
        def run_decisions(seed):
            rng = np.random.default_rng(seed)
            model = DecisionModel(
                trust_weight=2.0, confidence_weight=1.5,
                risk_tolerance=1.0, decision_noise=0.2, rng=rng,
            )
            detection = Detection(
                defect_ref=DefectInstance("porosity", 0.5, (0, 100)),
                confidence=0.8, flagged=True,
            )
            return [model.decide(detection, trust=0.5, structural_risk=0.3) for _ in range(20)]

        assert run_decisions(123) == run_decisions(123)


class TestHumanOperator:
    def _make_operator(self, seed=42):
        rng = np.random.default_rng(seed)
        return HumanOperator(
            operator_id="OP-001",
            trust_model=TrustModel(initial_trust=0.5, alpha=0.10, beta=0.10),
            decision_model=DecisionModel(
                trust_weight=2.0, confidence_weight=1.5,
                risk_tolerance=1.0, decision_noise=0.2, rng=rng,
            ),
            rng=rng,
        )

    def test_review_returns_valid_decision(self):
        op = self._make_operator()
        defect = DefectInstance("porosity", 0.5, (0, 100))
        detection = Detection(defect_ref=defect, confidence=0.8, flagged=True)
        time = TimeStep(step=0, shift_hour=1.0)

        decision = op.review_detection(detection, "PART-000001", time)
        assert decision.operator_id == "OP-001"
        assert decision.part_id == "PART-000001"
        assert decision.action in ("accept", "reject")
        assert decision.response_time > 0

    def test_fatigue_always_zero(self):
        op = self._make_operator()
        assert op.fatigue == 0.0
        for i in range(20):
            op.advance_time(TimeStep(step=i, shift_hour=float(i % 8)))
        assert op.fatigue == 0.0

    def test_trust_history_accessible(self):
        op = self._make_operator()
        detection = Detection(
            defect_ref=DefectInstance("porosity", 0.5, (0, 100)),
            confidence=0.8, flagged=True,
        )
        time = TimeStep(step=0, shift_hour=1.0)
        decision = op.review_detection(detection, "PART-000001", time)
        op.receive_feedback(decision, ground_truth=True)
        assert len(op.trust_history) >= 2

    def test_structural_risk_defaults_to_zero(self):
        op = self._make_operator()
        detection = Detection(
            defect_ref=DefectInstance("porosity", 0.5, (0, 100)),
            confidence=0.8, flagged=True,
        )
        time = TimeStep(step=0, shift_hour=1.0)
        # Should work without structural_risk kwarg
        decision = op.review_detection(detection, "PART-000001", time)
        assert decision.action in ("accept", "reject")


class TestArchetypes:
    def test_all_archetypes_have_required_keys(self):
        required = {"initial_trust", "alpha", "beta", "risk_tolerance",
                     "trust_weight", "confidence_weight", "decision_noise"}
        for name, params in ARCHETYPES.items():
            assert set(params.keys()) == required, f"{name} missing keys"

    def test_values_in_valid_ranges(self):
        for name, params in ARCHETYPES.items():
            assert 0.0 <= params["initial_trust"] <= 1.0, f"{name} initial_trust"
            assert 0.0 < params["alpha"] <= 1.0, f"{name} alpha"
            assert 0.0 < params["beta"] <= 1.0, f"{name} beta"
            assert params["risk_tolerance"] >= 0.0, f"{name} risk_tolerance"
            assert params["trust_weight"] > 0.0, f"{name} trust_weight"
            assert params["confidence_weight"] > 0.0, f"{name} confidence_weight"
            assert params["decision_noise"] > 0.0, f"{name} decision_noise"


class TestOperatorPopulation:
    def test_create_from_archetypes(self):
        config = PopulationConfig(
            num_operators=10,
            archetype_mix={
                "conservative_skeptic": 3,
                "calibrated_professional": 4,
                "automation_biased": 2,
                "algorithm_averse": 1,
            },
        )
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)
        pop.create_population()
        assert len(pop.get_all_operators()) == 10

    def test_default_creates_calibrated_professionals(self):
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

    def test_unknown_archetype_raises(self):
        config = PopulationConfig(
            num_operators=1,
            archetype_mix={"nonexistent_type": 1},
        )
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)

        with pytest.raises(ValueError, match="Unknown archetype"):
            pop.create_population()

    def test_get_trust_histories(self):
        config = PopulationConfig(num_operators=3)
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)
        pop.create_population()

        histories = pop.get_trust_histories()
        assert len(histories) == 3
        for op_id, hist in histories.items():
            assert len(hist) >= 1  # at least initial value

    def test_parameter_noise_creates_heterogeneity(self):
        config = PopulationConfig(
            num_operators=10,
            parameter_noise={"initial_trust": 0.1},
        )
        rng = np.random.default_rng(42)
        pop = OperatorPopulation(config, rng)
        pop.create_population()

        trusts = [op.trust for op in pop.get_all_operators()]
        assert len(set(trusts)) > 1  # heterogeneous

    def test_seed_reproducibility(self):
        config = PopulationConfig(num_operators=5)

        rng1 = np.random.default_rng(99)
        pop1 = OperatorPopulation(config, rng1)
        pop1.create_population()
        trusts1 = [op.trust for op in pop1.get_all_operators()]

        rng2 = np.random.default_rng(99)
        pop2 = OperatorPopulation(config, rng2)
        pop2.create_population()
        trusts2 = [op.trust for op in pop2.get_all_operators()]

        assert trusts1 == trusts2
