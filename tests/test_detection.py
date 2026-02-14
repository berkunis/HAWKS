"""Tests for the AIModelSimulator detection module."""

import numpy as np
import pytest

from hawks.config import AIModelConfig, ManufacturingConfig
from hawks.detection.ai_model import AIModelSimulator, _collect_defects
from hawks.manufacturing.digital_twin import ManufacturingTwin
from hawks.manufacturing.physics_state import PrintTrace
from hawks.types import (
    DefectInstance,
    DetectionResult,
    LayerResult,
    ModelPrediction,
    PartResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_parts_and_traces(seed=42, risk_threshold=0.5, n_parts=5):
    """Produce parts and traces from the manufacturing twin."""
    config = ManufacturingConfig(risk_threshold=risk_threshold, num_layers_per_part=50)
    twin = ManufacturingTwin(config, np.random.default_rng(seed))
    parts = twin.produce_parts(n_parts)
    traces = twin.get_last_traces()
    return parts, traces


def _make_defective_part(part_id="PART-TEST", n_defects=3, risk=0.8):
    """Create a synthetic part with defects and matching trace."""
    defects = [
        DefectInstance(defect_type="porosity", severity=0.7, location=(i, 0))
        for i in range(n_defects)
    ]
    layers = [LayerResult(layer_index=0, defects=defects)]
    part = PartResult(part_id=part_id, layers=layers, total_defect_count=n_defects)
    trace = PrintTrace(part_id=part_id, max_structural_risk=risk, total_layers=1)
    return part, trace


def _make_clean_part(part_id="PART-CLEAN", risk=0.1):
    """Create a synthetic part with no defects and matching trace."""
    layers = [LayerResult(layer_index=0, defects=[])]
    part = PartResult(part_id=part_id, layers=layers, total_defect_count=0)
    trace = PrintTrace(part_id=part_id, max_structural_risk=risk, total_layers=1)
    return part, trace


# ── Contract Tests ───────────────────────────────────────────────────────

class TestPredictContract:
    def test_predict_returns_model_prediction(self):
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part()

        result = model.predict(part, trace)
        assert isinstance(result, ModelPrediction)

    def test_prediction_part_id_matches(self):
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part(part_id="PART-XYZ")

        result = model.predict(part, trace)
        assert result.part_id == "PART-XYZ"

    def test_predicted_probability_in_range(self):
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        parts, traces = _make_parts_and_traces(n_parts=20, risk_threshold=0.3)

        for part, trace in zip(parts, traces):
            pred = model.predict(part, trace)
            assert 0.0 <= pred.predicted_defect_probability <= 1.0

    def test_confidence_score_in_range(self):
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        parts, traces = _make_parts_and_traces(n_parts=20, risk_threshold=0.3)

        for part, trace in zip(parts, traces):
            pred = model.predict(part, trace)
            assert 0.0 <= pred.confidence_score <= 1.0

    def test_predicted_positive_is_bool(self):
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part()

        result = model.predict(part, trace)
        assert isinstance(result.predicted_positive, bool)

    def test_correct_is_bool(self):
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part()

        result = model.predict(part, trace)
        assert isinstance(result.correct, bool)


# ── TPR/FPR Behavior ────────────────────────────────────────────────────

class TestTPRFPRBehavior:
    def test_high_tpr_catches_more_defective_parts(self):
        """Higher TPR should flag more defective parts as positive."""
        part, trace = _make_defective_part()
        n_trials = 200

        config_low = AIModelConfig(true_positive_rate=0.3)
        config_high = AIModelConfig(true_positive_rate=0.95)

        caught_low = sum(
            AIModelSimulator(config_low, np.random.default_rng(i)).predict(part, trace).predicted_positive
            for i in range(n_trials)
        )
        caught_high = sum(
            AIModelSimulator(config_high, np.random.default_rng(i)).predict(part, trace).predicted_positive
            for i in range(n_trials)
        )

        assert caught_high > caught_low

    def test_low_fpr_fewer_false_alarms(self):
        """Lower FPR should produce fewer false positives on clean parts."""
        part, trace = _make_clean_part()
        n_trials = 200

        config_high_fpr = AIModelConfig(false_positive_rate=0.5)
        config_low_fpr = AIModelConfig(false_positive_rate=0.02)

        fp_high = sum(
            AIModelSimulator(config_high_fpr, np.random.default_rng(i)).predict(part, trace).predicted_positive
            for i in range(n_trials)
        )
        fp_low = sum(
            AIModelSimulator(config_low_fpr, np.random.default_rng(i)).predict(part, trace).predicted_positive
            for i in range(n_trials)
        )

        assert fp_low < fp_high


# ── Calibration ──────────────────────────────────────────────────────────

class TestCalibration:
    def test_positive_bias_increases_predicted_probability(self):
        """Positive calibration bias should increase predicted probability."""
        part, trace = _make_defective_part(risk=0.5)

        config_zero = AIModelConfig(calibration_bias=0.0)
        config_pos = AIModelConfig(calibration_bias=2.0)

        model_zero = AIModelSimulator(config_zero, np.random.default_rng(42))
        model_pos = AIModelSimulator(config_pos, np.random.default_rng(42))

        pred_zero = model_zero.predict(part, trace)
        pred_pos = model_pos.predict(part, trace)

        assert pred_pos.predicted_defect_probability > pred_zero.predicted_defect_probability

    def test_negative_bias_decreases_predicted_probability(self):
        """Negative calibration bias should decrease predicted probability."""
        part, trace = _make_defective_part(risk=0.5)

        config_zero = AIModelConfig(calibration_bias=0.0)
        config_neg = AIModelConfig(calibration_bias=-2.0)

        model_zero = AIModelSimulator(config_zero, np.random.default_rng(42))
        model_neg = AIModelSimulator(config_neg, np.random.default_rng(42))

        pred_zero = model_zero.predict(part, trace)
        pred_neg = model_neg.predict(part, trace)

        assert pred_neg.predicted_defect_probability < pred_zero.predicted_defect_probability


# ── Correctness Flag ─────────────────────────────────────────────────────

class TestCorrectnessFlag:
    def test_true_positive_is_correct(self):
        """Predicting positive on a defective part is correct."""
        config = AIModelConfig(true_positive_rate=1.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part()

        pred = model.predict(part, trace)
        assert pred.predicted_positive is True
        assert pred.correct is True

    def test_true_negative_is_correct(self):
        """Predicting negative on a clean part is correct."""
        config = AIModelConfig(false_positive_rate=0.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_clean_part()

        pred = model.predict(part, trace)
        assert pred.predicted_positive is False
        assert pred.correct is True

    def test_false_positive_is_incorrect(self):
        """Predicting positive on a clean part is incorrect."""
        config = AIModelConfig(false_positive_rate=1.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_clean_part()

        pred = model.predict(part, trace)
        assert pred.predicted_positive is True
        assert pred.correct is False

    def test_false_negative_is_incorrect(self):
        """Predicting negative on a defective part is incorrect."""
        config = AIModelConfig(true_positive_rate=0.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part()

        pred = model.predict(part, trace)
        assert pred.predicted_positive is False
        assert pred.correct is False


# ── Bridge Tests ─────────────────────────────────────────────────────────

class TestBridge:
    def test_true_positive_bridge(self):
        """TP: one detection per defect, all flagged, no missed, no false alarms."""
        config = AIModelConfig(true_positive_rate=1.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part(n_defects=3)

        pred, det = model.inspect_part(part, trace)
        assert pred.predicted_positive is True
        assert len(det.detections) == 3
        assert all(d.flagged for d in det.detections)
        assert all(d.defect_ref is not None for d in det.detections)
        assert len(det.missed) == 0
        assert det.false_alarms == 0

    def test_false_positive_bridge(self):
        """FP: one detection with no defect ref, false_alarms=1."""
        config = AIModelConfig(false_positive_rate=1.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_clean_part()

        pred, det = model.inspect_part(part, trace)
        assert pred.predicted_positive is True
        assert len(det.detections) == 1
        assert det.detections[0].defect_ref is None
        assert det.detections[0].flagged is True
        assert det.false_alarms == 1
        assert len(det.missed) == 0

    def test_false_negative_bridge(self):
        """FN: no detections, all defects missed."""
        config = AIModelConfig(true_positive_rate=0.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part(n_defects=2)

        pred, det = model.inspect_part(part, trace)
        assert pred.predicted_positive is False
        assert len(det.detections) == 0
        assert len(det.missed) == 2
        assert det.false_alarms == 0

    def test_true_negative_bridge(self):
        """TN: no detections, no missed, no false alarms."""
        config = AIModelConfig(false_positive_rate=0.0)
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_clean_part()

        pred, det = model.inspect_part(part, trace)
        assert pred.predicted_positive is False
        assert len(det.detections) == 0
        assert len(det.missed) == 0
        assert det.false_alarms == 0

    def test_bridge_returns_detection_result(self):
        """Bridge should return a DetectionResult instance."""
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part()

        pred, det = model.inspect_part(part, trace)
        assert isinstance(det, DetectionResult)
        assert det.part_id == part.part_id


# ── Presets ───────────────────────────────────────────────────────────────

class TestPresets:
    def test_well_calibrated_preset(self):
        config = AIModelConfig(preset="well_calibrated")
        assert config.true_positive_rate == 0.90
        assert config.false_positive_rate == 0.05
        assert config.calibration_bias == 0.0
        assert config.confidence_noise == 0.05

    def test_overconfident_preset(self):
        config = AIModelConfig(preset="overconfident")
        assert config.true_positive_rate == 0.85
        assert config.false_positive_rate == 0.10
        assert config.calibration_bias == 1.5
        assert config.confidence_noise == 0.02

    def test_underconfident_preset(self):
        config = AIModelConfig(preset="underconfident")
        assert config.true_positive_rate == 0.90
        assert config.false_positive_rate == 0.05
        assert config.calibration_bias == -0.5
        assert config.confidence_noise == 0.15

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            AIModelConfig(preset="invalid_preset")


# ── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_risk_zero_no_math_error(self):
        """Risk of 0.0 should not cause log(0) errors."""
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_clean_part(risk=0.0)

        pred = model.predict(part, trace)
        assert 0.0 <= pred.predicted_defect_probability <= 1.0
        assert np.isfinite(pred.predicted_defect_probability)

    def test_risk_one_no_math_error(self):
        """Risk of 1.0 should not cause log(inf) errors."""
        config = AIModelConfig()
        model = AIModelSimulator(config, np.random.default_rng(42))
        part, trace = _make_defective_part(risk=1.0)

        pred = model.predict(part, trace)
        assert 0.0 <= pred.predicted_defect_probability <= 1.0
        assert np.isfinite(pred.predicted_defect_probability)

    def test_zero_noise_deterministic_confidence(self):
        """With noise=0, confidence should be deterministic across seeds."""
        config = AIModelConfig(confidence_noise=0.0)
        part, trace = _make_defective_part(risk=0.7)

        preds = [
            AIModelSimulator(config, np.random.default_rng(i)).predict(part, trace)
            for i in range(10)
        ]
        # All confidence scores should be the same (noise draw is N(0,0)=0)
        confidences = {p.confidence_score for p in preds}
        assert len(confidences) == 1

    def test_collect_defects_helper(self):
        """_collect_defects flattens defects from all layers."""
        part, _ = _make_defective_part(n_defects=3)
        defects = _collect_defects(part)
        assert len(defects) == 3

    def test_collect_defects_empty(self):
        """_collect_defects returns empty list for clean parts."""
        part, _ = _make_clean_part()
        defects = _collect_defects(part)
        assert len(defects) == 0


# ── Reproducibility ──────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_identical_results(self):
        """Same seed should produce identical predictions."""
        config = AIModelConfig()
        part, trace = _make_defective_part()

        pred1 = AIModelSimulator(config, np.random.default_rng(99)).predict(part, trace)
        pred2 = AIModelSimulator(config, np.random.default_rng(99)).predict(part, trace)

        assert pred1.predicted_positive == pred2.predicted_positive
        assert pred1.predicted_defect_probability == pred2.predicted_defect_probability
        assert pred1.confidence_score == pred2.confidence_score
        assert pred1.correct == pred2.correct

    def test_different_seeds_may_differ(self):
        """Different seeds should (usually) produce different results over many parts."""
        config = AIModelConfig()
        parts, traces = _make_parts_and_traces(n_parts=20, risk_threshold=0.3)

        results_a = [
            AIModelSimulator(config, np.random.default_rng(1)).predict(p, t)
            for p, t in zip(parts, traces)
        ]
        results_b = [
            AIModelSimulator(config, np.random.default_rng(2)).predict(p, t)
            for p, t in zip(parts, traces)
        ]

        # At least some confidence scores should differ
        any_differ = any(
            a.confidence_score != b.confidence_score
            for a, b in zip(results_a, results_b)
        )
        assert any_differ
