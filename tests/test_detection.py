"""Tests for the detection module."""

import numpy as np
import pytest

from hawks.config import DetectionConfig, ManufacturingConfig
from hawks.detection.detector import AIDefectDetector
from hawks.manufacturing.digital_twin import ManufacturingTwin


class TestAIDefectDetector:
    def _make_parts(self, seed=42, risk_threshold=0.5, n_parts=5):
        config = ManufacturingConfig(risk_threshold=risk_threshold, num_layers_per_part=50)
        twin = ManufacturingTwin(config, np.random.default_rng(seed))
        return twin.produce_parts(n_parts)

    def test_inspect_part_returns_detection_result(self):
        det_config = DetectionConfig()
        detector = AIDefectDetector(det_config, np.random.default_rng(42))
        parts = self._make_parts()

        result = detector.inspect_part(parts[0])
        assert result.part_id == parts[0].part_id

    def test_high_sensitivity_catches_more(self):
        """Higher sensitivity should yield fewer missed defects."""
        parts = self._make_parts(risk_threshold=0.3, n_parts=20)

        config_low = DetectionConfig(base_sensitivity=0.3)
        config_high = DetectionConfig(base_sensitivity=0.95)

        # Override per-type to match base
        for t in config_low.sensitivity_by_type:
            config_low.sensitivity_by_type[t] = 0.3
            config_high.sensitivity_by_type[t] = 0.95

        det_low = AIDefectDetector(config_low, np.random.default_rng(42))
        det_high = AIDefectDetector(config_high, np.random.default_rng(42))

        missed_low = sum(len(det_low.inspect_part(p).missed) for p in parts)
        missed_high = sum(len(det_high.inspect_part(p).missed) for p in parts)

        assert missed_high < missed_low

    def test_high_specificity_fewer_false_alarms(self):
        """Higher specificity should produce fewer false alarms."""
        parts = self._make_parts(risk_threshold=0.7, n_parts=20)

        config_low = DetectionConfig(base_specificity=0.5)
        config_high = DetectionConfig(base_specificity=0.99)

        det_low = AIDefectDetector(config_low, np.random.default_rng(42))
        det_high = AIDefectDetector(config_high, np.random.default_rng(42))

        fa_low = sum(det_low.inspect_part(p).false_alarms for p in parts)
        fa_high = sum(det_high.inspect_part(p).false_alarms for p in parts)

        assert fa_high <= fa_low

    def test_confidence_in_range(self):
        det_config = DetectionConfig()
        detector = AIDefectDetector(det_config, np.random.default_rng(42))
        parts = self._make_parts(risk_threshold=0.4, n_parts=10)

        for part in parts:
            result = detector.inspect_part(part)
            for det in result.detections:
                assert 0.0 <= det.confidence <= 1.0

    def test_reproducibility(self):
        det_config = DetectionConfig()
        parts = self._make_parts()

        r1 = AIDefectDetector(det_config, np.random.default_rng(99)).inspect_part(parts[0])
        r2 = AIDefectDetector(det_config, np.random.default_rng(99)).inspect_part(parts[0])

        assert len(r1.detections) == len(r2.detections)
        assert len(r1.missed) == len(r2.missed)
        assert r1.false_alarms == r2.false_alarms
