"""Tests for the manufacturing module."""

import numpy as np
import pytest

from hawks.config import ManufacturingConfig
from hawks.manufacturing.defect_gen import DefectGenerator
from hawks.manufacturing.digital_twin import ManufacturingTwin


class TestDefectGenerator:
    def test_sample_defects_returns_correct_layer_count(self):
        config = ManufacturingConfig(num_layers_per_part=50)
        rng = np.random.default_rng(42)
        gen = DefectGenerator(config, rng)

        result = gen.sample_defects(50)
        assert len(result) == 50

    def test_defect_types_are_valid(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        gen = DefectGenerator(config, rng)

        result = gen.sample_defects(100)
        valid_types = {"porosity", "cracking", "delamination", "geometric"}
        for layer_defects in result:
            for defect in layer_defects:
                assert defect.defect_type in valid_types

    def test_severity_in_range(self):
        config = ManufacturingConfig(defect_base_rate=0.5)  # higher rate for more samples
        rng = np.random.default_rng(42)
        gen = DefectGenerator(config, rng)

        result = gen.sample_defects(200)
        for layer_defects in result:
            for defect in layer_defects:
                assert 0.0 <= defect.severity <= 1.0

    def test_reproducibility(self):
        config = ManufacturingConfig()
        result1 = DefectGenerator(config, np.random.default_rng(99)).sample_defects(100)
        result2 = DefectGenerator(config, np.random.default_rng(99)).sample_defects(100)

        for l1, l2 in zip(result1, result2):
            assert len(l1) == len(l2)
            for d1, d2 in zip(l1, l2):
                assert d1 == d2

    def test_spatial_correlation_increases_clustering(self):
        """Higher spatial correlation should produce more consecutive defect layers."""
        rng_low = np.random.default_rng(42)
        rng_high = np.random.default_rng(42)

        config_low = ManufacturingConfig(spatial_correlation=0.0, defect_base_rate=0.05)
        config_high = ManufacturingConfig(spatial_correlation=0.9, defect_base_rate=0.05)

        n_trials = 10
        consecutive_low = 0
        consecutive_high = 0

        for _ in range(n_trials):
            result_low = DefectGenerator(config_low, rng_low).sample_defects(500)
            result_high = DefectGenerator(config_high, rng_high).sample_defects(500)

            for i in range(1, len(result_low)):
                if result_low[i - 1] and result_low[i]:
                    consecutive_low += 1
            for i in range(1, len(result_high)):
                if result_high[i - 1] and result_high[i]:
                    consecutive_high += 1

        assert consecutive_high >= consecutive_low


class TestManufacturingTwin:
    def test_produce_parts_count(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        twin = ManufacturingTwin(config, rng)

        parts = twin.produce_parts(3)
        assert len(parts) == 3

    def test_part_ids_are_unique(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        twin = ManufacturingTwin(config, rng)

        parts = twin.produce_parts(10)
        ids = [p.part_id for p in parts]
        assert len(set(ids)) == 10

    def test_part_has_correct_layer_count(self):
        config = ManufacturingConfig(num_layers_per_part=30)
        rng = np.random.default_rng(42)
        twin = ManufacturingTwin(config, rng)

        parts = twin.produce_parts(1)
        assert len(parts[0].layers) == 30

    def test_total_defect_count_matches_layers(self):
        config = ManufacturingConfig(defect_base_rate=0.1)
        rng = np.random.default_rng(42)
        twin = ManufacturingTwin(config, rng)

        parts = twin.produce_parts(5)
        for part in parts:
            actual = sum(len(layer.defects) for layer in part.layers)
            assert part.total_defect_count == actual
