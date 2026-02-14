"""Tests for the manufacturing module."""

import numpy as np
import pytest

from hawks.config import ManufacturingConfig
from hawks.manufacturing.physics_engine import PhysicsEngine
from hawks.manufacturing.digital_twin import ManufacturingTwin


class TestPhysicsEngine:
    def test_layer_count(self):
        config = ManufacturingConfig(num_layers_per_part=50)
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        layer_defects, trace = engine.simulate_print("TEST-001")
        assert len(layer_defects) == 50
        assert trace.total_layers == 50
        assert len(trace.states) == 50
        assert len(trace.risks) == 50

    def test_defect_types_are_valid(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        layer_defects, _ = engine.simulate_print("TEST-001")
        valid_types = {"porosity", "cracking", "delamination", "geometric"}
        for layer in layer_defects:
            for defect in layer:
                assert defect.defect_type in valid_types

    def test_severity_in_range(self):
        config = ManufacturingConfig(risk_threshold=0.1)
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        layer_defects, _ = engine.simulate_print("TEST-001")
        for layer in layer_defects:
            for defect in layer:
                assert 0.0 <= defect.severity <= 1.0

    def test_reproducibility(self):
        config = ManufacturingConfig()
        defects1, trace1 = PhysicsEngine(config, np.random.default_rng(99)).simulate_print("P1")
        defects2, trace2 = PhysicsEngine(config, np.random.default_rng(99)).simulate_print("P1")

        # Identical defects
        for l1, l2 in zip(defects1, defects2):
            assert len(l1) == len(l2)
            for d1, d2 in zip(l1, l2):
                assert d1 == d2

        # Identical traces
        for s1, s2 in zip(trace1.states, trace2.states):
            assert s1 == s2
        for r1, r2 in zip(trace1.risks, trace2.risks):
            assert r1 == r2

    def test_z_height_monotonically_increasing(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        _, trace = engine.simulate_print("TEST-001")
        for i in range(1, len(trace.states)):
            assert trace.states[i].z_height > trace.states[i - 1].z_height

    def test_stress_monotonically_increasing(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        _, trace = engine.simulate_print("TEST-001")
        for i in range(1, len(trace.states)):
            assert trace.states[i].cumulative_stress >= trace.states[i - 1].cumulative_stress

    def test_adhesion_in_range(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        _, trace = engine.simulate_print("TEST-001")
        for state in trace.states:
            assert 0.0 <= state.adhesion <= 1.0

    def test_risk_in_range(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        _, trace = engine.simulate_print("TEST-001")
        for risk in trace.risks:
            assert 0.0 <= risk.structural_risk <= 1.0

    def test_defects_only_where_risk_exceeds_threshold(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        layer_defects, trace = engine.simulate_print("TEST-001")
        for i, (defects, risk) in enumerate(zip(layer_defects, trace.risks)):
            if defects:
                assert risk.structural_risk > config.risk_threshold
            if risk.structural_risk <= config.risk_threshold:
                assert len(defects) == 0

    def test_higher_threshold_fewer_defects(self):
        config_low = ManufacturingConfig(risk_threshold=0.3)
        config_high = ManufacturingConfig(risk_threshold=0.7)

        defects_low, _ = PhysicsEngine(config_low, np.random.default_rng(42)).simulate_print("P1")
        defects_high, _ = PhysicsEngine(config_high, np.random.default_rng(42)).simulate_print("P1")

        count_low = sum(len(d) for d in defects_low)
        count_high = sum(len(d) for d in defects_high)
        assert count_high <= count_low

    def test_dominant_factor_is_valid(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        engine = PhysicsEngine(config, rng)

        _, trace = engine.simulate_print("TEST-001")
        valid_factors = {
            "thermal_instability",
            "adhesion_deficit",
            "vibration_norm",
            "height_fraction",
        }
        for risk in trace.risks:
            assert risk.dominant_factor in valid_factors


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
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        twin = ManufacturingTwin(config, rng)

        parts = twin.produce_parts(5)
        for part in parts:
            actual = sum(len(layer.defects) for layer in part.layers)
            assert part.total_defect_count == actual

    def test_get_last_traces(self):
        config = ManufacturingConfig()
        rng = np.random.default_rng(42)
        twin = ManufacturingTwin(config, rng)

        parts = twin.produce_parts(3)
        traces = twin.get_last_traces()
        assert len(traces) == 3
        for trace, part in zip(traces, parts):
            assert trace.part_id == part.part_id
            assert trace.total_layers == config.num_layers_per_part
