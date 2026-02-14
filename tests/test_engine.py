"""Tests for the engine module — simulation, clock, metrics, and integration."""

import numpy as np
import pandas as pd
import pytest

from hawks.config import ClockConfig, HAWKSConfig
from hawks.engine.clock import SimulationClock
from hawks.engine.metrics import MetricsCollector
from hawks.engine.simulation import SimulationEngine
from hawks.types import (
    DefectInstance,
    Detection,
    DetectionResult,
    LayerResult,
    OperatorDecision,
    PartResult,
    TimeStep,
)


class TestSimulationClock:
    def test_tick_advances_step(self):
        config = ClockConfig(num_steps=100, parts_per_step=5)
        clock = SimulationClock(config, shift_duration_hours=8.0)

        ts = clock.tick()
        assert ts.step == 0
        ts = clock.tick()
        assert ts.step == 1

    def test_shift_hour_wraps(self):
        config = ClockConfig(num_steps=10, parts_per_step=5)
        clock = SimulationClock(config, shift_duration_hours=8.0)

        hours = []
        for _ in range(10):
            ts = clock.tick()
            hours.append(ts.shift_hour)

        # Should wrap around after reaching shift duration
        assert all(0 <= h < 8.0 for h in hours)


class TestMetricsCollector:
    def test_record_and_to_dataframe(self):
        collector = MetricsCollector()

        defect = DefectInstance("porosity", 0.5, (0, 100))
        detection = Detection(defect_ref=defect, confidence=0.8, flagged=True)
        part = PartResult(
            part_id="P1",
            layers=[LayerResult(layer_index=0, defects=[defect])],
            total_defect_count=1,
        )
        det_result = DetectionResult(
            part_id="P1", detections=[detection], missed=[], false_alarms=0
        )
        decision = OperatorDecision(
            operator_id="OP-001",
            part_id="P1",
            detection=detection,
            action="accept",
            correct=True,
            response_time=2.5,
            trust_level=0.6,
            fatigue_level=0.1,
        )

        ts = TimeStep(step=0, shift_hour=0.0)
        collector.record_step(ts, [part], [det_result], [decision])

        df = collector.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["part_id"] == "P1"
        assert df.iloc[0]["correct"] == True

    def test_summary_returns_metrics(self):
        collector = MetricsCollector()

        defect = DefectInstance("porosity", 0.5, (0, 100))
        detection = Detection(defect_ref=defect, confidence=0.8, flagged=True)
        part = PartResult(
            part_id="P1",
            layers=[LayerResult(layer_index=0, defects=[defect])],
            total_defect_count=1,
        )
        det_result = DetectionResult(
            part_id="P1", detections=[detection], missed=[], false_alarms=0
        )
        decision = OperatorDecision(
            operator_id="OP-001",
            part_id="P1",
            detection=detection,
            action="accept",
            correct=True,
            response_time=2.5,
            trust_level=0.6,
            fatigue_level=0.1,
        )

        ts = TimeStep(step=0, shift_hour=0.0)
        collector.record_step(ts, [part], [det_result], [decision])

        summary = collector.summary()
        assert "total_parts" in summary
        assert "operator_accuracy" in summary


class TestSimulationEngine:
    def test_smoke_test_small_run(self):
        """Full pipeline with small config — should produce valid DataFrame."""
        config = HAWKSConfig()
        config.clock.num_steps = 10
        config.clock.parts_per_step = 3
        config.population.num_operators = 5

        engine = SimulationEngine(config)
        metrics = engine.run()

        df = metrics.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "step" in df.columns
        assert "part_id" in df.columns
        # No NaN in step or part_id
        assert df["step"].notna().all()
        assert df["part_id"].notna().all()

    def test_seed_reproducibility(self):
        """Same config + seed should produce identical results."""
        config = HAWKSConfig()
        config.clock.num_steps = 20
        config.clock.parts_per_step = 3
        config.population.num_operators = 5

        metrics1 = SimulationEngine(config).run()
        metrics2 = SimulationEngine(config).run()

        df1 = metrics1.to_dataframe()
        df2 = metrics2.to_dataframe()

        assert len(df1) == len(df2)
        # Check key columns match
        pd.testing.assert_frame_equal(df1, df2)

    def test_summary_sanity(self):
        """Summary metrics should be reasonable."""
        config = HAWKSConfig()
        config.clock.num_steps = 50
        config.clock.parts_per_step = 5
        config.population.num_operators = 10

        metrics = SimulationEngine(config).run()
        summary = metrics.summary()

        assert summary["total_parts"] > 0
        assert summary["total_steps"] == 50

    def test_step_by_step_execution(self):
        """Running step-by-step should work for debugging."""
        config = HAWKSConfig()
        config.clock.num_steps = 5
        config.clock.parts_per_step = 2
        config.population.num_operators = 3

        engine = SimulationEngine(config)
        for _ in range(5):
            ts = engine._clock.tick()
            engine.step(ts)

        df = engine._metrics.to_dataframe()
        assert len(df) > 0
