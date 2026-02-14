"""Simulation engine — the orchestrator that runs the per-step pipeline."""

from __future__ import annotations

from hawks.config import HAWKSConfig
from hawks.detection.ai_model import AIModelSimulator
from hawks.engine.clock import SimulationClock
from hawks.engine.metrics import MetricsCollector
from hawks.manufacturing.digital_twin import ManufacturingTwin
from hawks.operators.population import OperatorPopulation
from hawks.seed import SeedManager
from hawks.types import OperatorDecision, TimeStep


class SimulationEngine:
    """Orchestrator that runs the per-step simulation pipeline.

    Per-step pipeline:
    1. ManufacturingTwin.produce_parts(n)     -> list[PartResult]
    2. AIModelSimulator.inspect_part(part, trace) -> (ModelPrediction, DetectionResult)
    3. OperatorPopulation.assign_operator()    -> HumanOperator
    4. HumanOperator.review_detection(det)     -> list[OperatorDecision]
    5. HumanOperator.receive_feedback(...)     (trust update)
    6. MetricsCollector.record_step(...)       (recording)
    7. All operators: advance_time(time)       (fatigue/vigilance)
    """

    def __init__(self, config: HAWKSConfig) -> None:
        self._config = config
        self._seed_manager = SeedManager(config.master_seed)

        # Build all modules with independent RNGs
        self._twin = ManufacturingTwin(
            config.manufacturing,
            self._seed_manager.spawn("manufacturing"),
        )
        self._model = AIModelSimulator(
            config.detection,
            self._seed_manager.spawn("detection"),
        )
        self._population = OperatorPopulation(
            config.population,
            self._seed_manager.spawn("operators"),
        )
        self._population.create_population()

        self._clock = SimulationClock(
            config.clock,
            config.population.shift_duration_hours,
        )
        self._metrics = MetricsCollector()

    def run(self) -> MetricsCollector:
        """Execute full simulation loop and return collected metrics."""
        for _ in range(self._clock.total_steps):
            time = self._clock.tick()
            self.step(time)
        return self._metrics

    def step(self, time: TimeStep) -> None:
        """Execute a single simulation step."""
        # Handle shift changes
        if self._clock.is_shift_change():
            self._population.handle_shift_change()

        # 1. Produce parts
        parts = self._twin.produce_parts(self._clock.parts_per_step)
        traces = self._twin.get_last_traces()

        # 2. Inspect each part
        detection_results = []
        for part, trace in zip(parts, traces):
            _prediction, det_result = self._model.inspect_part(part, trace)
            detection_results.append(det_result)

        # 3-5. For each part, assign operator and review flagged detections
        all_decisions: list[OperatorDecision] = []

        for part, det_result, trace in zip(parts, detection_results, traces):
            flagged = [d for d in det_result.detections if d.flagged]
            if not flagged:
                continue

            operator = self._population.assign_operator(time)

            for detection in flagged:
                # 4. Operator reviews detection
                decision = operator.review_detection(
                    detection, part.part_id, time,
                    structural_risk=trace.max_structural_risk,
                )
                all_decisions.append(decision)

                # 5. Provide feedback for trust update
                ground_truth = detection.defect_ref is not None
                operator.receive_feedback(decision, ground_truth)

        # 6. Record metrics
        self._metrics.record_step(time, parts, detection_results, all_decisions)

        # 7. Advance time for all operators
        for op in self._population.get_all_operators():
            op.advance_time(time)
