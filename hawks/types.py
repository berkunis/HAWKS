"""Shared domain data types for HAWKS simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DefectInstance:
    """A single ground-truth defect in a manufactured part."""

    defect_type: str  # "porosity" | "cracking" | "delamination" | "geometric"
    severity: float  # 0.0–1.0
    location: tuple[int, int]  # (layer_index, spatial_position)


@dataclass
class Detection:
    """An AI detector's assessment of a potential defect."""

    defect_ref: DefectInstance | None  # None for false positives
    confidence: float  # 0.0–1.0
    flagged: bool  # above threshold?


@dataclass(frozen=True)
class LayerResult:
    """Ground-truth defect information for a single layer."""

    layer_index: int
    defects: list[DefectInstance] = field(default_factory=list)


@dataclass(frozen=True)
class PartResult:
    """Complete ground-truth result for a manufactured part."""

    part_id: str
    layers: list[LayerResult] = field(default_factory=list)
    total_defect_count: int = 0


@dataclass(frozen=True)
class DetectionResult:
    """AI detector output for a single part."""

    part_id: str
    detections: list[Detection] = field(default_factory=list)
    missed: list[DefectInstance] = field(default_factory=list)
    false_alarms: int = 0


@dataclass(frozen=True)
class ModelPrediction:
    """AI model's probabilistic prediction for a single part."""

    part_id: str
    predicted_defect_probability: float
    confidence_score: float
    predicted_positive: bool
    correct: bool


@dataclass(frozen=True)
class OperatorDecision:
    """A human operator's decision on a flagged detection."""

    operator_id: str
    part_id: str
    detection: Detection
    action: str  # "accept" | "reject" | "escalate"
    correct: bool  # vs ground truth
    response_time: float
    trust_level: float  # operator's trust at decision time
    fatigue_level: float  # operator's fatigue at decision time


@dataclass(frozen=True)
class TimeStep:
    """Discrete simulation time step."""

    step: int
    shift_hour: float  # hours into current shift
