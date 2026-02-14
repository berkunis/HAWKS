"""Frozen dataclasses for physics-based layer simulation state."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerState:
    """Physical state variables for a single printed layer."""

    layer_index: int
    z_height: float
    nozzle_temp: float
    bed_temp: float
    ambient_temp: float
    vibration: float
    extrusion_error: float
    cooling_rate: float
    adhesion: float
    cumulative_stress: float


@dataclass(frozen=True)
class LayerRisk:
    """Structural risk assessment for a single layer."""

    layer_index: int
    thermal_instability: float
    adhesion_deficit: float
    vibration_norm: float
    height_fraction: float
    cumulative_stress: float
    structural_risk: float
    dominant_factor: str


@dataclass(frozen=True)
class PrintTrace:
    """Complete physics trace for one printed part."""

    part_id: str
    states: list[LayerState] = field(default_factory=list)
    risks: list[LayerRisk] = field(default_factory=list)
    max_structural_risk: float = 0.0
    total_layers: int = 0
