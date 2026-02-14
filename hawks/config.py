"""Configuration dataclasses for HAWKS simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ManufacturingConfig:
    num_layers_per_part: int = 100

    # Geometry
    layer_height_mm: float = 0.2

    # Temperature setpoints
    nozzle_temp_setpoint: float = 260.0
    bed_temp_setpoint: float = 70.0
    ambient_temp_initial: float = 25.0

    # Dynamics
    nozzle_drift_rate: float = 0.02
    bed_decay_rate: float = 0.015

    # Noise standard deviations
    nozzle_noise_std: float = 1.5
    bed_noise_std: float = 0.8
    ambient_noise_std: float = 0.3
    vibration_noise_std: float = 0.1
    extrusion_noise_std: float = 0.05

    # Ambient bounds
    ambient_temp_min: float = 20.0
    ambient_temp_max: float = 35.0

    # Vibration model
    vibration_base: float = 0.5
    vibration_height_coeff: float = 0.03

    # Cooling model
    cooling_gamma: float = 0.1
    cooling_height_damping: float = 0.05

    # Adhesion sigmoid weights
    adhesion_w_nozzle: float = 0.02
    adhesion_w_bed: float = 0.03
    adhesion_w_cooling: float = 0.01
    adhesion_w_extrusion: float = 2.0
    adhesion_w_vibration: float = 1.5
    adhesion_bias: float = -5.0

    # Stress accumulation weights
    stress_w_thermal: float = 0.3
    stress_w_adhesion: float = 0.3
    stress_w_vibration: float = 0.2
    stress_w_height: float = 0.2

    # Risk sigmoid weights
    risk_w_thermal: float = 2.0
    risk_w_adhesion: float = 2.5
    risk_w_vibration: float = 1.5
    risk_w_height: float = 1.0
    risk_w_stress: float = 1.5
    risk_bias: float = -5.0

    # Normalization constants
    nozzle_norm_std: float = 3.0
    vibration_normalizer: float = 1.2
    thermal_rolling_window: int = 5

    # Defect threshold
    risk_threshold: float = 0.5


@dataclass
class DetectionConfig:
    base_sensitivity: float = 0.85
    base_specificity: float = 0.90
    confidence_noise_std: float = 0.1
    sensitivity_by_type: dict[str, float] = field(
        default_factory=lambda: {
            "porosity": 0.90,
            "cracking": 0.85,
            "delamination": 0.80,
            "geometric": 0.75,
        }
    )
    severity_sensitivity_curve: str = "sigmoid"


@dataclass
class OperatorConfig:
    trust_initial: tuple[float, float] = (0.5, 0.15)
    fatigue_rate: float = 0.01
    skill_level: tuple[float, float] = (0.7, 0.2)
    automation_bias_strength: float = 0.3
    vigilance_decrement_rate: float = 0.005


@dataclass
class PopulationConfig:
    num_operators: int = 20
    operator_defaults: OperatorConfig = field(default_factory=OperatorConfig)
    shift_duration_hours: float = 8.0
    assignment_policy: str = "round_robin"


@dataclass
class ClockConfig:
    num_steps: int = 1000
    parts_per_step: int = 5


@dataclass
class HAWKSConfig:
    manufacturing: ManufacturingConfig = field(default_factory=ManufacturingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)
    master_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> HAWKSConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        manufacturing = ManufacturingConfig(**data.get("manufacturing", {}))
        detection = DetectionConfig(**data.get("detection", {}))

        pop_data = data.get("population", {})
        op_defaults = pop_data.pop("operator_defaults", {})
        if op_defaults:
            # Convert trust_initial and skill_level lists to tuples
            for key in ("trust_initial", "skill_level"):
                if key in op_defaults:
                    op_defaults[key] = tuple(op_defaults[key])
            pop_data["operator_defaults"] = OperatorConfig(**op_defaults)
        population = PopulationConfig(**pop_data)

        clock = ClockConfig(**data.get("clock", {}))
        master_seed = data.get("master_seed", 42)

        return cls(
            manufacturing=manufacturing,
            detection=detection,
            population=population,
            clock=clock,
            master_seed=master_seed,
        )
