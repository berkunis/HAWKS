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
class AIModelConfig:
    true_positive_rate: float = 0.90
    false_positive_rate: float = 0.05
    calibration_bias: float = 0.0
    confidence_noise: float = 0.05
    preset: str | None = None

    _PRESETS: dict[str, dict[str, float]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        presets = {
            "well_calibrated": {
                "true_positive_rate": 0.90,
                "false_positive_rate": 0.05,
                "calibration_bias": 0.0,
                "confidence_noise": 0.05,
            },
            "overconfident": {
                "true_positive_rate": 0.85,
                "false_positive_rate": 0.10,
                "calibration_bias": 1.5,
                "confidence_noise": 0.02,
            },
            "underconfident": {
                "true_positive_rate": 0.90,
                "false_positive_rate": 0.05,
                "calibration_bias": -0.5,
                "confidence_noise": 0.15,
            },
        }
        if self.preset is not None:
            if self.preset not in presets:
                raise ValueError(
                    f"Unknown preset '{self.preset}'. "
                    f"Valid presets: {list(presets.keys())}"
                )
            for attr, value in presets[self.preset].items():
                object.__setattr__(self, attr, value)


@dataclass
class PopulationConfig:
    num_operators: int = 20
    shift_duration_hours: float = 8.0
    assignment_policy: str = "round_robin"
    archetype_mix: dict[str, int] | None = None
    parameter_noise: dict[str, float] | None = None


@dataclass
class ClockConfig:
    num_steps: int = 1000
    parts_per_step: int = 5


@dataclass
class HAWKSConfig:
    manufacturing: ManufacturingConfig = field(default_factory=ManufacturingConfig)
    detection: AIModelConfig = field(default_factory=AIModelConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)
    master_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> HAWKSConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        manufacturing = ManufacturingConfig(**data.get("manufacturing", {}))
        detection = AIModelConfig(**data.get("detection", {}))

        population = PopulationConfig(**data.get("population", {}))

        clock = ClockConfig(**data.get("clock", {}))
        master_seed = data.get("master_seed", 42)

        return cls(
            manufacturing=manufacturing,
            detection=detection,
            population=population,
            clock=clock,
            master_seed=master_seed,
        )
