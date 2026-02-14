"""Configuration dataclasses for HAWKS simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ManufacturingConfig:
    num_layers_per_part: int = 100
    defect_base_rate: float = 0.02
    defect_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "porosity": 0.4,
            "cracking": 0.25,
            "delamination": 0.2,
            "geometric": 0.15,
        }
    )
    spatial_correlation: float = 0.3
    severity_distribution: str = "beta"


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
