"""Physics-based layer simulation for vertical NylonX carbon fiber printing."""

from __future__ import annotations

from collections import deque

import numpy as np

from hawks.config import ManufacturingConfig
from hawks.manufacturing.physics_state import LayerRisk, LayerState, PrintTrace
from hawks.types import DefectInstance

# Dominant risk factor → defect type mapping
_FACTOR_TO_DEFECT: dict[str, str] = {
    "thermal_instability": "porosity",
    "adhesion_deficit": "delamination",
    "vibration_norm": "geometric",
    "height_fraction": "cracking",
}


def _stable_sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


class PhysicsEngine:
    """Layer-by-layer physics simulation for NylonX printing.

    Tracks 9 state variables per layer, computes structural risk from
    thermal variance / adhesion instability / vibration / height, and
    derives defects from the physics rather than random coin flips.
    """

    def __init__(self, config: ManufacturingConfig, rng: np.random.Generator) -> None:
        self._cfg = config
        self._rng = rng

    def simulate_print(
        self, part_id: str
    ) -> tuple[list[list[DefectInstance]], PrintTrace]:
        """Run a full layer-by-layer physics simulation for one part.

        Returns
        -------
        layer_defects : list[list[DefectInstance]]
            Per-layer defect lists (same contract as DefectGenerator.sample_defects).
        trace : PrintTrace
            Full physics trace for diagnostics / inspection.
        """
        cfg = self._cfg
        num_layers = cfg.num_layers_per_part
        z_max = num_layers * cfg.layer_height_mm

        # Accumulators
        states: list[LayerState] = []
        risks: list[LayerRisk] = []
        layer_defects: list[list[DefectInstance]] = []

        # Rolling window for thermal instability
        nozzle_history: deque[float] = deque(maxlen=cfg.thermal_rolling_window)

        # Initial conditions
        ambient_temp = cfg.ambient_temp_initial
        cumulative_stress = 0.0

        for i in range(num_layers):
            # --- State evolution ---
            z = (i + 1) * cfg.layer_height_mm

            # 5 RNG draws per layer: nozzle, bed, ambient, vibration, extrusion
            nozzle_noise = self._rng.normal(0.0, cfg.nozzle_noise_std)
            bed_noise = self._rng.normal(0.0, cfg.bed_noise_std)
            ambient_noise = self._rng.normal(0.0, cfg.ambient_noise_std)
            vibration_noise = self._rng.normal(0.0, cfg.vibration_noise_std)
            extrusion_error = self._rng.normal(0.0, cfg.extrusion_noise_std)

            nozzle_temp = cfg.nozzle_temp_setpoint + cfg.nozzle_drift_rate * i + nozzle_noise
            bed_temp = cfg.bed_temp_setpoint * np.exp(-cfg.bed_decay_rate * z) + bed_noise
            ambient_temp = np.clip(
                ambient_temp + ambient_noise,
                cfg.ambient_temp_min,
                cfg.ambient_temp_max,
            )
            vibration = max(
                0.0,
                cfg.vibration_base * (1.0 + cfg.vibration_height_coeff * z) + vibration_noise,
            )

            # Derived quantities
            cooling_rate = (
                cfg.cooling_gamma * (nozzle_temp - ambient_temp)
                / (1.0 + cfg.cooling_height_damping * z)
            )

            adhesion = _stable_sigmoid(
                cfg.adhesion_w_nozzle * nozzle_temp
                + cfg.adhesion_w_bed * bed_temp
                - cfg.adhesion_w_cooling * cooling_rate
                - cfg.adhesion_w_extrusion * abs(extrusion_error)
                - cfg.adhesion_w_vibration * vibration
                + cfg.adhesion_bias
            )

            # Stress accumulation
            if i > 0:
                delta_nozzle = abs(nozzle_temp - states[i - 1].nozzle_temp)
            else:
                delta_nozzle = 0.0

            cumulative_stress += (
                cfg.stress_w_thermal * delta_nozzle
                + cfg.stress_w_adhesion * (1.0 - adhesion)
                + cfg.stress_w_vibration * vibration
                + cfg.stress_w_height * (z / z_max)
            ) / num_layers

            state = LayerState(
                layer_index=i,
                z_height=z,
                nozzle_temp=nozzle_temp,
                bed_temp=bed_temp,
                ambient_temp=ambient_temp,
                vibration=vibration,
                extrusion_error=extrusion_error,
                cooling_rate=cooling_rate,
                adhesion=adhesion,
                cumulative_stress=cumulative_stress,
            )
            states.append(state)
            nozzle_history.append(nozzle_temp)

            # --- Structural risk ---
            if len(nozzle_history) >= 2:
                thermal_instability = float(np.std(list(nozzle_history))) / cfg.nozzle_norm_std
            else:
                thermal_instability = 0.0

            adhesion_deficit = 1.0 - adhesion
            vibration_norm = vibration / cfg.vibration_normalizer
            height_fraction = z / z_max

            structural_risk = _stable_sigmoid(
                cfg.risk_w_thermal * thermal_instability
                + cfg.risk_w_adhesion * adhesion_deficit
                + cfg.risk_w_vibration * vibration_norm
                + cfg.risk_w_height * height_fraction
                + cfg.risk_w_stress * cumulative_stress
                + cfg.risk_bias
            )

            # Dominant factor (weighted contributions to risk sigmoid)
            factors = {
                "thermal_instability": cfg.risk_w_thermal * thermal_instability,
                "adhesion_deficit": cfg.risk_w_adhesion * adhesion_deficit,
                "vibration_norm": cfg.risk_w_vibration * vibration_norm,
                "height_fraction": cfg.risk_w_height * height_fraction,
            }
            dominant_factor = max(factors, key=factors.get)  # type: ignore[arg-type]

            risk = LayerRisk(
                layer_index=i,
                thermal_instability=thermal_instability,
                adhesion_deficit=adhesion_deficit,
                vibration_norm=vibration_norm,
                height_fraction=height_fraction,
                cumulative_stress=cumulative_stress,
                structural_risk=structural_risk,
                dominant_factor=dominant_factor,
            )
            risks.append(risk)

            # --- Defect generation ---
            defects: list[DefectInstance] = []
            if structural_risk > cfg.risk_threshold:
                severity = np.clip(
                    (structural_risk - cfg.risk_threshold) / (1.0 - cfg.risk_threshold),
                    0.0,
                    1.0,
                )
                defect_type = _FACTOR_TO_DEFECT[dominant_factor]
                spatial_pos = int(self._rng.integers(0, 1000))
                defects.append(
                    DefectInstance(
                        defect_type=defect_type,
                        severity=float(severity),
                        location=(i, spatial_pos),
                    )
                )
            layer_defects.append(defects)

        max_risk = max(r.structural_risk for r in risks) if risks else 0.0

        trace = PrintTrace(
            part_id=part_id,
            states=states,
            risks=risks,
            max_structural_risk=max_risk,
            total_layers=num_layers,
        )

        return layer_defects, trace
