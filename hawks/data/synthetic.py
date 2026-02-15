"""Synthetic dataset generator for human\u2013AI interaction data."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hawks.config import AIModelConfig, ManufacturingConfig
from hawks.detection.ai_model import AIModelSimulator
from hawks.manufacturing.digital_twin import ManufacturingTwin
from hawks.operators.decision import DecisionModel
from hawks.operators.operator import HumanOperator
from hawks.operators.population import ARCHETYPES
from hawks.operators.trust import TrustModel
from hawks.seed import SeedManager
from hawks.types import Detection, TimeStep


class SyntheticDataGenerator:
    """Generate flat, ML-ready datasets from simulated human\u2013AI interactions.

    Wraps the HAWKS physics engine, AI model, and operator archetypes into
    a single loop that produces a Pandas DataFrame (or PyTorch tensors)
    with per-interaction records.
    """

    def __init__(
        self,
        n_operators: int = 20,
        n_steps: int = 1000,
        archetype_mix: dict[str, int] | None = None,
        ai_preset: str = "well_calibrated",
        master_seed: int = 42,
    ) -> None:
        if n_operators < 1:
            raise ValueError("n_operators must be >= 1")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        # Validate ai_preset by constructing config (raises on unknown)
        AIModelConfig(preset=ai_preset)

        if archetype_mix is not None:
            for key in archetype_mix:
                if key not in ARCHETYPES:
                    raise ValueError(
                        f"Unknown archetype '{key}'. "
                        f"Valid archetypes: {sorted(ARCHETYPES.keys())}"
                    )
            if sum(archetype_mix.values()) != n_operators:
                raise ValueError(
                    f"archetype_mix values sum to {sum(archetype_mix.values())}, "
                    f"but n_operators is {n_operators}"
                )

        self._n_operators = n_operators
        self._n_steps = n_steps
        self._archetype_mix = archetype_mix
        self._ai_preset = ai_preset
        self._master_seed = master_seed

    def _build_archetype_assignments(self) -> list[str]:
        """Return a list of archetype names, one per operator."""
        if self._archetype_mix is not None:
            assignments: list[str] = []
            for name, count in self._archetype_mix.items():
                assignments.extend([name] * count)
            return assignments

        # Default: distribute equally across all 4 archetypes
        archetype_names = sorted(ARCHETYPES.keys())
        n = self._n_operators
        base, extra = divmod(n, len(archetype_names))
        assignments = []
        for i, name in enumerate(archetype_names):
            count = base + (1 if i < extra else 0)
            assignments.extend([name] * count)
        return assignments

    def _resolved_archetype_mix(self) -> dict[str, int]:
        """Return the actual {archetype: count} dict for the current config."""
        return dict(Counter(self._build_archetype_assignments()))

    def _create_operators(
        self, assignments: list[str], rng: np.random.Generator
    ) -> list[tuple[HumanOperator, str]]:
        """Create operators following the OperatorPopulation pattern."""
        operators: list[tuple[HumanOperator, str]] = []
        for i, archetype_name in enumerate(assignments):
            params = dict(ARCHETYPES[archetype_name])

            trust_model = TrustModel(
                initial_trust=params["initial_trust"],
                alpha=params["alpha"],
                beta=params["beta"],
            )

            op_rng = np.random.default_rng(rng.integers(0, 2**32))

            decision_model = DecisionModel(
                trust_weight=params["trust_weight"],
                confidence_weight=params["confidence_weight"],
                risk_tolerance=params["risk_tolerance"],
                decision_noise=params["decision_noise"],
                rng=op_rng,
            )

            operator = HumanOperator(
                operator_id=f"OP-{i + 1:03d}",
                trust_model=trust_model,
                decision_model=decision_model,
                rng=op_rng,
            )
            operators.append((operator, archetype_name))
        return operators

    def metadata(self) -> dict[str, Any]:
        """Return a dict capturing every parameter needed to reproduce the dataset."""
        ai_cfg = AIModelConfig(preset=self._ai_preset)
        return {
            "version": "1.0",
            "generator": "SyntheticDataGenerator",
            "master_seed": self._master_seed,
            "n_operators": self._n_operators,
            "n_steps": self._n_steps,
            "n_interactions": self._n_operators * self._n_steps,
            "ai_preset": self._ai_preset,
            "ai_accuracy": {
                "true_positive_rate": ai_cfg.true_positive_rate,
                "false_positive_rate": ai_cfg.false_positive_rate,
                "calibration_bias": ai_cfg.calibration_bias,
                "confidence_noise": ai_cfg.confidence_noise,
            },
            "archetype_distribution": self._resolved_archetype_mix(),
            "archetype_params": {
                name: dict(params) for name, params in ARCHETYPES.items()
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate(self) -> pd.DataFrame:
        """Run the simulation and return a DataFrame with per-interaction records.

        Returns a DataFrame with n_operators * n_steps rows and 8 columns:
        step, operator_id, archetype, structural_risk, ai_confidence,
        ai_correctness, operator_trust, operator_decision.
        """
        seed_mgr = SeedManager(self._master_seed)

        twin = ManufacturingTwin(
            ManufacturingConfig(), seed_mgr.spawn("manufacturing")
        )
        ai_model = AIModelSimulator(
            AIModelConfig(preset=self._ai_preset), seed_mgr.spawn("detection")
        )

        op_rng = seed_mgr.spawn("operators")
        assignments = self._build_archetype_assignments()
        operators = self._create_operators(assignments, op_rng)

        records: list[dict[str, Any]] = []

        for step in range(self._n_steps):
            parts = twin.produce_parts(self._n_operators)
            traces = twin.get_last_traces()
            time = TimeStep(step=step, shift_hour=float(step % 8))

            for part, trace, (operator, archetype) in zip(
                parts, traces, operators
            ):
                prediction = ai_model.predict(part, trace)

                has_defects = part.total_defect_count > 0
                # Pick first defect as reference if any exist
                defect_ref = None
                if has_defects:
                    for layer in part.layers:
                        if layer.defects:
                            defect_ref = layer.defects[0]
                            break

                detection = Detection(
                    defect_ref=defect_ref,
                    confidence=prediction.confidence_score,
                    flagged=prediction.predicted_positive,
                )

                decision = operator.review_detection(
                    detection, part.part_id, time, trace.max_structural_risk
                )

                ground_truth = has_defects
                operator.receive_feedback(decision, ground_truth)

                records.append(
                    {
                        "step": step,
                        "operator_id": operator.operator_id,
                        "archetype": archetype,
                        "structural_risk": trace.max_structural_risk,
                        "ai_confidence": prediction.confidence_score,
                        "ai_correctness": prediction.correct,
                        "operator_trust": decision.trust_level,
                        "operator_decision": decision.action,
                    }
                )

        df = pd.DataFrame(records)
        # Enforce dtypes
        df["step"] = df["step"].astype("int64")
        df["structural_risk"] = df["structural_risk"].astype("float64")
        df["ai_confidence"] = df["ai_confidence"].astype("float64")
        df["ai_correctness"] = df["ai_correctness"].astype("bool")
        df["operator_trust"] = df["operator_trust"].astype("float64")

        df.attrs["hawks_metadata"] = self.metadata()
        return df

    @staticmethod
    def to_tensors(
        df: pd.DataFrame, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Convert a generated DataFrame to PyTorch tensors.

        Returns a dict with keys:
        - features: float32 tensor (N, 4) \u2014 structural_risk, ai_confidence,
          ai_correctness, operator_trust
        - decisions: int64 tensor (N,) \u2014 accept=0, reject=1
        - archetypes: int64 tensor (N,) \u2014 alphabetically sorted encoding
        - encodings: dict with label\u2192int mappings for decisions and archetypes
        - metadata: provenance dict (if provided or available in df.attrs)
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for to_tensors(). "
                "Install it with: pip install torch"
            )

        # Features
        features = torch.tensor(
            df[["structural_risk", "ai_confidence", "ai_correctness", "operator_trust"]]
            .astype("float32")
            .values,
            dtype=torch.float32,
        )

        # Decision encoding
        decision_encoding = {"accept": 0, "reject": 1}
        decisions = torch.tensor(
            df["operator_decision"].map(decision_encoding).values,
            dtype=torch.int64,
        )

        # Archetype encoding (alphabetically sorted)
        unique_archetypes = sorted(df["archetype"].unique())
        archetype_encoding = {name: i for i, name in enumerate(unique_archetypes)}
        archetypes = torch.tensor(
            df["archetype"].map(archetype_encoding).values,
            dtype=torch.int64,
        )

        result = {
            "features": features,
            "decisions": decisions,
            "archetypes": archetypes,
            "encodings": {
                "decisions": decision_encoding,
                "archetypes": archetype_encoding,
            },
        }

        # Attach metadata: explicit arg > df.attrs
        resolved_metadata = metadata or df.attrs.get("hawks_metadata")
        if resolved_metadata is not None:
            result["metadata"] = resolved_metadata

        return result

    def save(self, path: str | Path, fmt: str = "pt") -> Path:
        """Generate data and save to disk.

        Args:
            path: Output file path.
            fmt: ``"pt"`` for PyTorch tensors or ``"csv"`` for CSV + sidecar JSON.

        Returns:
            The path of the primary file written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.generate()

        if fmt == "pt":
            import torch

            tensors = self.to_tensors(df)
            torch.save(tensors, path)
        elif fmt == "csv":
            df.to_csv(path, index=False)
            meta_path = path.with_suffix(".meta.json")
            meta_path.write_text(
                json.dumps(df.attrs["hawks_metadata"], indent=2) + "\n"
            )
        else:
            raise ValueError(f"Unknown format '{fmt}'. Use 'pt' or 'csv'.")

        return path
