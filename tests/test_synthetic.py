"""Tests for hawks.data.synthetic \u2014 SyntheticDataGenerator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hawks.config import AIModelConfig
from hawks.data.synthetic import SyntheticDataGenerator
from hawks.operators.population import ARCHETYPES

# ---------------------------------------------------------------------------
# TestSyntheticDataGenerator
# ---------------------------------------------------------------------------


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator.generate()."""

    def test_generate_shape(self) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=10, master_seed=42)
        df = gen.generate()
        assert df.shape == (40, 8)

    def test_generate_columns(self) -> None:
        gen = SyntheticDataGenerator(n_operators=2, n_steps=5, master_seed=42)
        df = gen.generate()
        expected = {
            "step",
            "operator_id",
            "archetype",
            "structural_risk",
            "ai_confidence",
            "ai_correctness",
            "operator_trust",
            "operator_decision",
        }
        assert set(df.columns) == expected

    def test_generate_no_nans(self) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=10, master_seed=42)
        df = gen.generate()
        assert df.isna().sum().sum() == 0

    def test_generate_value_bounds(self) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=50, master_seed=42)
        df = gen.generate()

        assert df["step"].min() >= 0
        assert df["step"].max() < 50

        assert df["structural_risk"].min() >= 0.0
        assert df["structural_risk"].max() <= 1.0

        assert df["ai_confidence"].min() >= 0.0
        assert df["ai_confidence"].max() <= 1.0

        assert df["operator_trust"].min() >= 0.0
        assert df["operator_trust"].max() <= 1.0

        assert set(df["operator_decision"].unique()).issubset({"accept", "reject"})

        assert set(df["ai_correctness"].unique()).issubset({True, False})

    def test_generate_dtypes(self) -> None:
        gen = SyntheticDataGenerator(n_operators=2, n_steps=5, master_seed=42)
        df = gen.generate()

        assert df["step"].dtype == np.int64
        assert df["structural_risk"].dtype == np.float64
        assert df["ai_confidence"].dtype == np.float64
        assert df["ai_correctness"].dtype == np.dtype("bool")
        assert df["operator_trust"].dtype == np.float64
        assert df["operator_id"].dtype == object
        assert df["archetype"].dtype == object
        assert df["operator_decision"].dtype == object

    def test_seed_reproducibility(self) -> None:
        gen1 = SyntheticDataGenerator(n_operators=4, n_steps=20, master_seed=42)
        gen2 = SyntheticDataGenerator(n_operators=4, n_steps=20, master_seed=42)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        gen1 = SyntheticDataGenerator(n_operators=4, n_steps=20, master_seed=42)
        gen2 = SyntheticDataGenerator(n_operators=4, n_steps=20, master_seed=99)
        df1 = gen1.generate()
        df2 = gen2.generate()
        # At least one numeric column should differ
        assert not df1["structural_risk"].equals(df2["structural_risk"])

    def test_default_archetype_mix(self) -> None:
        gen = SyntheticDataGenerator(n_operators=20, n_steps=1, master_seed=42)
        df = gen.generate()
        counts = df["archetype"].value_counts()
        # 20 / 4 archetypes = 5 each
        assert len(counts) == 4
        for count in counts.values:
            assert count == 5

    def test_default_archetype_mix_uneven(self) -> None:
        gen = SyntheticDataGenerator(n_operators=10, n_steps=1, master_seed=42)
        df = gen.generate()
        counts = df["archetype"].value_counts()
        assert len(counts) == 4
        # 10 / 4 = 2 remainder 2, so two archetypes get 3 and two get 2
        sorted_counts = sorted(counts.values, reverse=True)
        assert sorted_counts == [3, 3, 2, 2]

    def test_custom_archetype_mix(self) -> None:
        mix = {
            "conservative_skeptic": 2,
            "calibrated_professional": 3,
            "automation_biased": 1,
            "algorithm_averse": 4,
        }
        gen = SyntheticDataGenerator(
            n_operators=10, n_steps=1, archetype_mix=mix, master_seed=42
        )
        df = gen.generate()
        counts = df["archetype"].value_counts().to_dict()
        assert counts["conservative_skeptic"] == 2
        assert counts["calibrated_professional"] == 3
        assert counts["automation_biased"] == 1
        assert counts["algorithm_averse"] == 4

    def test_invalid_archetype_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown archetype"):
            SyntheticDataGenerator(
                n_operators=5,
                n_steps=10,
                archetype_mix={"nonexistent_type": 5},
            )

    def test_archetype_mix_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            SyntheticDataGenerator(
                n_operators=10,
                n_steps=10,
                archetype_mix={
                    "conservative_skeptic": 3,
                    "calibrated_professional": 3,
                },
            )

    def test_invalid_ai_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            SyntheticDataGenerator(
                n_operators=5, n_steps=10, ai_preset="nonexistent_preset"
            )


# ---------------------------------------------------------------------------
# TestToTensors
# ---------------------------------------------------------------------------


class TestToTensors:
    """Tests for SyntheticDataGenerator.to_tensors()."""

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=10, master_seed=42)
        return gen.generate()

    def test_to_tensors_shapes(self, sample_df: pd.DataFrame) -> None:
        import torch

        result = SyntheticDataGenerator.to_tensors(sample_df)
        n = len(sample_df)
        assert result["features"].shape == (n, 4)
        assert result["decisions"].shape == (n,)
        assert result["archetypes"].shape == (n,)

    def test_to_tensors_dtypes(self, sample_df: pd.DataFrame) -> None:
        import torch

        result = SyntheticDataGenerator.to_tensors(sample_df)
        assert result["features"].dtype == torch.float32
        assert result["decisions"].dtype == torch.int64
        assert result["archetypes"].dtype == torch.int64

    def test_to_tensors_encoding_consistency(self, sample_df: pd.DataFrame) -> None:
        result = SyntheticDataGenerator.to_tensors(sample_df)
        encodings = result["encodings"]

        assert "decisions" in encodings
        assert "archetypes" in encodings
        assert encodings["decisions"] == {"accept": 0, "reject": 1}

        # Archetype encoding keys should match unique archetypes in df
        assert set(encodings["archetypes"].keys()) == set(
            sample_df["archetype"].unique()
        )
        # Values should be consecutive ints starting from 0
        vals = sorted(encodings["archetypes"].values())
        assert vals == list(range(len(vals)))

    def test_to_tensors_values(self, sample_df: pd.DataFrame) -> None:
        import torch

        result = SyntheticDataGenerator.to_tensors(sample_df)

        # Decisions must be 0 or 1
        assert torch.all((result["decisions"] == 0) | (result["decisions"] == 1))

        # Archetypes must be in [0, n_unique)
        n_unique = len(sample_df["archetype"].unique())
        assert result["archetypes"].min() >= 0
        assert result["archetypes"].max() < n_unique

    def test_to_tensors_without_torch(self, sample_df: pd.DataFrame) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="PyTorch is required"):
                SyntheticDataGenerator.to_tensors(sample_df)


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Tests for SyntheticDataGenerator.metadata()."""

    def test_metadata_keys(self) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=10, master_seed=42)
        meta = gen.metadata()
        expected_keys = {
            "version",
            "generator",
            "master_seed",
            "n_operators",
            "n_steps",
            "n_interactions",
            "ai_preset",
            "ai_accuracy",
            "archetype_distribution",
            "archetype_params",
            "created_at",
        }
        assert set(meta.keys()) == expected_keys

    def test_metadata_seed_recorded(self) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=10, master_seed=99)
        meta = gen.metadata()
        assert meta["master_seed"] == 99

    def test_metadata_ai_accuracy_recorded(self) -> None:
        gen = SyntheticDataGenerator(
            n_operators=4, n_steps=10, ai_preset="well_calibrated", master_seed=42
        )
        meta = gen.metadata()
        ai_cfg = AIModelConfig(preset="well_calibrated")
        assert meta["ai_accuracy"]["true_positive_rate"] == ai_cfg.true_positive_rate
        assert meta["ai_accuracy"]["false_positive_rate"] == ai_cfg.false_positive_rate
        assert meta["ai_accuracy"]["calibration_bias"] == ai_cfg.calibration_bias
        assert meta["ai_accuracy"]["confidence_noise"] == ai_cfg.confidence_noise

    def test_metadata_archetype_distribution(self) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=5, master_seed=42)
        meta = gen.metadata()
        df = gen.generate()
        # Compare metadata distribution to actual DataFrame counts (per step)
        actual_counts = df.groupby("archetype")["step"].nunique()
        for archetype, count in meta["archetype_distribution"].items():
            # count is number of operators of this archetype
            # Each operator appears once per step, so total rows = count * n_steps
            assert len(df[df["archetype"] == archetype]) == count * 5


# ---------------------------------------------------------------------------
# TestSave
# ---------------------------------------------------------------------------


class TestSave:
    """Tests for SyntheticDataGenerator.save()."""

    def test_save_pt_contains_metadata(self, tmp_path: Path) -> None:
        import torch

        gen = SyntheticDataGenerator(n_operators=4, n_steps=5, master_seed=42)
        out = gen.save(tmp_path / "test.pt", fmt="pt")
        assert out.exists()
        data = torch.load(out, weights_only=False)
        assert "metadata" in data
        assert data["metadata"]["master_seed"] == 42

    def test_save_csv_creates_sidecar(self, tmp_path: Path) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=5, master_seed=42)
        out = gen.save(tmp_path / "test.csv", fmt="csv")
        assert out.exists()
        meta_path = tmp_path / "test.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["master_seed"] == 42

    def test_save_csv_roundtrip(self, tmp_path: Path) -> None:
        gen = SyntheticDataGenerator(n_operators=4, n_steps=5, master_seed=42)
        out = gen.save(tmp_path / "test.csv", fmt="csv")
        df = pd.read_csv(out)
        assert df.shape == (20, 8)
